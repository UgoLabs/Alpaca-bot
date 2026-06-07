"""
Fine-tune a separate bearish options model (put debit / bear call credit).

BUY = open bearish structure (first tradable in priority order).
SELL = close — same 3-action head, different reward env than the bullish model.

Prereq: scripts/rebuild_multi_strategy_marks.py (bearish mark columns)

Usage:
  .\\.venv\\Scripts\\python.exe scripts/train_options_bearish_from_swing.py
  .\\.venv\\Scripts\\python.exe scripts/train_options_bearish_from_swing.py \\
      --init-from models/options_from_swing_200_ep30 --episodes 50
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OPTIONS_MODEL_PATH, OptionsTraderConfig, TrainingConfig  # noqa: E402

TrainingConfig.WINDOW_SIZE = 60
TrainingConfig.MAX_HOLD_BARS = 20
TrainingConfig.ENTRY_LOOKAHEAD_BARS = 5
TrainingConfig.ENTRY_REWARD_COEF = 0.5
TrainingConfig.INVALID_ACTION_PENALTY = 0.0003
TrainingConfig.SLIPPAGE_BPS = 1.5
TrainingConfig.TRANSACTION_COST_BPS = 0.0
TrainingConfig.FLAT_DOWNTREND_BONUS = 0.0

from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.data.options_historical import OPTIONS_DATA_MIN_DATE  # noqa: E402
from src.data.options_spread_dataset import (  # noqa: E402
    MARKS_DIR,
    load_options_multi_training_tensors,
)
from src.environments.vector_env_spread import VectorizedMultiStrategySpreadEnv  # noqa: E402

DEFAULT_WATCHLIST = os.path.join(
    "config", "watchlists", getattr(OptionsTraderConfig, "TRAIN_WATCHLIST", "options_liquid_200.txt")
)
MODEL_PREFIX = "options_bearish_from_swing"


def _normalize_prefix(path: str) -> str:
    path = str(path).replace("\\", "/")
    for suffix in ("_balanced.pth", "_aggressive.pth", "_conservative.pth"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path.replace(".pth", "") if path.endswith(".pth") else path


def _default_init_prefix() -> str:
    return _normalize_prefix(str(OPTIONS_MODEL_PATH))


def _find_latest_checkpoint(prefix: str) -> tuple[str | None, int]:
    pattern = os.path.join("models", f"{prefix}_ep*_balanced.pth")
    best_ep = 0
    best_path = None
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        marker = f"{prefix}_ep"
        try:
            ep = int(base.split(marker, 1)[1].split("_", 1)[0])
            if ep > best_ep:
                best_ep = ep
                best_path = os.path.join("models", f"{prefix}_ep{ep}")
        except Exception:
            continue
    return best_path, best_ep


def _save_meta(prefix: str, payload: dict) -> None:
    path = os.path.join("models", f"{prefix}_train.meta.json")
    os.makedirs("models", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train(
    *,
    watchlist_path: str,
    episodes: int = 50,
    max_steps: int | None = None,
    train_every: int = 32,
    store_every: int = 32,
    log_dir: str = "logs/runs/options_bearish_from_swing",
    init_from: str | None = None,
    freeze_ratio: float = 0.3,
    initial_epsilon: float = 0.15,
    epsilon_decay: float = 0.995,
    final_epsilon: float | None = None,
    resume_latest: bool = True,
    model_prefix: str = MODEL_PREFIX,
    checkpoint_every: int = 5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training bearish options model on {device.upper()}")

    loaded = load_options_multi_training_tensors(
        watchlist_path, mode="bearish", min_date=OPTIONS_DATA_MIN_DATE,
    )
    if loaded is None:
        print(f"No bearish tensors. Run rebuild_multi_strategy_marks.py first ({MARKS_DIR}).")
        return

    data, prices, spread_marks, entry_premium, tradable, is_credit, tickers = loaded
    print(f"Symbols: {len(tickers)} — {', '.join(tickers[:8])}{'…' if len(tickers) > 8 else ''}")
    print(f"Strategies: put_debit, bear_call_credit — shape {tuple(spread_marks.shape)}")

    env = VectorizedMultiStrategySpreadEnv(
        data, spread_marks, entry_premium, tradable, is_credit, device=device,
        strategy_order=(0, 1),
    )
    num_features = data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device,
    )

    resolved_prefix: str | None = None
    start_ep = 0
    if init_from:
        resolved_prefix = _normalize_prefix(init_from)
    elif resume_latest:
        resolved_prefix, start_ep = _find_latest_checkpoint(model_prefix)
    if not resolved_prefix:
        resolved_prefix = _default_init_prefix()

    try:
        agent.load(resolved_prefix)
        print(f"Loaded weights: {resolved_prefix}")
    except Exception as e:
        print(f"Load failed: {e}")
        return

    _save_meta(
        model_prefix,
        {
            "init_from": resolved_prefix,
            "mode": "bearish",
            "watchlist": watchlist_path,
            "tickers": tickers,
        },
    )

    freeze_until = int(episodes * freeze_ratio)
    if freeze_until > 0 and start_ep < freeze_until:
        agent.freeze_feature_extractors()
        for sa in agent.agents:
            sa.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, sa.policy_net.parameters()),
                lr=1e-6,
            )
    else:
        for sa in agent.agents:
            for p in sa.policy_net.parameters():
                p.requires_grad = True
            sa.optimizer = torch.optim.Adam(sa.policy_net.parameters(), lr=1e-7)

    use_eps_schedule = final_epsilon is not None
    if use_eps_schedule:
        eps_hi = float(initial_epsilon)
        eps_lo = float(final_epsilon)
        for sa in agent.agents:
            sa.epsilon = eps_hi
    else:
        for sa in agent.agents:
            sa.epsilon = float(initial_epsilon)

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    best_reward = float("-inf")
    last_prefix = os.path.join("models", f"{model_prefix}_last")
    meta_path = os.path.join("models", f"{model_prefix}_best.meta.json")
    ckpt = max(1, int(checkpoint_every))
    steps_per_episode = max_steps or env.total_steps - TrainingConfig.WINDOW_SIZE

    for episode in range(start_ep, episodes):
        if use_eps_schedule and episodes > 1:
            t = episode / max(episodes - 1, 1)
            eps = eps_hi + (eps_lo - eps_hi) * t
            for sa in agent.agents:
                sa.epsilon = eps

        state = env.reset()
        total_reward = 0.0
        dummy_ids = torch.zeros((env.num_envs, 64), dtype=torch.long, device=device)
        dummy_mask = torch.ones((env.num_envs, 64), dtype=torch.long, device=device)
        dummy_ids_np = dummy_ids[0].cpu().numpy()
        dummy_mask_np = dummy_mask[0].cpu().numpy()

        pbar = tqdm(range(steps_per_episode), desc=f"Bearish ep {episode+1}/{episodes}", leave=False)
        for step_idx in pbar:
            actions = agent.batch_act(state, dummy_ids, dummy_mask)
            next_state, rewards, dones, _ = env.step(actions)

            if (step_idx % store_every) == 0:
                state_cpu = state.detach().cpu().numpy()
                next_cpu = next_state.detach().cpu().numpy()
                act_cpu = actions.detach().cpu().numpy()
                rew_cpu = rewards.detach().cpu().numpy()
                done_cpu = dones.detach().cpu().numpy()
                for i in range(env.num_envs):
                    s = (state_cpu[i], dummy_ids_np, dummy_mask_np)
                    ns = (next_cpu[i], dummy_ids_np, dummy_mask_np)
                    agent.remember(s, act_cpu[i], rew_cpu[i], ns, done_cpu[i])

            if (step_idx % train_every) == 0:
                agent.train_step()

            state = next_state
            total_reward += rewards.sum().item()
            pbar.set_postfix({"R": f"{total_reward:.1f}", "eps": f"{agent.balanced.epsilon:.3f}"})

        pbar.close()
        writer.add_scalar("Reward/Episode_Total", total_reward, episode)
        print(f"Ep {episode+1}: reward={total_reward:.2f} eps={agent.balanced.epsilon:.4f}")

        if not use_eps_schedule:
            for sa in agent.agents:
                if sa.epsilon > sa.epsilon_min:
                    sa.epsilon *= epsilon_decay

        agent.save(last_prefix)
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join("models", f"{model_prefix}_best"))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"best_reward": best_reward, "episode": episode}, f)

        if (episode + 1) % ckpt == 0:
            agent.save(os.path.join("models", f"{model_prefix}_ep{episode+1}"))

    writer.close()
    print(f"Done. Best reward: {best_reward:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Train bearish options model (put debit / bear call)")
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--init-from", default="", help="Default: deployed OPTIONS_MODEL_PATH prefix")
    ap.add_argument("--no-resume-latest", action="store_true")
    ap.add_argument("--model-prefix", default=MODEL_PREFIX)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()

    train(
        watchlist_path=args.watchlist,
        episodes=args.episodes,
        max_steps=args.max_steps or None,
        init_from=args.init_from or None,
        resume_latest=not args.no_resume_latest,
        model_prefix=args.model_prefix,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
