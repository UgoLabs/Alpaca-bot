"""
Fine-tune swing Gen7 on call-debit spread rewards (Alpaca historical marks).

1) Run: python scripts/download_options_bars.py
2) Then: python scripts/train_options_from_swing.py

Usage:
  .\\.venv\\Scripts\\python.exe scripts/train_options_from_swing.py
  .\\.venv\\Scripts\\python.exe scripts/train_options_from_swing.py --episodes 60 --freeze-ratio 0.3
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

from config.settings import OptionsTraderConfig, SWING_MODEL_PATH, TrainingConfig  # noqa: E402

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
    load_options_training_tensors,
)
from src.environments.vector_env_spread import VectorizedSpreadEnv  # noqa: E402

DEFAULT_WATCHLIST = os.path.join(
    "config", "watchlists", getattr(OptionsTraderConfig, "TRAIN_WATCHLIST", "options_liquid_200.txt")
)
MODEL_PREFIX = "options_from_swing"


def _swing_checkpoint_prefix() -> str:
    path = str(SWING_MODEL_PATH)
    for suffix in ("_balanced.pth", "_aggressive.pth", "_conservative.pth"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path.replace(".pth", "")


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
    log_dir: str = "logs/runs/options_from_swing",
    init_from_swing: bool = True,
    init_from: str | None = None,
    freeze_ratio: float = 0.3,
    initial_epsilon: float = 0.15,
    epsilon_decay: float = 0.995,
    resume_latest: bool = True,
    model_prefix: str = MODEL_PREFIX,
    checkpoint_every: int = 5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training options-from-swing on {device.upper()}")

    loaded = load_options_training_tensors(watchlist_path, min_date=OPTIONS_DATA_MIN_DATE)
    if loaded is None:
        print(f"No training tensors. Run download_options_bars.py first (marks in {MARKS_DIR}).")
        return

    data, prices, spread_marks, entry_premium, tradable, tickers = loaded
    print(f"Symbols: {len(tickers)} — {', '.join(tickers[:8])}{'…' if len(tickers) > 8 else ''}")
    print(f"Tensor shape: {tuple(data.shape)} (envs, time, features)")
    ckpt = max(1, int(checkpoint_every))
    print(f"Checkpoints: every {ckpt} ep + best + last -> models/{model_prefix}_*")

    env = VectorizedSpreadEnv(
        data, spread_marks, entry_premium, tradable, device=device,
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
        resolved_prefix = init_from
    elif resume_latest:
        resolved_prefix, start_ep = _find_latest_checkpoint(model_prefix)
    if not resolved_prefix and init_from_swing:
        resolved_prefix = _swing_checkpoint_prefix()

    if not resolved_prefix:
        print("No checkpoint to load.")
        return

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
            "watchlist": watchlist_path,
            "tickers": tickers,
            "num_features": num_features,
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

    for sa in agent.agents:
        sa.epsilon = float(initial_epsilon)
        sa.epsilon_decay = float(epsilon_decay)

    writer = SummaryWriter(log_dir=log_dir)
    seq_len = 64
    dummy_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long, device=device)
    dummy_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long, device=device)
    dummy_ids_np = dummy_ids[0].detach().cpu().numpy()
    dummy_mask_np = dummy_mask[0].detach().cpu().numpy()

    best_reward = float("-inf")
    meta_path = os.path.join("models", f"{model_prefix}_best.meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                best_reward = float(json.load(f).get("best_reward", best_reward))
        except Exception:
            pass

    last_prefix = os.path.join("models", f"{model_prefix}_last")

    for episode in range(start_ep, episodes):
        if freeze_until > 0 and episode == freeze_until:
            print(f"Unfreezing at episode {episode}")
            for sa in agent.agents:
                for p in sa.policy_net.parameters():
                    p.requires_grad = True
                sa.optimizer = torch.optim.Adam(sa.policy_net.parameters(), lr=1e-7)

        state = env.reset()
        total_reward = 0.0
        steps_per_episode = int(env.total_steps)
        if max_steps:
            steps_per_episode = min(steps_per_episode, int(max_steps))

        pbar = tqdm(range(steps_per_episode), desc=f"Ep{episode+1}/{episodes}", leave=False)
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
    ap = argparse.ArgumentParser(description="Fine-tune swing model on spread PnL")
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--train-every", type=int, default=32)
    ap.add_argument("--store-every", type=int, default=32)
    ap.add_argument("--log-dir", default="logs/runs/options_from_swing")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--no-swing-init", action="store_true")
    ap.add_argument("--no-resume-latest", action="store_true")
    ap.add_argument("--freeze-ratio", type=float, default=0.3)
    ap.add_argument("--initial-epsilon", type=float, default=0.15)
    ap.add_argument("--epsilon-decay", type=float, default=0.995)
    ap.add_argument("--model-prefix", default=MODEL_PREFIX)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save ep{N} every N episodes (always saves best + last each episode)",
    )
    args = ap.parse_args()

    train(
        watchlist_path=args.watchlist,
        episodes=args.episodes,
        max_steps=args.max_steps or None,
        train_every=args.train_every,
        store_every=args.store_every,
        log_dir=args.log_dir,
        init_from_swing=not args.no_swing_init,
        init_from=args.init_from or None,
        freeze_ratio=args.freeze_ratio,
        initial_epsilon=args.initial_epsilon,
        epsilon_decay=args.epsilon_decay,
        resume_latest=not args.no_resume_latest,
        model_prefix=args.model_prefix,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
