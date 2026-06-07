"""
Fine-tune swing model on all options strategies in one env (bull + bear).

BUY opens first tradable in priority order:
  call_debit -> bull_put_credit -> long_call -> put_debit -> bear_call_credit

Default: init swing_gen7_refined_ep380, eps 0.7 -> 0.05 over 150 episodes.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/train_options_unified_from_swing.py
  .\\.venv\\Scripts\\python.exe scripts/train_options_unified_from_swing.py \\
      --episodes 150 --initial-epsilon 0.7 --final-epsilon 0.05
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
    load_options_multi_training_tensors,
)
from src.environments.vector_env_spread import VectorizedMultiStrategySpreadEnv  # noqa: E402

DEFAULT_WATCHLIST = os.path.join(
    "config", "watchlists", getattr(OptionsTraderConfig, "TRAIN_WATCHLIST", "options_liquid_200.txt")
)
DEFAULT_INIT = "models/swing_gen7_refined_ep380"
MODEL_PREFIX = "options_unified_gen380"
STRATEGY_NAMES = (
    "call_debit",
    "bull_put_credit",
    "long_call",
    "put_debit",
    "bear_call_credit",
)
STRATEGY_ORDER = tuple(range(len(STRATEGY_NAMES)))


def _normalize_prefix(path: str) -> str:
    path = str(path).replace("\\", "/")
    for suffix in ("_balanced.pth", "_aggressive.pth", "_conservative.pth"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path.replace(".pth", "") if path.endswith(".pth") else path


def _swing_checkpoint_prefix() -> str:
    return _normalize_prefix(str(SWING_MODEL_PATH))


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
    episodes: int = 150,
    max_steps: int | None = None,
    train_every: int = 32,
    store_every: int = 32,
    log_dir: str = "logs/runs/options_unified_gen380",
    init_from: str | None = None,
    freeze_ratio: float = 0.3,
    initial_epsilon: float = 0.7,
    final_epsilon: float = 0.05,
    resume_latest: bool = False,
    model_prefix: str = MODEL_PREFIX,
    checkpoint_every: int = 5,
    clear_replay: bool = False,
    sell_bias: float = 0.15,
    entry_reward_coef: float | None = None,
    policy_lr: float | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training unified options (bull+bear) on {device.upper()}")

    loaded = load_options_multi_training_tensors(
        watchlist_path, mode="all", min_date=OPTIONS_DATA_MIN_DATE,
    )
    if loaded is None:
        print(f"No training tensors. Run download_options_bars.py + rebuild_multi_strategy_marks.py")
        print(f"  marks dir: {MARKS_DIR}")
        return

    data, _prices, spread_marks, entry_premium, tradable, is_credit, tickers = loaded
    print(f"Symbols: {len(tickers)} — {', '.join(tickers[:8])}{'…' if len(tickers) > 8 else ''}")
    print(f"Tensor shape: {tuple(data.shape)} (envs, time, features)")
    print(f"Strategies: {', '.join(STRATEGY_NAMES)} — shape {tuple(spread_marks.shape)}")
    print(f"Credit flags: {is_credit.tolist()}")
    ckpt = max(1, int(checkpoint_every))
    print(f"Checkpoints: every {ckpt} ep + best + last -> models/{model_prefix}_*")

    env = VectorizedMultiStrategySpreadEnv(
        data, spread_marks, entry_premium, tradable, is_credit, device=device,
        strategy_order=STRATEGY_ORDER,
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
        resolved_prefix = _swing_checkpoint_prefix()

    try:
        agent.load(resolved_prefix)
        print(f"Loaded weights: {resolved_prefix}")
    except Exception as e:
        print(f"Load failed: {e}")
        return

    if clear_replay:
        agent.clear_memory()
        print("Cleared replay buffers (fresh fine-tune memory)")

    if entry_reward_coef is not None:
        TrainingConfig.ENTRY_REWARD_COEF = float(entry_reward_coef)
        print(f"ENTRY_REWARD_COEF={TrainingConfig.ENTRY_REWARD_COEF}")

    _save_meta(
        model_prefix,
        {
            "init_from": resolved_prefix,
            "watchlist": watchlist_path,
            "tickers": tickers,
            "num_features": num_features,
            "strategies": list(STRATEGY_NAMES),
            "mode": "all",
        },
    )

    lr_frozen = float(policy_lr) if policy_lr is not None else 1e-6
    lr_full = float(policy_lr) if policy_lr is not None else 1e-7

    freeze_until = int(episodes * freeze_ratio)
    if freeze_until > 0 and start_ep < freeze_until:
        agent.freeze_feature_extractors()
        for sa in agent.agents:
            sa.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, sa.policy_net.parameters()),
                lr=lr_frozen,
            )
    else:
        for sa in agent.agents:
            for p in sa.policy_net.parameters():
                p.requires_grad = True
            sa.optimizer = torch.optim.Adam(sa.policy_net.parameters(), lr=lr_full)

    eps_hi = float(initial_epsilon)
    eps_lo = float(final_epsilon)
    print(f"Epsilon schedule: {eps_hi:.3f} -> {eps_lo:.3f} linear over {episodes} episodes")
    print(f"Exploration sell_bias={sell_bias:.2f} (only applies when epsilon>0)")

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

    def _set_epsilon(ep_idx: int) -> float:
        if episodes <= 1:
            return eps_lo
        t = ep_idx / max(1, episodes - 1)
        return eps_hi + (eps_lo - eps_hi) * t

    for episode in range(start_ep, episodes):
        ep_val = _set_epsilon(episode)
        for sa in agent.agents:
            sa.epsilon = ep_val

        if freeze_until > 0 and episode == freeze_until and freeze_until < episodes:
            print(f"Unfreezing at episode {episode}")
            for sa in agent.agents:
                for p in sa.policy_net.parameters():
                    p.requires_grad = True
                sa.optimizer = torch.optim.Adam(sa.policy_net.parameters(), lr=lr_full)

        state = env.reset()
        total_reward = 0.0
        steps_per_episode = int(env.total_steps)
        if max_steps:
            steps_per_episode = min(steps_per_episode, int(max_steps))

        pbar = tqdm(range(steps_per_episode), desc=f"Ep{episode+1}/{episodes}", leave=False)
        for step_idx in pbar:
            actions = agent.batch_act(state, dummy_ids, dummy_mask, sell_bias=sell_bias)
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
    ap = argparse.ArgumentParser(description="Unified bull+bear options training from swing")
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--train-every", type=int, default=32)
    ap.add_argument("--store-every", type=int, default=32)
    ap.add_argument("--log-dir", default="logs/runs/options_unified_gen380")
    ap.add_argument("--init-from", default=DEFAULT_INIT, help="Checkpoint prefix (default: swing ep380)")
    ap.add_argument("--resume-latest", action="store_true", help="Resume options_unified_gen380_* instead of fresh init")
    ap.add_argument("--freeze-ratio", type=float, default=0.3)
    ap.add_argument("--initial-epsilon", type=float, default=0.7)
    ap.add_argument("--final-epsilon", type=float, default=0.05)
    ap.add_argument("--model-prefix", default=MODEL_PREFIX)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--clear-replay", action="store_true", help="Wipe replay buffers after loading checkpoint")
    ap.add_argument("--sell-bias", type=float, default=0.15,
                    help="Epsilon-explore SELL probability in batch_act (use 0.33 for uniform)")
    ap.add_argument("--entry-reward-coef", type=float, default=None)
    ap.add_argument("--policy-lr", type=float, default=None)
    ap.add_argument(
        "--finetune-safe",
        action="store_true",
        help="Safe ep100 fine-tune: eps=0, frozen features, clear replay, 25 ep",
    )
    args = ap.parse_args()

    if args.finetune_safe:
        # Low epsilon + uniform explore (sell_bias=0.33) — not eps=0 (no learning)
        # and not eps=0.2 + sell_bias=0.15 (kills BUY, like ft v1).
        init = args.init_from or "models/options_unified_gen380_ep100"
        prefix = args.model_prefix if args.model_prefix != MODEL_PREFIX else "options_unified_gen380_ft2"
        train(
            watchlist_path=args.watchlist,
            episodes=args.episodes if args.episodes != 150 else 25,
            train_every=64,
            log_dir=args.log_dir if args.log_dir != "logs/runs/options_unified_gen380" else "logs/runs/options_unified_gen380_ft2",
            init_from=init,
            freeze_ratio=1.0,
            initial_epsilon=0.03,
            final_epsilon=0.01,
            model_prefix=prefix,
            checkpoint_every=args.checkpoint_every,
            clear_replay=True,
            sell_bias=0.33,
            entry_reward_coef=1.0,
            policy_lr=5e-8,
        )
        return

    train(
        watchlist_path=args.watchlist,
        episodes=args.episodes,
        max_steps=args.max_steps or None,
        train_every=args.train_every,
        store_every=args.store_every,
        log_dir=args.log_dir,
        init_from=None if args.resume_latest else args.init_from,
        freeze_ratio=args.freeze_ratio,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        resume_latest=args.resume_latest,
        model_prefix=args.model_prefix,
        checkpoint_every=args.checkpoint_every,
        clear_replay=args.clear_replay,
        sell_bias=args.sell_bias,
        entry_reward_coef=args.entry_reward_coef,
        policy_lr=args.policy_lr,
    )


if __name__ == "__main__":
    main()
