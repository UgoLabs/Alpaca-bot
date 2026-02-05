import sys
import os
import glob
import argparse
import json
import faulthandler
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

# GPU performance knobs (safe defaults; no effect on CPU)
if torch.cuda.is_available():
    try:
        # Allow TF32 on matmul/conv for a speed boost on Ampere+ GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Prefer faster matmul kernels where available (PyTorch 2+).
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# Force UTF-8 for Windows
if sys.platform == 'win32':
    # type: ignore
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Print Python tracebacks even on hard crashes (segfault/access violation)
try:
    faulthandler.enable()
except Exception:
    pass

from config.settings import TrainingConfig
# Override Window Size for Swing Trading
TrainingConfig.WINDOW_SIZE = 60

from src.environments.vector_env import VectorizedTradingEnv  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.core.indicators import add_technical_indicators  # noqa: E402


def _find_latest_phase2_checkpoint_prefix() -> tuple[str | None, int]:
    """Return (prefix, episode_number) for the latest saved Phase-2 checkpoint.

    Expects files like models/swing_phase2_ep{N}_balanced.pth.
    """
    pattern = os.path.join("models", "swing_phase2_ep*_balanced.pth")
    paths = glob.glob(pattern)
    if not paths:
        return None, 0

    best_ep = 0
    for path in paths:
        base = os.path.basename(path)
        # swing_phase2_ep30_balanced.pth
        try:
            ep_str = base.split("swing_phase2_ep", 1)[1].split("_", 1)[0]
            ep_num = int(ep_str)
            if ep_num > best_ep:
                best_ep = ep_num
        except Exception:
            continue

    if best_ep <= 0:
        return None, 0
    return os.path.join("models", f"swing_phase2_ep{best_ep}"), best_ep


def _load_best_reward(meta_path: str) -> float | None:
    try:
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        val = payload.get("best_reward")
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _save_best_reward(meta_path: str, best_reward: float, episode: int) -> None:
    try:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        payload = {
            "best_reward": float(best_reward),
            "episode": int(episode),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        # Don't crash training if metadata write fails
        return


def load_swing_data(cutoff_date=None, oversample_bear=True, bear_multiplier=3):
    pattern = os.path.join("data/historical_swing", "*_1D.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("No data found in data/historical_swing. Run download_swing_data.py first.")
        return None, None
        
    data_list = []
    price_list = []
    
    # We want to use the full history (10 years ~ 2500 days)
    # We will pad shorter sequences to the max length found
    max_seq_len = 0
    
    processed_dfs = []
    bear_market_dfs = []  # Separate list for bear market data (computed after cutoff)
    
    # Define bear market periods (2018 correction + 2022 bear market)
    bear_periods = [
        (pd.to_datetime('2018-01-01'), pd.to_datetime('2019-01-01')),  # 2018 correction (~20% drop)
        (pd.to_datetime('2022-01-01'), pd.to_datetime('2023-01-01')),  # 2022 bear market (~25% drop)
    ]
    
    for f in tqdm(files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(f)
            
            # Standardize Columns
            df.columns = [c.lower() for c in df.columns]
            col_map = {'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
            
            # Capitalize existing full names
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col.capitalize()] = df[col]

            # Handle Date Index for Filtering
            if 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
                df.set_index('Date', inplace=True)
            elif 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp'])
                df.set_index('Date', inplace=True)

            # Add Indicators
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 200: # Minimum history
                continue
            
            processed_dfs.append(df)
            if len(df) > max_seq_len:
                max_seq_len = len(df)
                
        except Exception:
            continue
            
    if not processed_dfs:
        return None, None

    # Default behavior: 8/2 split (train on everything before last 2 years).
    # If caller passes cutoff_date explicitly, we respect it.
    if cutoff_date is None:
        try:
            latest_dt = max(df.index.max() for df in processed_dfs if len(df.index) > 0)
            cutoff_dt = (pd.to_datetime(latest_dt) - pd.DateOffset(years=1)).normalize()
            cutoff_date = cutoff_dt.strftime("%Y-%m-%d")
            print(f"9/1 split: cutoff={cutoff_date}")
        except Exception:
            cutoff_date = None

    # Apply cutoff after indicators so the split is based on clean data.
    if cutoff_date:
        cutoff_ts = pd.to_datetime(cutoff_date)
        processed_dfs = [df[df.index < cutoff_ts] for df in processed_dfs]
        processed_dfs = [df for df in processed_dfs if len(df) >= 200]
        if not processed_dfs:
            print(f"All sequences filtered out by cutoff_date={cutoff_date}.")
            return None, None

        # Recompute max_seq_len after split
        max_seq_len = max(len(df) for df in processed_dfs)

    # Extract bear market portions for oversampling (from TRAIN data only)
    if oversample_bear:
        bear_market_dfs = []
        for df in processed_dfs:
            for bear_start, bear_end in bear_periods:
                bear_df = df[(df.index >= bear_start) & (df.index < bear_end)]
                if len(bear_df) >= 100:
                    bear_market_dfs.append(bear_df)
    
    # Oversample bear market data
    if oversample_bear and bear_market_dfs:
        for _ in range(bear_multiplier - 1):
            processed_dfs.extend(bear_market_dfs)
        for df in bear_market_dfs:
            if len(df) > max_seq_len:
                max_seq_len = len(df)
        
    print(f"Loaded {len(processed_dfs)} sequences, max_len={max_seq_len}")
    
    for df in processed_dfs:
        # Extract Raw Prices (Close)
        raw_close = df['Close'].values
        
        # Extract Features (original 11 features for compatibility with existing model)
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12',
        ]
        
        # Ensure all exist
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < 5:
            continue
            
        features = df[valid_cols].values
        
        # Normalize
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        norm_features = (features - mean) / std
        
        # Pad to max_seq_len (Pre-padding)
        curr_len = len(norm_features)
        if curr_len < max_seq_len:
            pad_len = max_seq_len - curr_len
            # Pad with first value to simulate "waiting"
            # Or pad with zeros. Edge padding is safer for time series.
            norm_features = np.pad(norm_features, ((pad_len, 0), (0, 0)), mode='edge')
            raw_close = np.pad(raw_close, (pad_len, 0), mode='edge')
        
        data_list.append(norm_features)
        price_list.append(raw_close)
            
    # Stack into Tensors
    # Shape: (Num_Envs, Time, Features)
    data_tensor = torch.FloatTensor(np.array(data_list))
    price_tensor = torch.FloatTensor(np.array(price_list))
    
    return data_tensor, price_tensor

def train_phase_2(
    max_steps: int | None = None,
    episodes: int = 50,
    train_every: int = 32,
    store_every: int = 32,
    log_dir: str = "logs/runs/swing_phase_2",
    init_from: str | None = None,
    start_episode: int = 0,
    resume_latest: bool = True,
    more_episodes: int = 0,
    best_reward_override: float | None = None,
    checkpoint_every: int = 1,
    initial_epsilon: float = 0.05,
    epsilon_decay: float = 0.99,
    train_cutoff_date: str | None = None,
    oversample_bear: bool = True,
    num_symbols: int | None = None,
    seed: int = 42,
    early_stopping_patience: int = 0,
    model_prefix: str = "swing_phase2",
    freeze_ratio: float = 0.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on Device: {device.upper()}")
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # 1. Load Data
    data, prices = load_swing_data(cutoff_date=train_cutoff_date, oversample_bear=oversample_bear)
    if data is None:
        return
    
    # Optional: subsample symbols (environments) for faster iteration.
    # This is the single biggest lever for speed because cost scales ~linearly with num_envs.
    if num_symbols is not None:
        try:
            num_symbols = int(num_symbols)
        except Exception:
            num_symbols = None
    if num_symbols is not None and num_symbols > 0 and num_symbols < int(data.shape[0]):
        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = torch.randperm(int(data.shape[0]), generator=g)[: int(num_symbols)]
        data = data[idx]
        prices = prices[idx]
        print(f"Using {num_symbols} symbols (of {len(idx)} available)")
    else:
        print(f"Using all {data.shape[0]} symbols")
    
    # 2. Initialize Environment
    env = VectorizedTradingEnv(data, prices, device=device)
    
    # 3. Initialize Ensemble Agent
    num_features = data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )
    
    # 4. Load starting weights (explicit init_from > latest phase2 > phase2 best > phase1 best)
    resolved_prefix: str | None = None
    resolved_start_episode = max(0, int(start_episode))

    if init_from:
        resolved_prefix = str(init_from)
    elif resume_latest:
        latest_prefix, latest_ep = _find_latest_phase2_checkpoint_prefix()
        if latest_prefix:
            resolved_prefix = latest_prefix
            resolved_start_episode = max(resolved_start_episode, int(latest_ep))
        elif os.path.exists("models/swing_best_phase2_balanced.pth"):
            resolved_prefix = "models/swing_best_phase2"

    if not resolved_prefix:
        resolved_prefix = "models/swing_best"

    try:
        agent.load(resolved_prefix)
        print(f"Loaded: {resolved_prefix} (ep={resolved_start_episode})")
    except Exception as e:
        print(f"Could not load '{resolved_prefix}': {e}")
        return

    # 5. CONFIG FREEZING
    freeze_limit = 0
    if freeze_ratio > 0:
        freeze_limit = int(episodes * freeze_ratio)
        print(f"Freezing active until episode {freeze_limit} (Ratio: {freeze_ratio}).")

    # Apply initial freeze state
    if resolved_start_episode < freeze_limit:
        print(f"Starting in FROZEN state (ep {resolved_start_episode} < {freeze_limit})")
        agent.freeze_feature_extractors()
        # Ensure optimizer has correct LR (Phase 2 uses 1e-6)
        for sub_agent in agent.agents:
            sub_agent.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, sub_agent.policy_net.parameters()), 
                lr=1e-6
            )
    else:
        print("Starting in UNFROZEN state (full fine-tuning)")
        agent.unfreeze_all()
        # Ensure optimizer has correct LR (Phase 2 uses 1e-6)
        for sub_agent in agent.agents:
            sub_agent.optimizer = torch.optim.Adam(sub_agent.policy_net.parameters(), lr=1e-6)
    
    # 6. Exploration settings
    try:
        initial_epsilon = float(initial_epsilon)
    except Exception:
        initial_epsilon = 0.05
    try:
        epsilon_decay = float(epsilon_decay)
    except Exception:
        epsilon_decay = 0.99

    # Keep within reasonable bounds
    if initial_epsilon < 0.0:
        initial_epsilon = 0.0
    if initial_epsilon > 1.0:
        initial_epsilon = 1.0
    if epsilon_decay <= 0.0:
        epsilon_decay = 0.99
    if epsilon_decay > 1.0:
        epsilon_decay = 1.0

    for sub_agent in agent.agents:
        sub_agent.epsilon = initial_epsilon
        sub_agent.epsilon_decay = epsilon_decay

    # 7. Training Loop
    EPISODES = int(episodes)  # Upper bound of episode index (exclusive)
    
    # Dummy Text Data
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    # Cache CPU numpy versions once (avoid per-step GPU->CPU copies)
    dummy_text_ids_np = dummy_text_ids[0].detach().cpu().numpy()
    dummy_text_mask_np = dummy_text_mask[0].detach().cpu().numpy()
    
    # Track Best Reward (persisted) so we don't overwrite the best checkpoint with a worse run.
    best_model_meta_path = os.path.join("models", f"{model_prefix}_best.meta.json")
    persisted_best = _load_best_reward(best_model_meta_path)

    if best_reward_override is not None:
        best_reward = float(best_reward_override)
        _save_best_reward(best_model_meta_path, best_reward, episode=max(0, int(start_episode)))
    elif persisted_best is not None:
        best_reward = float(persisted_best)
    else:
        best_reward = float("-inf")

    # Early stopping
    no_improve_count = 0
    early_stop_patience = max(0, int(early_stopping_patience))

    train_every_n = max(1, int(train_every))
    store_every_n = max(1, int(store_every))
    checkpoint_every_n = max(1, int(checkpoint_every))

    # Always keep a crash-resume checkpoint up to date
    last_prefix = os.path.join("models", f"{model_prefix}_last")

    start_ep = max(0, int(resolved_start_episode))

    # Convenience: run N more episodes starting at start_ep
    add_n = max(0, int(more_episodes))
    if add_n > 0:
        EPISODES = start_ep + add_n

    if start_ep >= EPISODES:
        writer.close()
        return

    try:
        for episode in range(start_ep, EPISODES):
            # Check for Unfreeze
            if freeze_limit > 0 and episode == freeze_limit:
                print(f"🔥 UNFREEZING all layers at episode {episode}! Using lower LR (1e-7).")
                agent.unfreeze_all()
                for sub_agent in agent.agents:
                    sub_agent.optimizer = torch.optim.Adam(sub_agent.policy_net.parameters(), lr=1e-7)

            current_eps = agent.balanced.epsilon
            print(f"[Ep {episode+1}/{EPISODES}] Eps={current_eps:.4f}")
            
            state = env.reset()  # (Envs, Window, Features)
            total_reward = 0.0

            steps_per_episode = int(env.total_steps)
            if max_steps is not None:
                steps_per_episode = min(steps_per_episode, int(max_steps))

            pbar = tqdm(total=steps_per_episode, desc=f"Ep{episode+1}", leave=False)

            for step_idx in range(steps_per_episode):
                # Ensemble Action
                actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)

                # Step Env
                next_state, rewards, dones, _ = env.step(actions)

                # Store Experience (optionally subsampled)
                if (step_idx % store_every_n) == 0:
                    state_cpu = state.detach().cpu().numpy()
                    next_state_cpu = next_state.detach().cpu().numpy()
                    actions_cpu = actions.detach().cpu().numpy()
                    rewards_cpu = rewards.detach().cpu().numpy()
                    dones_cpu = dones.detach().cpu().numpy()

                    for i in range(env.num_envs):
                        s = (state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                        ns = (next_state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                        agent.remember(s, actions_cpu[i], rewards_cpu[i], ns, dones_cpu[i])

                losses = None
                if (step_idx % train_every_n) == 0:
                    losses = agent.train_step()

                global_step = episode * steps_per_episode + step_idx
                if losses:
                    for name, loss in losses.items():
                        writer.add_scalar(f"Loss/{name}", loss, global_step)

                writer.add_scalar("Epsilon/Balanced", agent.balanced.epsilon, global_step)

                state = next_state
                total_reward += rewards.sum().item()

                pbar.update(1)
                pbar.set_postfix({
                    'Reward': f"{total_reward:.1f}", 
                    'Epsilon': f"{agent.balanced.epsilon:.3f}"
                })

            pbar.close()
            
            # Episode summary
            print(f"  -> Reward={total_reward:.2f} | NoImprove={no_improve_count}")
            
            writer.add_scalar("Reward/Episode_Total", total_reward, episode)

            # Decay epsilon at END of episode (not per step)
            for sub_agent in agent.agents:
                if sub_agent.epsilon > sub_agent.epsilon_min:
                    sub_agent.epsilon *= epsilon_decay

            # Always save a crash-resume checkpoint
            if ((episode - start_ep + 1) % checkpoint_every_n) == 0:
                agent.save(last_prefix)

            # Save Best Model
            if total_reward > best_reward:
                best_reward = total_reward
                no_improve_count = 0
                print(f"Best: {best_reward:.4f}")
                agent.save(os.path.join("models", f"{model_prefix}_best"))
                _save_best_reward(best_model_meta_path, best_reward, episode=episode)
            else:
                no_improve_count += 1

            # Early stopping check
            if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
                print(f"Early stopping: no improvement for {early_stop_patience} episodes")
                break

            # Regular Checkpoint
            if (episode + 1) % 10 == 0:
                save_prefix = os.path.join("models", f"{model_prefix}_ep{episode+1}")
                agent.save(save_prefix)
    except KeyboardInterrupt:
        try:
            agent.save(last_prefix)
        except Exception:
            pass
        writer.close()
        raise SystemExit(0)
    except Exception as e:
        print(f"Crashed: {e}")
        try:
            agent.save(last_prefix)
        except Exception:
            pass
        raise
            
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train swing bot (Phase 2 fine-tuning)")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on steps per episode (0 = full)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--train-every", type=int, default=32, help="Run gradient update every N env steps")
    parser.add_argument("--store-every", type=int, default=32, help="Store replay transition every N env steps")
    parser.add_argument("--log-dir", type=str, default="logs/runs/swing_phase_2", help="TensorBoard log directory")
    parser.add_argument("--init-from", type=str, default="", help="Checkpoint prefix to initialize from (e.g., models/swing_phase2_ep30)")
    parser.add_argument("--start-episode", type=int, default=0, help="Episode offset when resuming from a saved epN checkpoint")
    parser.add_argument("--no-resume-latest", action="store_true", help="Disable auto-resume from latest swing_phase2_ep* checkpoint")
    parser.add_argument("--more-episodes", type=int, default=0, help="Train N additional episodes starting from the resolved start episode")
    parser.add_argument("--best-reward", type=float, default=None, help="Baseline best reward to prevent overwriting best checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save crash-resume checkpoint every N episodes")
    parser.add_argument("--initial-epsilon", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Epsilon decay factor applied each train step")
    parser.add_argument("--train-cutoff-date", type=str, default=None, help="Cutoff date for training data (YYYY-MM-DD)")
    parser.add_argument("--no-oversample", action="store_true", help="Disable bear market oversampling")
    parser.add_argument("--num-symbols", type=int, default=0, help="Train on N random symbols (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (affects symbol subsampling)")
    parser.add_argument("--early-stopping", type=int, default=0, help="Stop if no improvement for N episodes (0 = disabled)")
    parser.add_argument("--model-prefix", type=str, default="swing_phase2", help="Prefix for saved models (e.g. swing_phase2_modelA)")
    parser.add_argument("--freeze-ratio", type=float, default=0.0, help="Ratio of episodes to freeze feature extractors (e.g. 0.2)")
    args = parser.parse_args()

    max_steps = None if int(args.max_steps) <= 0 else int(args.max_steps)
    train_phase_2(
        max_steps=max_steps,
        episodes=int(args.episodes),
        train_every=int(args.train_every),
        store_every=int(args.store_every),
        log_dir=str(args.log_dir),
        init_from=(str(args.init_from).strip() or None),
        start_episode=int(args.start_episode),
        resume_latest=(not bool(args.no_resume_latest)),
        more_episodes=int(args.more_episodes),
        best_reward_override=args.best_reward,
        checkpoint_every=int(args.checkpoint_every),
        initial_epsilon=float(args.initial_epsilon),
        epsilon_decay=float(args.epsilon_decay),
        train_cutoff_date=args.train_cutoff_date,
        oversample_bear=(not bool(args.no_oversample)),
        num_symbols=(None if int(args.num_symbols) <= 0 else int(args.num_symbols)),
        seed=int(args.seed),
        early_stopping_patience=int(args.early_stopping),
        model_prefix=str(args.model_prefix),
        freeze_ratio=float(args.freeze_ratio),
    )
