import sys
import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from collections import deque

# Force UTF-8 for Windows
if sys.platform == 'win32':
    # type: ignore
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
from datetime import datetime

# Override Window Size for Crypto (Multi-Scale)
# We will use 60 bars (60 Days) as the primary input
TrainingConfig.WINDOW_SIZE = 60

from src.environments.vector_env import VectorizedTradingEnv  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.core.indicators import add_technical_indicators  # noqa: E402


def load_crypto_data(start_date: str, end_date: str, max_steps: int = 10000):
    print(f"📥 Loading Historical Crypto Data ({start_date} to {end_date})...")
    # match *_1D.csv
    pattern = os.path.join("data/historical", "*_1D.csv")
    all_files = glob.glob(pattern)
    
    # Filter for likely crypto symbols
    watchlist_path = "config/watchlists/crypto_watchlist.txt"
    if os.path.exists(watchlist_path):
        with open(watchlist_path, 'r') as f:
            crypto_symbols = [line.strip().replace('/', '').replace('-', '') for line in f if line.strip()]
    else:
        crypto_symbols = ["BTC", "ETH", "SOL", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "XRP", "BNB"]
        
    files = []
    for f in all_files:
        filename = os.path.basename(f).upper()
        # Check if any crypto symbol is in the filename (ignoring "USD" or similar suffixes if possible)
        # Our files are like BTCUSD_1D.csv
        found = False
        for sym in crypto_symbols:
             # simple check: if 'BTC' in 'BTCUSD_1D.CSV'
             if sym in filename:
                 found = True
                 break
        if found:
            files.append(f)
            
    if not files:
        print("❌ No crypto data found in data/historical matching pattern!")
        return None, None, [], 0
        
    print(f"📋 Found {len(files)} crypto data files for range.")
        
    data_list = []
    price_list = []
    
    # Convert string dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    final_steps = 0
    
    for f in tqdm(files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(f)
            
            # Standardize Columns
            df.columns = [c.lower() for c in df.columns]
            col_map = {'date': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 'adj close': 'Close'}
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
            
            # Ensure 'Date' exists and is datetime
            if 'Date' not in df.columns:
                # try to find a date column
                for c in df.columns:
                    if 'date' in c.lower() or 'time' in c.lower():
                        df['Date'] = df[c]
                        break
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Filter by Date Range
            mask = (df['Date'] >= start_dt) & (df['Date'] < end_dt)
            df = df.loc[mask].copy()
            
            if len(df) < 60: # Need at least window size
                continue

            # Capitalize existing full names for indicators
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col.lower() in df.columns:
                    df[col.capitalize()] = df[col.lower()]
            
            # Fill missing
            df = df.ffill().bfill()

            # Add Indicators
            df = add_technical_indicators(df)
            
            # Add Crypto-Specific Features
            df.index = df['Date']
            
            # Normalize time (Day of week, Month) - Hour doesn't matter for 1D
            day = df.index.dayofweek
            month = df.index.month
            
            df['sin_day'] = np.sin(2 * np.pi * day / 7)
            df['cos_day'] = np.cos(2 * np.pi * day / 7)
            df['sin_month'] = np.sin(2 * np.pi * month / 12)
            df['cos_month'] = np.cos(2 * np.pi * month / 12)
            
            # Missing columns from previous logic (hour) - just set to 0 for 1D
            df['sin_hour'] = 0.0
            df['cos_hour'] = 0.0

            # Volatility Normalization
            df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
            df['volatility'] = df['log_ret'].rolling(window=20).std().fillna(0) # 20 days vol
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 100:
                continue
                
            # If max_steps is set, we take the WHOLE range defined by dates, 
            # but we truncate if strictly necessary (though date range usually implies length)
            # data structure expects [episodes/symbols, timesteps, features]
            # We want all symbols to have SAME number of timesteps for simple batching?
            # Or padding? 
            # To simplify: We find the common date index intersection or just pad.
            # For now, let's just pad to the max length found in this batch or max_steps.
            
            # Extract Raw Prices
            raw_close = df['Close'].values
            
            # Extract Features
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12',
                'atr', 'bb_width',
                'sin_day', 'cos_day', 'sin_month', 'cos_month',
                'volatility'
            ]
            
            # Ensure all exist
            valid_cols = [c for c in feature_cols if c in df.columns]
            features = df[valid_cols].values
            
            # Normalize
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
            data_list.append(norm_features)
            price_list.append(raw_close)
            
        except Exception as e:
            # print(f"Error processing {f}: {e}")
            continue
            
    if not data_list:
        return None, None, [], 0

    # Determine max length in this batch
    lengths = [len(x) for x in data_list]
    max_len = max(lengths)
    final_steps = max_len
    
    # Pad to max_len
    padded_data = []
    padded_prices = []
    
    num_features = data_list[0].shape[1]
    
    for i in range(len(data_list)):
        d = data_list[i]
        p = price_list[i]
        curr_len = len(d)
        
        if curr_len < max_len:
            pad_len = max_len - curr_len
            # Pad with 0? Or repeat? 0 is safer for masks if we had them.
            # Since we don't have masks in VectorEnv easily, we pad with last value
            d_pad = np.pad(d, ((0, pad_len), (0, 0)), mode='edge')
            p_pad = np.pad(p, (0, pad_len), mode='edge')
            padded_data.append(d_pad)
            padded_prices.append(p_pad)
        else:
            padded_data.append(d)
            padded_prices.append(p)
            
    # Stack into Tensors
    data_tensor = torch.FloatTensor(np.array(padded_data))
    price_tensor = torch.FloatTensor(np.array(padded_prices))
    
    return data_tensor, price_tensor, files, final_steps

def train(max_steps: int = 10000, episodes: int = 200, train_every: int = 4, store_every: int = 4, log_dir: str = "logs/runs/crypto_experiment_1"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Crypto Bot Training on {device}...")
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # Define Split Dates (2/2 Strategy)
    # Train: 2022-01-01 -> 2023-12-31 (2 Years)
    # Test:  2024-01-01 -> 2025-12-31 (2 Years / Present)
    
    train_start = "2022-01-01"
    train_end = "2024-01-01"
    
    test_start = "2024-01-01"
    # End date far in future to capture everything up to now
    test_end = "2030-01-01"
    
    # 1. Load Data
    print("--- Loading Training Data ---")
    train_data, train_prices, train_files, train_steps_real = load_crypto_data(train_start, train_end, max_steps)
    
    print("--- Loading Test Data ---")
    test_data, test_prices, test_files, test_steps_real = load_crypto_data(test_start, test_end, max_steps)
    
    if train_data is None:
        print("❌ Training data failed to load.")
        return

    print(f"📊 Train Shape: {train_data.shape} ({len(train_files)} symbols)")
    if test_data is not None:
        print(f"📊 Test Shape:  {test_data.shape} ({len(test_files)} symbols)")
    
    # 2. Initialize Training Environment
    env = VectorizedTradingEnv(train_data, train_prices, device=device)
    # Crypto Fees
    env.transaction_cost_bps = 10.0 # 0.10%
    env.slippage_bps = 5.0          # 0.05%
    
    # 3. Initialize Agent
    num_features = train_data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )

    
    # Check for existing checkpoints
    start_episode = 0
    checkpoints = glob.glob("models/crypto_ep*_balanced.pth")
    if checkpoints:
        ep_nums = []
        for cp in checkpoints:
            try:
                num = int(cp.split("crypto_ep")[1].split("_")[0])
                ep_nums.append(num)
            except:
                pass
        
        if ep_nums:
            latest_ep = max(ep_nums)
            print(f"🔄 Found checkpoint for Episode {latest_ep}. Resuming...")
            agent.load(f"models/crypto_ep{latest_ep}")
            start_episode = latest_ep
            
            # Recalculate Epsilon
            epsilon_decay = 0.99999
            total_steps_passed = start_episode * env.total_steps
            calculated_epsilon = max(0.05, 1.0 * (epsilon_decay ** total_steps_passed))
            
            print(f"🔄 Adjusting Epsilon to {calculated_epsilon:.6f}")
            for sub_agent in agent.agents:
                sub_agent.epsilon = calculated_epsilon
                sub_agent.epsilon_decay = epsilon_decay
    else:
        epsilon_decay = 0.99999
        for sub_agent in agent.agents:
            sub_agent.epsilon_decay = epsilon_decay

    # 4. Training Loop
    EPISODES = int(episodes)
    
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    # Cache CPU numpy versions once (major speedup vs per-step GPU->CPU copies)
    dummy_text_ids_np = dummy_text_ids[0].detach().cpu().numpy()
    dummy_text_mask_np = dummy_text_mask[0].detach().cpu().numpy()
    
    for episode in range(start_episode, EPISODES):
        state = env.reset()
        total_reward = 0
        steps_per_episode = min(int(env.total_steps), int(max_steps))
        train_every_n = max(1, int(train_every))
        store_every_n = max(1, int(store_every))
        
        pbar = tqdm(total=steps_per_episode, desc=f"Episode {episode+1}/{EPISODES}")
        
        for step_idx in range(steps_per_episode):
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
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
            pbar.set_postfix({'Reward': f"{total_reward:.4f}", 'Eps': f"{agent.balanced.epsilon:.2f}"})
            
        pbar.close()
        writer.add_scalar("Reward/Episode_Total", total_reward, episode)

        # Save best checkpoint (rolling average)
        if episode == start_episode:
            recent_rewards = deque(maxlen=5)
            best_avg_reward = float("-inf")

        recent_rewards.append(total_reward)
        rolling_avg = float(np.mean(recent_rewards))
        writer.add_scalar("Reward/RollingAvg", rolling_avg, episode)

        if rolling_avg > best_avg_reward:
            best_avg_reward = rolling_avg
            best_prefix = "models/crypto_best"
            agent.save(best_prefix)
            print(f"🏆 New BEST rolling reward {best_avg_reward:.4f}. Saved to {best_prefix}_*.pth")
        
        # Save Models
        if (episode + 1) % 5 == 0:
            save_prefix = f"models/crypto_ep{episode+1}"
            agent.save(save_prefix)
            print(f"💾 Saved models to {save_prefix}_*.pth")
            
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crypto ensemble agent")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max timesteps per symbol per episode (1Min bars)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--train-every", type=int, default=4, help="Run gradient update every N env steps")
    parser.add_argument("--store-every", type=int, default=4, help="Store replay transition every N env steps")
    parser.add_argument("--log-dir", type=str, default="logs/runs/crypto_experiment_1", help="TensorBoard log directory")
    args = parser.parse_args()
    train(max_steps=args.max_steps, episodes=args.episodes, train_every=args.train_every, store_every=args.store_every, log_dir=args.log_dir)
