import sys
import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

# Force UTF-8 for Windows
if sys.platform == 'win32':
    # type: ignore
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
# Override Window Size for Crypto (Multi-Scale)
# We will use 60 bars (1 hour of 1Min data) as the primary input
TrainingConfig.WINDOW_SIZE = 60

from src.environments.vector_env import VectorizedTradingEnv  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.core.indicators import add_technical_indicators  # noqa: E402


def load_crypto_data():
    print("📥 Loading Historical Crypto Data (1Min)...")
    # We filter for crypto files (usually have '/' in name or are in specific list)
    # Assuming files are named like 'BTCUSD_1Min.csv' or similar
    pattern = os.path.join("data/historical", "*_1Min.csv")
    all_files = glob.glob(pattern)
    
    # Filter for likely crypto symbols (e.g., BTC, ETH, SOL, etc.)
    # Or check against watchlist
    watchlist_path = "config/watchlists/crypto_watchlist.txt"
    if os.path.exists(watchlist_path):
        with open(watchlist_path, 'r') as f:
            crypto_symbols = [line.strip().replace('/', '') for line in f if line.strip()]
    else:
        crypto_symbols = ["BTC", "ETH", "SOL", "ADA", "DOGE", "DOT", "AVAX", "MATIC"]
        
    files = []
    for f in all_files:
        filename = os.path.basename(f).upper()
        # Check if any crypto symbol is in the filename
        if any(sym in filename for sym in crypto_symbols):
            files.append(f)
            
    if not files:
        print("❌ No crypto data found in data/historical!")
        return None, None
        
    print(f"📋 Found {len(files)} crypto data files.")
        
    data_list = []
    price_list = []
    
    # Crypto episodes can be long. Let's use 10,000 steps (approx 1 week of minutes)
    MAX_STEPS = 10000
    
    for f in tqdm(files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(f)
            
            # Standardize Columns
            df.columns = [c.lower() for c in df.columns]
            col_map = {'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 'vwap': 'VWAP'}
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
            
            # Capitalize existing full names
            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                if col in df.columns:
                    df[col.capitalize()] = df[col]

            # Add Indicators (Now includes ATR and BB Width)
            df = add_technical_indicators(df)
            
            # Add Crypto-Specific Features
            # 1. Time Encoding (24/7 market)
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
            
            # Normalize time (Hour of day, Day of week)
            hour = df.index.hour
            day = df.index.dayofweek
            
            df['sin_hour'] = np.sin(2 * np.pi * hour / 24)
            df['cos_hour'] = np.cos(2 * np.pi * hour / 24)
            df['sin_day'] = np.sin(2 * np.pi * day / 7)
            df['cos_day'] = np.cos(2 * np.pi * day / 7)
            
            # 2. Volatility Normalization (Rolling Z-Score for Close)
            # Crypto regimes change fast, so we normalize returns relative to recent volatility
            df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
            df['volatility'] = df['log_ret'].rolling(window=60).std().fillna(0)
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 1000:
                continue
                
            # Slice the last MAX_STEPS
            df_slice = df.iloc[-MAX_STEPS:]
            
            # Extract Raw Prices (Close)
            raw_close = df_slice['Close'].values
            
            # Extract Features
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12',
                'atr', 'bb_width',
                'sin_hour', 'cos_hour', 'sin_day', 'cos_day',
                'volatility'
            ]
            
            # Ensure all exist
            valid_cols = [c for c in feature_cols if c in df_slice.columns]
            features = df_slice[valid_cols].values
            
            # Normalize
            # For crypto, we use robust scaling (median/IQR) or just standard scaling
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
            if len(norm_features) < MAX_STEPS:
                pad_len = MAX_STEPS - len(norm_features)
                norm_features = np.pad(norm_features, ((pad_len, 0), (0, 0)), mode='edge')
                raw_close = np.pad(raw_close, (pad_len, 0), mode='edge')
            
            data_list.append(norm_features)
            price_list.append(raw_close)
            
        except Exception:
            continue
            
    if not data_list:
        return None, None
        
    # Stack into Tensors
    data_tensor = torch.FloatTensor(np.array(data_list))
    price_tensor = torch.FloatTensor(np.array(price_list))
    
    return data_tensor, price_tensor

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Crypto Bot Training on {device}...")
    
    writer = SummaryWriter(log_dir="logs/runs/crypto_experiment_1")
    
    # 1. Load Data
    data, prices = load_crypto_data()
    if data is None:
        return
    
    print(f"📊 Data Shape: {data.shape}")
    
    # 2. Initialize Environment
    # Crypto has higher fees (usually 0.1% to 0.25% taker)
    env = VectorizedTradingEnv(data, prices, device=device)
    env.transaction_cost_bps = 10.0 # 0.10% per trade
    env.slippage_bps = 5.0          # 0.05% slippage
    
    # 3. Initialize Ensemble Agent
    num_features = data.shape[2]
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
    EPISODES = 100
    
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    
    for episode in range(start_episode, EPISODES):
        state = env.reset()
        total_reward = 0
        steps_per_episode = env.total_steps
        
        pbar = tqdm(total=steps_per_episode, desc=f"Episode {episode+1}/{EPISODES}")
        
        for step_idx in range(steps_per_episode):
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            next_state, rewards, dones, _ = env.step(actions)
            
            # Store Experience
            state_cpu = state.cpu().numpy()
            next_state_cpu = next_state.cpu().numpy()
            actions_cpu = actions.cpu().numpy()
            rewards_cpu = rewards.cpu().numpy()
            dones_cpu = dones.cpu().numpy()
            
            for i in range(env.num_envs):
                s = (state_cpu[i], dummy_text_ids[i].cpu().numpy(), dummy_text_mask[i].cpu().numpy())
                ns = (next_state_cpu[i], dummy_text_ids[i].cpu().numpy(), dummy_text_mask[i].cpu().numpy())
                agent.remember(s, actions_cpu[i], rewards_cpu[i], ns, dones_cpu[i])
            
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
        
        # Save Models
        if (episode + 1) % 5 == 0:
            save_prefix = f"models/crypto_ep{episode+1}"
            agent.save(save_prefix)
            print(f"💾 Saved models to {save_prefix}_*.pth")
            
    writer.close()

if __name__ == "__main__":
    train()
