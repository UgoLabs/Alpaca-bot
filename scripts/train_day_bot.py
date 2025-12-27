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
# Override Window Size for Day Trading (Intraday)
# 78 bars = 1 full trading day (6.5 hours / 5 mins)
TrainingConfig.WINDOW_SIZE = 78 

from src.environments.vector_env import VectorizedTradingEnv  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.core.indicators import add_technical_indicators  # noqa: E402


def load_day_trading_data():
    print("📥 Loading Historical Intraday Data (1Min -> 5Min)...")
    # Use the existing 1Min data which is very rich (1M+ rows)
    pattern = os.path.join("data/historical", "*_1Min.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No data found in data/historical!")
        return None, None
        
    data_list = []
    price_list = []
    
    # We will process a subset of symbols to save RAM, or use a generator in future
    # For now, let's take the top 50 most liquid symbols
    files = files[:50]
    
    # Max Sequence Length for Day Trading
    # We don't need 10 years of 5-min data in one sequence (too big for GPU RAM)
    # Instead, we can split long histories into "Episodes" of e.g., 1 month (approx 2000 steps)
    # But for simplicity in this script, let's take the LAST 10,000 steps (approx 5 months)
    MAX_STEPS = 10000 
    
    for f in tqdm(files, desc="Processing CSVs"):
        try:
            # Read CSV
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

            # Ensure Datetime
            if 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp'])
                df.set_index('Date', inplace=True)
            
            # Resample to 5Min
            agg_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            if 'VWAP' in df.columns:
                agg_dict['VWAP'] = 'last' # Approximation
                
            df_5m = df.resample('5min').agg(agg_dict).dropna()
            
            # Add Indicators
            df_5m = add_technical_indicators(df_5m)
            
            # Add Intraday Features (Time of Day)
            # Normalize time to [0, 1]
            day_minutes = df_5m.index.hour * 60 + df_5m.index.minute
            df_5m['time_sin'] = np.sin(2 * np.pi * day_minutes / (24 * 60))
            df_5m['time_cos'] = np.cos(2 * np.pi * day_minutes / (24 * 60))
            
            # Add VWAP Distance if available
            if 'VWAP' in df_5m.columns:
                df_5m['vwap_dist'] = (df_5m['Close'] - df_5m['VWAP']) / df_5m['VWAP']
            else:
                df_5m['vwap_dist'] = 0.0
                
            df_5m = df_5m.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df_5m) < 1000:
                continue
                
            # Slice the last MAX_STEPS
            df_slice = df_5m.iloc[-MAX_STEPS:]
            
            # Extract Raw Prices (Close)
            raw_close = df_slice['Close'].values
            
            # Extract Features
            # We add 'time_sin', 'time_cos', 'vwap_dist' to the standard set
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12',
                'time_sin', 'time_cos', 'vwap_dist'
            ]
            
            # Ensure all exist
            valid_cols = [c for c in feature_cols if c in df_slice.columns]
            features = df_slice[valid_cols].values
            
            # Normalize
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
            # Pad if needed (though we sliced from end, so usually full)
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
    # Shape: (Num_Envs, Time, Features)
    data_tensor = torch.FloatTensor(np.array(data_list))
    price_tensor = torch.FloatTensor(np.array(price_list))
    
    return data_tensor, price_tensor

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Day Trader Training on {device}...")
    
    writer = SummaryWriter(log_dir="logs/runs/day_experiment_1")
    
    # 1. Load Data
    data, prices = load_day_trading_data()
    if data is None:
        return
    
    print(f"📊 Data Shape: {data.shape}")
    
    # 2. Initialize Environment
    # Day Trading needs higher transaction costs simulation
    # We can override the config here or in the env
    env = VectorizedTradingEnv(data, prices, device=device)
    env.transaction_cost_bps = 1.0 # 1 basis point (0.01%) per trade
    env.slippage_bps = 1.0         # 1 basis point slippage
    
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
    checkpoints = glob.glob("models/day_ep*_balanced.pth")
    if checkpoints:
        ep_nums = []
        for cp in checkpoints:
            try:
                num = int(cp.split("day_ep")[1].split("_")[0])
                ep_nums.append(num)
            except:
                pass
        
        if ep_nums:
            latest_ep = max(ep_nums)
            print(f"🔄 Found checkpoint for Episode {latest_ep}. Resuming...")
            agent.load(f"models/day_ep{latest_ep}")
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
            save_prefix = f"models/day_ep{episode+1}"
            agent.save(save_prefix)
            print(f"💾 Saved models to {save_prefix}_*.pth")
            
    writer.close()

if __name__ == "__main__":
    train()
