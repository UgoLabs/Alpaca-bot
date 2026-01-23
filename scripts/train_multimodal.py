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

from src.environments.vector_env import VectorizedTradingEnv  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.core.indicators import add_technical_indicators  # noqa: E402


def load_all_data(timeframe="1D", train_end_date="2023-12-31"):
    """
    Load data with train/test split to prevent data leakage.
    
    Args:
        timeframe: "1D" for swing, "1Min" for scalp
        train_end_date: Last date for training data (everything after is held out for testing)
    """
    print(f"📥 Loading Historical Data ({timeframe})...")
    print(f"📅 Training cutoff: {train_end_date} (data after this is reserved for backtesting)")
    
    if timeframe == "1Min":
        data_dir = "data/historical"
    else:
        data_dir = "data/historical_swing"
        
    pattern = os.path.join(data_dir, f"*_{timeframe}.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No data found in {data_dir} for {timeframe}!")
        return None, None
        
    data_list = []
    price_list = []
    # 8 years of daily data = ~2000 trading days. Use actual data length, no padding.
    MAX_LEN = 2000
    
    # --- SMART SELECTION LOGIC ---
    # User requested: Top 100, Tech Heavy, Diversified, Important Symbols
    PRIORITY_SYMBOLS = [
        # Tech Giants (Mag 7 + Friends)
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 
        'AVGO', 'QCOM', 'INTC', 'CRM', 'ADBE', 'ORCL', 'IBM', 'NFLX',
        # High Growth / Popular Tech
        'PLTR', 'SNOW', 'U', 'COIN', 'SHOP', 'SQ', 'ROKU', 'NET', 'CRWD', 'PANW',
        # Semiconductors
        'TSM', 'MU', 'LRCX', 'AMAT', 'KLAC',
        # Major ETFs (Market Health)
        'SPY', 'QQQ', 'IWM', 'DIA', 'ARKK', 'SMH', 'XLK', 'XLF',
        # Financials
        'JPM', 'BAC', 'GS', 'V', 'MA', 'BLK',
        # Consumer / Retail
        'WMT', 'COST', 'HD', 'NKE', 'SBUX', 'MCD', 'DIS',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'LLY', 'ABBV',
        # Industrial / Energy
        'CAT', 'BA', 'GE', 'XOM', 'CVX',
        # Crypto / High Vol
        'MSTR', 'MARA', 'RIOT'
    ]
    
    priority_files = []
    other_files = []
    
    # Categorize files
    for f in files:
        filename = os.path.basename(f)
        # Dynamic replacement for cleaner symbol extraction
        symbol = filename.replace(f'_{timeframe}.csv', '').upper() 
        if symbol in PRIORITY_SYMBOLS:
            priority_files.append(f)
        else:
            other_files.append(f)
            
    # Select All Available
    # TARGET_COUNT = 100
    
    # 1. Take all available priority files
    selected_files = list(priority_files)
    selected_files.extend(other_files) # Add all other files
    
    # Randomize order
    import random
    random.shuffle(selected_files)
    
    # 2. Fill the rest with random selection from others to diversify
    # remaining_slots = TARGET_COUNT - len(selected_files)
    
    # if remaining_slots > 0 and other_files:
        # Shuffle others to ensure random diversity each run
        # import random
        # random.shuffle(other_files)
        # selected_files.extend(other_files[:remaining_slots])
    
    print(f"📊 Training Selection: {len(priority_files)} Priority + {len(other_files)} Others = {len(selected_files)} Total Symbols")
    
    for f in tqdm(selected_files, desc="Processing CSVs"):
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
            
            # === TRAIN/TEST SPLIT: Filter to training period only ===
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] <= train_end_date]
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[df['Date'] <= train_end_date]

            # Add Indicators
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 1000:
                continue
                
            # Extract Raw Prices (Close)
            raw_close = df['Close'].values[-MAX_LEN:]
            
            # Extract Features (Normalized)
            # We need to normalize the data for the agent
            # Simple Z-score normalization for now or use the state.py logic
            # For speed, let's just take the columns we need
            # Note: indicators.py returns lowercase columns (rsi, macd, etc.)
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12'
            ]
            
            # Ensure all exist
            valid_cols = [c for c in feature_cols if c in df.columns]
            if len(valid_cols) < 5:
                continue
                
            features = df[valid_cols].values[-MAX_LEN:]
            
            # Normalize
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
            # Pad if too short
            if len(norm_features) < MAX_LEN:
                pad_len = MAX_LEN - len(norm_features)
                norm_features = np.pad(norm_features, ((pad_len, 0), (0, 0)), mode='edge')
                raw_close = np.pad(raw_close, (pad_len, 0), mode='edge')
            
            data_list.append(norm_features)
            price_list.append(raw_close)
            
        except Exception:
            continue
            
    if not data_list:
        return None, None
    
    # === BEAR MARKET OVERSAMPLING (DISABLED - too slow) ===
    # Instead, the random start positions in VectorizedEnv naturally sample all periods
    print(f"📊 Loaded {len(data_list)} symbols")
        
    # Stack into Tensors
    # Shape: (Num_Envs, Time, Features)
    data_tensor = torch.FloatTensor(np.array(data_list))
    price_tensor = torch.FloatTensor(np.array(price_list))
    
    return data_tensor, price_tensor

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Multi-Modal Ensemble Training on {device}...")
    
    # Initialize TensorBoard Writer immediately
    writer = SummaryWriter(log_dir="logs/runs/ensemble_experiment_1")
    
    # 1. Load Data
    data, prices = load_all_data(timeframe="1D")
    if data is None:
        return
    
    print(f"📊 Data Shape: {data.shape}")
    
    # 2. Initialize Environment with LIVE TRADING PARAMETERS
    # Position sizing: 1/10 of capital per position (max 10 positions)
    env = VectorizedTradingEnv(data, prices, device=device, position_pct=0.10)
    
    # Override initial balance to match live account
    env.initial_balance = 10000.0
    env.balance.fill_(10000.0)
    env.prev_equity.fill_(10000.0)
    
    print(f"💰 Live params: $10,000 starting capital, 10% position sizing (max 10 positions)")
    
    # 3. Initialize Ensemble Agent
    num_features = data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )
    
    # Check for existing checkpoints (skip if --fresh flag is passed)
    start_episode = 0
    fresh_start = "--fresh" in sys.argv
    
    if fresh_start:
        print("🆕 Fresh start requested. Ignoring existing checkpoints.")
    else:
        checkpoints = glob.glob("models/ensemble_ep*_balanced.pth")
        if checkpoints:
            # Extract episode numbers
            ep_nums = []
            for cp in checkpoints:
                try:
                    # Format: models/ensemble_ep5_balanced.pth
                    num = int(cp.split("ensemble_ep")[1].split("_")[0])
                    ep_nums.append(num)
                except:
                    pass
            
            if ep_nums:
                latest_ep = max(ep_nums)
                print(f"🔄 Found checkpoint for Episode {latest_ep}. Resuming...")
                agent.load(f"models/ensemble_ep{latest_ep}")
                start_episode = latest_ep
                
                # Recalculate Epsilon for the new schedule
                # User requested fixed exploration of 0.4
                new_epsilon = 0.4
                # Adjust decay to reach 0.05 by Episode 100 (approx 250k steps)
                new_decay = 0.99999 
                print(f"🔄 Adjusting Epsilon to {new_epsilon:.4f} and Decay to {new_decay} for Episode {start_episode+1}")
                
                for sub_agent in agent.agents:
                    sub_agent.epsilon = new_epsilon
                    sub_agent.epsilon_decay = new_decay

    # 4. Training Loop
    EPISODES = 200 
    # writer = SummaryWriter(log_dir="logs/runs/ensemble_experiment_1") # Moved to top
    
    # Dummy Text Data (since we don't have historical news)
    # Shape: (Num_Envs, Seq_Len)
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    
    for episode in range(start_episode, EPISODES):
        state = env.reset()  # (Envs, Window, Features)
        total_reward = 0
        
        # Run for a fixed number of steps per episode (e.g., full data length)
        # Since env auto-resets, we don't need to check for done.
        steps_per_episode = env.total_steps
        
        pbar = tqdm(total=steps_per_episode, desc=f"Episode {episode+1}/{EPISODES}")
        
        # Pre-compute dummy text once (saves repeated .cpu().numpy() calls)
        dummy_text_ids_np = dummy_text_ids[0].cpu().numpy()
        dummy_text_mask_np = dummy_text_mask[0].cpu().numpy()
        
        for step_idx in range(steps_per_episode):
            # Ensemble Action
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            
            # Step Env
            next_state, rewards, dones, _ = env.step(actions)
            
            # Store Experience & Train (OPTIMIZED: sample subset instead of all)
            # Only store 10% of transitions per step to reduce memory overhead
            sample_size = max(1, env.num_envs // 10)
            sample_indices = np.random.choice(env.num_envs, sample_size, replace=False)
            
            state_cpu = state[sample_indices].cpu().numpy()
            next_state_cpu = next_state[sample_indices].cpu().numpy()
            actions_cpu = actions[sample_indices].cpu().numpy()
            rewards_cpu = rewards[sample_indices].cpu().numpy()
            dones_cpu = dones[sample_indices].cpu().numpy()
            
            for i in range(sample_size):
                s = (state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                ns = (next_state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                agent.remember(s, actions_cpu[i], rewards_cpu[i], ns, dones_cpu[i])
            
            # Train Step (Trains all 3 agents)
            losses = agent.train_step()
            
            # Log to TensorBoard (less frequently to reduce overhead)
            if step_idx % 100 == 0:
                global_step = episode * steps_per_episode + step_idx
                if losses:
                    for name, loss in losses.items():
                        writer.add_scalar(f"Loss/{name}", loss, global_step)
                writer.add_scalar("Epsilon/Balanced", agent.balanced.epsilon, global_step)
            
            state = next_state
            total_reward += rewards.sum().item()
                
            pbar.update(1)
            pbar.set_postfix({'Reward': f"{total_reward:.4f}", 'Eps (Bal)': f"{agent.balanced.epsilon:.2f}"})
            
        pbar.close()
        writer.add_scalar("Reward/Episode_Total", total_reward, episode)
        
        # ===== EPSILON DECAY (per episode) =====
        # Decay from 1.0 to 0.05 over 50 episodes
        # Formula: eps * decay^episode = eps_min => decay = (eps_min/eps)^(1/episodes)
        # decay = (0.05/1.0)^(1/50) ≈ 0.94
        eps_decay_per_episode = 0.94
        for sub_agent in agent.agents:
            sub_agent.epsilon = max(sub_agent.epsilon_min, sub_agent.epsilon * eps_decay_per_episode)
        
        print(f"📉 Epsilon decayed to {agent.balanced.epsilon:.4f}")
        
        # Save Models
        if (episode + 1) % 5 == 0:
            save_prefix = f"models/ensemble_ep{episode+1}"
            agent.save(save_prefix)
            print(f"💾 Saved models to {save_prefix}_*.pth")
            
    writer.close()

if __name__ == "__main__":
    train()
