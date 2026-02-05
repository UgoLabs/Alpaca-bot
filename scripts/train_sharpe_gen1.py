import sys
import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import json

# Force UTF-8 for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
# FORCE SETTINGS FOR SHARPE GEN 1
# We disable the "Hard" holding penalty because the Sharpe Reward (Mean-Variance) handles risk naturally.
TrainingConfig.HOLDING_LOSS_PENALTY = 0.0 
TrainingConfig.LOSS_THRESHOLD_PCT = 1.0 # Effectively disabled

# Import the NEW Sharpe Environment
from src.environments.vector_env_sharpe import VectorizedTradingEnv
from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

# =============================================================================
# CONFIGURATION
# =============================================================================
# Gen 4 - Fresh start requested by user (Feb 3, 2026)
MODEL_SAVE_PREFIX = "models/sharpe_gen4_ep"

# Training Parameters
START_EPISODE = 16
NUM_EPISODES = 50  # Fewer episodes but actually training each one
TOTAL_EPISODES = 50 # Fixed total, so we resume correctly

# Fresh Training Rates
LEARNING_RATE_SCALE = 1.0 
EPSILON_START = 1.0
EPSILON_MIN = 0.10
DECAY = 0.95 

# Data
TRAIN_END_DATE = "2023-12-31" # Leave 2024 for validation

# =============================================================================
# DATA LOADER
# =============================================================================
def load_all_data(timeframe="1D", train_end_date="2025-10-31"):
    print(f"📥 Loading Historical Data ({timeframe})...")
    
    if timeframe == "1Min":
        data_dir = "data/historical"
    else:
        # Prefer Swing data for Gen 1 Sharpe
        data_dir = "data/historical_swing"
        
    pattern = os.path.join(data_dir, f"*_{timeframe}.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No data found in {data_dir} for {timeframe}!")
        return None, None
        
    MAX_LEN = 2000
    
    selected_files = files 
    print(f"📊 Training on {len(selected_files)} Symbols")
    
    processed_data = []
    prices_list = []
    
    for f in tqdm(selected_files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            col_map = {'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col.capitalize()] = df[col]
            
            # Filter Date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] <= train_end_date]
                
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 200: 
                continue
                
            # Normalize
            # Explicitly select features to ensure consistency across all files
            # This fixes the "stack expects each tensor to be equal size" error
            feature_names = [
                'sma_10', 'sma_20', 'sma_50', 'sma_200', 
                'atr', 'bb_width', 'bb_upper', 'bb_lower', 
                'ema_12', 'ema_26', 
                'macd', 'macd_signal', 'macd_diff', 
                'adx', 'rsi', 'stoch_k', 'stoch_d', 
                'williams_r', 'roc', 'bb_pband', 
                'volume_sma', 'volume_ratio', 'obv', 'mfi', 'price_vs_sma20'
            ]
            
            # Check if all exist, if not, fill 0
            for c in feature_names:
                if c not in df.columns:
                    df[c] = 0.0
            
            # Get raw data
            raw_data = df[feature_names].values
            
            # CRITICAL: Normalize per-feature to prevent NaN explosion
            # OBV can be 900 million, prices 100-500, RSI 0-100
            mean = np.mean(raw_data, axis=0, keepdims=True)
            std = np.std(raw_data, axis=0, keepdims=True) + 1e-8
            normalized_data = (raw_data - mean) / std
            
            # Clip extreme values to prevent outliers
            normalized_data = np.clip(normalized_data, -10, 10)
            
            # Replace any remaining NaN/inf with 0
            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
            data_t = torch.tensor(normalized_data, dtype=torch.float32)
            
            # Pad or Truncate to MAX_LEN
            if len(data_t) > MAX_LEN:
                data_t = data_t[-MAX_LEN:]
                prices = df['close'].values[-MAX_LEN:]
            else:
                # Pad with zeros (not ideal but rare with <200 filter)
                pad_len = MAX_LEN - len(data_t)
                data_t = torch.cat([torch.zeros(pad_len, data_t.shape[1]), data_t])
                prices = np.pad(df['close'].values, (pad_len, 0), 'edge')
            
            processed_data.append(data_t)
            prices_list.append(prices)
            
        except Exception as e:
            print(f"⚠️ Error {f}: {e}")
            
    if not processed_data:
        return None, None
        
    return torch.stack(processed_data), np.array(prices_list)

# =============================================================================
# MAIN TRAINER
# =============================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Sharpe Gen 1 Training on {device}")
    
    # 1. Load Data
    data_tensor, price_tensor = load_all_data(train_end_date=TRAIN_END_DATE)
    if data_tensor is None:
        return

    # 2. Init Environment
    # Ensure prices are float32 to match internal tensors
    env = VectorizedTradingEnv(data_tensor, torch.tensor(price_tensor, dtype=torch.float32), device=device)
    
    # 3. Init Agent (Fresh Ensemble)
    # state_dim should be num_features (25), not the flattened size, because Transformer accepts (Batch, Seq, Feat)
    state_dim = data_tensor.shape[2]
    action_dim = 3 # HOLD, BUY, SELL
    
    # Vision Channels MUST match state_dim because MultiModalAgent passes TS data to VisionHead
    # VisionHead performs parallel 1D Conv signal processing
    vision_channels = state_dim 
    
    agent = EnsembleAgent(state_dim, vision_channels, action_dim, device=device)
    
    # Sync Batch Size (Reduced to avoid OOM)
    for sub in agent.agents:
        sub.batch_size = 128 # Reduced from 512
    
    # Init Writer
    writer = SummaryWriter(log_dir=f"logs/runs/sharpe_gen4")
    
    # Dummy Text Inputs (Seq Len 64 default)
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_ids_np = dummy_text_ids[0].detach().cpu().numpy()
    dummy_text_mask_np = dummy_text_mask[0].detach().cpu().numpy()
    
    # RESUME LOGIC
    resume_ep = START_EPISODE - 1
    resume_path = f"{MODEL_SAVE_PREFIX}{resume_ep}"
    if resume_ep > 0 and os.path.exists(f"{resume_path}_balanced.pth"):
        print(f"📥 Resuming from {resume_path}...")
        agent.load(resume_path)
        # Put back in train mode because load() forces eval
        for sub in agent.agents:
            sub.policy_net.train()
    
    # 4. Loop
    global_step = 0
    # Calculate correct starting epsilon
    epsilon = max(EPSILON_MIN, EPSILON_START * (DECAY ** (START_EPISODE - 1)))
    print(f"🔄 Starting Epsilon: {epsilon:.4f}")
    
    for episode in range(START_EPISODE, TOTAL_EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        # Update Epsilon for all sub-agents
        for sub in agent.agents:
            sub.epsilon = max(EPSILON_MIN, epsilon)
        
        # Action Stats
        actions_count = {0:0, 1:0, 2:0}
        
        # Fixed number of steps per episode (env auto-resets when done)
        STEPS_PER_EPISODE = 2000
        
        pbar = tqdm(total=STEPS_PER_EPISODE, desc=f"Ep {episode}/{TOTAL_EPISODES} ε={epsilon:.2f}")
        
        for step_ctr in range(STEPS_PER_EPISODE):
            # Select Action
            action = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            # Store experiences - sample more aggressively for better learning
            # Store 50 random environments every step
            state_cpu = state.detach().cpu().numpy()
            n_state_cpu = next_state.detach().cpu().numpy()
            act_cpu = action.detach().cpu().numpy()
            rew_cpu = reward.detach().cpu().numpy()
            done_cpu = done.detach().cpu().numpy() if isinstance(done, torch.Tensor) else np.zeros(env.num_envs, dtype=bool)
            
            # Store 50 samples per step = 100,000 samples per episode
            indices = np.random.choice(env.num_envs, size=min(50, env.num_envs), replace=False)
            for i in indices:
                s = (state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                ns = (n_state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                agent.remember(s, act_cpu[i], rew_cpu[i], ns, done_cpu[i])
            
            # Train every step
            loss_dict = agent.train_step()
            
            state = next_state
            total_reward += reward.mean().item()
            global_step += 1
            
            # Update Stats
            u, c = torch.unique(action, return_counts=True)
            for k, v in zip(u.tolist(), c.tolist()):
                actions_count[k] += v
                
            pbar.update(1)
            
            # Log Loss occasionally
            if global_step % 100 == 0 and loss_dict:
                writer.add_scalar("Loss/Aggressive", loss_dict.get("aggressive", 0), global_step)
                writer.add_scalar("Loss/Conservative", loss_dict.get("conservative", 0), global_step)
        
        pbar.close()
        
        pbar.close()
        
        # Decay Epsilon
        epsilon = max(EPSILON_MIN, epsilon * DECAY)
        
        # Log Metrics
        avg_reward = total_reward / STEPS_PER_EPISODE
        
        writer.add_scalar("Reward/Average", avg_reward, episode)
        
        # Portfolio Metrics
        # Calculate Final Equity
        current_step_safe = torch.clamp(env.current_step, max=env.prices.shape[1]-1)
        current_prices = env.prices[env._row_indices, current_step_safe]
        final_equity = env.balance + (env.shares * current_prices)
        
        returns_pct = (final_equity - env.initial_balance) / env.initial_balance
        avg_return = returns_pct.mean().item()
        
        print(f"✅ Ep {episode} | Avg Reward: {avg_reward:.4f} | Avg Return: {avg_return*100:.2f}%")
        print(f"Action Dist: {actions_count}")
        
        # Save Model - EVERY EPISODE for testing
        path = f"{MODEL_SAVE_PREFIX}{episode}"
        agent.save(path)
        print(f"💾 Saved to {path}")
        
        # Free memory
        torch.cuda.empty_cache()

    print("🏁 Training Complete!")

if __name__ == "__main__":
    train()
