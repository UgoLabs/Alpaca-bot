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
# FORCE UNBOUND SETTINGS FOR FINETUNING
TrainingConfig.USE_TRAILING_STOP = False
TrainingConfig.USE_PROFIT_TAKE = False
TrainingConfig.WINDOW_SIZE = 60
TrainingConfig.REGIME_REWARD_MULT = 2.0

# GEN12 SETTINGS
# We rely on the Quadratic Underwater Penalty in Gen12 env
# These config values might be used inside for legacy reasons, but env_gen12 has hardcoded logic too
TrainingConfig.HOLDING_LOSS_PENALTY = 0.0001
TrainingConfig.LOSS_THRESHOLD_PCT = 0.00 # Immediate penalty

# Import the NEW Gen12 Environment
from src.environments.vector_env_gen12 import VectorizedTradingEnv
from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

# =============================================================================
# CONFIGURATION
# =============================================================================
# Start from Ep 482 (The Transition Model)
# Gen13 Strategy: Positive Reinforcement for Sniper Entries
TARGET_MODEL_PATH = "models/swing_gen11_risk_aware_ep482" 
NEW_PREFIX = "models/swing_gen13_sniper_ep"

START_EPISODE = 520
NUM_EPISODES = 20  
TOTAL_EPISODES = START_EPISODE + NUM_EPISODES - 1

LEARNING_RATE_SCALE = 0.20
EPSILON_START = 0.05
EPSILON_MIN = 0.01
DECAY = 0.95

# =============================================================================
# DATA LOADER (Same as Gen10)
# =============================================================================
def load_all_data(timeframe="1D", train_end_date="2025-10-31"):
    print(f"📥 Loading Historical Data ({timeframe})...")
    
    if timeframe == "1Min":
        data_dir = "data/historical"
    else:
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
                
            raw_close = df['Close'].values[-MAX_LEN:]
            
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12'
            ]
            
            # Normalization
            data = df[feature_cols].values[-MAX_LEN:]
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0) + 1e-8
            norm_data = (data - means) / stds
            
            # Pad
            if len(norm_data) < MAX_LEN:
                pad_len = MAX_LEN - len(norm_data)
                padding = np.zeros((pad_len, len(feature_cols)))
                norm_data = np.vstack([padding, norm_data])
                raw_close = np.concatenate([np.zeros(pad_len), raw_close])
            
            processed_data.append(norm_data)
            prices_list.append(raw_close)
            
        except Exception as e:
            continue
            
    if not processed_data:
        return None, None
        
    return np.array(processed_data), np.array(prices_list)

# =============================================================================
# TRAINING LOOP
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Gen11 RISK-AWARE Finetuning on {device}")
    
    # Load Data
    data_tensor, price_tensor = load_all_data()
    if data_tensor is None:
        return
        
    data_tensor = torch.FloatTensor(data_tensor)
    price_tensor = torch.FloatTensor(price_tensor)
    
    # Initialize Gen11 Env
    env = VectorizedTradingEnv(data_tensor, price_tensor, device=device, position_pct=1.0)
    
    # Initialize Agent
    state_dim = env.num_features
    action_dim = 3
    agent = EnsembleAgent(state_dim, state_dim, action_dim, device=device)
    
    # Load Checkpoint Ep 410 (The Berserker)
    agent.load(TARGET_MODEL_PATH)

    # Scale Learning Rates
    for sub_agent in agent.agents:
        for param_group in sub_agent.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * LEARNING_RATE_SCALE
        sub_agent.epsilon = EPSILON_START
    
    writer = SummaryWriter(log_dir=f"logs/runs/gen11_risk_aware")
    
    print(f"🏋️ Training for {NUM_EPISODES} episodes (Ep {START_EPISODE} -> {TOTAL_EPISODES})")
    
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_ids_np = dummy_text_ids[0].detach().cpu().numpy()
    dummy_text_mask_np = dummy_text_mask[0].detach().cpu().numpy()
    
    for episode in range(START_EPISODE, TOTAL_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False
        step_ctr = 0
        
        pbar = tqdm(total=env.total_steps, desc=f"Ep {episode}/{TOTAL_EPISODES}", leave=False)
        
        while not done:
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            next_state, reward, done_flag, info = env.step(actions) 
            
            if env.current_step[0].item() >= env.total_steps - 2:
                done = True
                
            if step_ctr % 20 == 0:
                state_cpu = state.detach().cpu().numpy()
                next_state_cpu = next_state.detach().cpu().numpy()
                actions_cpu = actions.detach().cpu().numpy()
                rewards_cpu = reward.detach().cpu().numpy()
                dones_cpu = done_flag.detach().cpu().numpy()
                
                for i in range(env.num_envs):
                     s = (state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                     ns = (next_state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                     agent.remember(s, actions_cpu[i], rewards_cpu[i], ns, dones_cpu[i])

            if step_ctr % 20 == 0:
                losses = agent.train_step()
                if losses:
                    writer.add_scalar("Loss/train", losses.get('balanced', 0), episode * env.total_steps + step_ctr)
            
            state = next_state
            total_reward += reward.sum().item()
            pbar.update(1)
            step_ctr += 1
            
        pbar.close()
        
        for sub_agent in agent.agents:
            if sub_agent.epsilon > EPSILON_MIN:
                sub_agent.epsilon *= DECAY
            
        avg_reward = total_reward / env.num_envs
        writer.add_scalar("Reward/episode", avg_reward, episode)
        
        print(f"✅ Ep {episode} | Avg Reward: {avg_reward:.4f} | Epsilon: {agent.agents[0].epsilon:.4f}")
        
        # Save EVERY episode for granular analysis
        save_path = f"{NEW_PREFIX}{episode}"
        agent.save(save_path)
        
        # CSV Log
        csv_path = f"logs/trades_swing_gen13_ep{episode}.csv"
        with open(csv_path, 'w') as f:
            f.write("episode,reward\n")
            f.write(f"{episode},{avg_reward}\n")

    print("🏁 Finetuning Complete!")
    writer.close()

if __name__ == "__main__":
    main()
