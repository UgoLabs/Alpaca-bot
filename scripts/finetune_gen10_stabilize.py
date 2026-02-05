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
TrainingConfig.REGIME_REWARD_MULT = 2.0  # Reward good moves in bear markets

# STABILIZATION SETTINGS
TrainingConfig.HOLDING_LOSS_PENALTY = 0.0005  # 5x stronger penalty for holding losers (was 0.0001)
TrainingConfig.LOSS_THRESHOLD_PCT = 0.01      # Penalize starting at -1% instead of -2%
TrainingConfig.EXIT_PROFIT_BONUS = 0.005      # Keep the exit bonus

from src.environments.vector_env import VectorizedTradingEnv
from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

# =============================================================================
# CONFIGURATION
# =============================================================================
# Start from the High-Risk, High-Reward Model (Ep 410)
# We want to tame "The Berserker"
TARGET_MODEL_PATH = "models/swing_gen9_unbound_ep410" 
NEW_PREFIX = "models/swing_gen10_stable_ep"

START_EPISODE = 411
NUM_EPISODES = 20
TOTAL_EPISODES = START_EPISODE + NUM_EPISODES - 1

# Hyperparameters for Finetuning
# Very low learning rate to nudge weights without destroying the +259% knowledge
LEARNING_RATE_SCALE = 0.25 # Relative to base LR
EPSILON_START = 0.05       # Low exploration (exploitation focus)
EPSILON_MIN = 0.01
DECAY = 0.95

# =============================================================================
# DATA LOADER
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
    # Use ALL files for finetuning to ensure generalization
    
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
            
            # Pad if shorter than MAX_LEN
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
    print(f"🚀 Starting Gen10 STABILIZE Finetuning on {device}")
    
    # Load Data
    data_tensor, price_tensor = load_all_data()
    if data_tensor is None:
        return
        
    data_tensor = torch.FloatTensor(data_tensor)
    price_tensor = torch.FloatTensor(price_tensor)
    
    # Initialize Environment
    # position_pct=1.0 per env (effectively managing 1 position per slot)
    env = VectorizedTradingEnv(data_tensor, price_tensor, device=device, position_pct=1.0)
    
    # Initialize Agent
    state_dim = env.num_features
    action_dim = 3
    # Same architecture
    agent = EnsembleAgent(state_dim, state_dim, action_dim, device=device)
    
    # Load Checkpoint
    agent.load(TARGET_MODEL_PATH)

    # Adjust Hyperparameters for Finetuning
    # Scale down learning rate
    for sub_agent in agent.agents:
        for param_group in sub_agent.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * LEARNING_RATE_SCALE
        sub_agent.epsilon = EPSILON_START
    
    writer = SummaryWriter(log_dir=f"logs/runs/gen10_stabilize")
    
    print(f"🏋️ Training for {NUM_EPISODES} episodes (Ep {START_EPISODE} -> {TOTAL_EPISODES})")
    
    # Create Dummy Text Data for Act
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_ids_np = dummy_text_ids[0].detach().cpu().numpy()
    dummy_text_mask_np = dummy_text_mask[0].detach().cpu().numpy()
    
    best_reward = -float('inf')
    
    for episode in range(START_EPISODE, TOTAL_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        # tqdm bar for steps inside episode
        pbar = tqdm(total=env.total_steps, desc=f"Ep {episode}/{TOTAL_EPISODES}", leave=False)
        
        # We need a step counter for training frequency
        step_ctr = 0
        
        while not done:
            # Select Action (Use batch_act for better ensemble logic, or act for simple voting)
            # act() computes voting. batch_act also computes voting but with bias logic.
            # Using act() to be consistent with simple finetuning unless batch_act is preferred.
            # train_swing_phase2 uses batch_act. Let's use batch_act.
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            
            # Step Env
            next_state, reward, done_flag, info = env.step(actions) 
            
            # Check if environment is finished
            current_step_idx = env.current_step[0].item()
            if current_step_idx >= env.total_steps - 2:
                done = True
                
            # Store Transition (Cpu conversion is slow, do it every N steps)
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

            # Train (Replay)
            if step_ctr % 20 == 0:
                losses = agent.train_step()
                if losses:
                    writer.add_scalar("Loss/train", losses.get('balanced', 0), episode * env.total_steps + current_step_idx)
            
            state = next_state
            total_reward += reward.sum().item()
            pbar.update(1)
            step_ctr += 1
            
        pbar.close()
        
        # Decay Epsilon
        for sub_agent in agent.agents:
            if sub_agent.epsilon > EPSILON_MIN:
                sub_agent.epsilon *= DECAY
            
        avg_reward = total_reward / env.num_envs
        writer.add_scalar("Reward/episode", avg_reward, episode)
        
        print(f"✅ Ep {episode} | Avg Reward: {avg_reward:.4f} | Epsilon: {agent.agents[0].epsilon:.4f}")
        
        # Save Regular Checkpoints
        if episode % 5 == 0:
            save_path = f"{NEW_PREFIX}{episode}" # .save appends suffix
            agent.save(save_path)
            
            # CSV Log
            csv_path = f"logs/trades_swing_gen10_stabilize_ep{episode}.csv"
            # (Just a dummy file to mark progress)
            with open(csv_path, 'w') as f:
                f.write("episode,reward\n")
                f.write(f"{episode},{avg_reward}\n")

    print("🏁 Finetuning Complete!")
    writer.close()

if __name__ == "__main__":
    main()
