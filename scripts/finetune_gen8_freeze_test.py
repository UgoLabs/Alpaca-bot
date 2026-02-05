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
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.vector_env import VectorizedTradingEnv
from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

# =============================================================================
# CONFIGURATION
# =============================================================================
# The user wants to start from the 380 checkpoint
TARGET_MODEL_PATH = "models/trades_swing_gen7_refined_ep380" 
NEW_PREFIX = "models/trades_swing_gen8_freeze_ep"

START_EPISODE = 381
NUM_FREEZE_EPISODES = 10
NUM_UNFREEZE_EPISODES = 40
TOTAL_EPISODES = START_EPISODE + NUM_FREEZE_EPISODES + NUM_UNFREEZE_EPISODES - 1 # 381 + 10 + 40 - 1 = 430

# Freeze Cutoff
FREEZE_UNTIL_EPISODE = START_EPISODE + NUM_FREEZE_EPISODES - 1 # 381 -> 390 (inclusive)

LEARNING_RATE_SCALE = 0.5 
EPSILON_START = 0.05
EPSILON_MIN = 0.01
DECAY = 0.99

# =============================================================================
# DATA LOADER
# =============================================================================
def load_all_data(timeframe="1D", train_end_date="2025-02-01"):
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
        
    data_list = []
    price_list = []
    MAX_LEN = 2000
    
    selected_files = files 
    import random
    random.shuffle(selected_files)
    
    print(f"📊 Training on {len(selected_files)} Symbols")
    
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
            
            # Filter Date (Generic cutoff)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] <= train_end_date]
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[df['Date'] <= train_end_date]
                
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 200: 
                continue
                
            raw_close = df['Close'].values[-MAX_LEN:]
            
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12'
            ]
            
            valid_cols = [c for c in feature_cols if c in df.columns]
            if len(valid_cols) < 5:
                continue
                
            features = df[valid_cols].values[-MAX_LEN:]
            
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
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
        
    data_tensor = torch.FloatTensor(np.array(data_list))
    price_tensor = torch.FloatTensor(np.array(price_list))
    
    return data_tensor, price_tensor

def set_feature_extractors_trainable(agent_ensemble, trainable=True):
    """
    Freeze/Unfreeze the 'heads' (feature extractors) of the model.
    When trainable=False, only the fusion layer and decision heads are updated.
    """
    count_frozen = 0
    count_unfrozen = 0
    
    for sub_agent in agent_ensemble.agents: # Aggressive, Conservative, Balanced
         # policy_net is MultiModalAgent
         model = sub_agent.policy_net
         
         # 1. TS Head (Transformer)
         for param in model.ts_head.parameters():
             param.requires_grad = trainable
         
         # 2. Vision Head (ResNet1D)
         for param in model.vision_head.parameters():
             param.requires_grad = trainable
             
         # 3. Text Head
         for param in model.text_head.parameters():
             param.requires_grad = trainable
         
         # 4. Fusion Layer - ALWAYS UNPROZEN
         for param in model.fusion_layer.parameters():
             param.requires_grad = True # Always learn how to combine
             
         # 5. Output Streams - ALWAYS UNFROZEN
         for param in model.value_stream.parameters():
             param.requires_grad = True
         for param in model.advantage_stream.parameters():
             param.requires_grad = True
    
    state = "UNFROZEN" if trainable else "FROZEN"
    print(f"🧊 Feature Extractors are now {state}. High-level reasoning layers remain trainable.")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting GEN 8 Freeze Experiment on {device}")
    print(f"🎯 Target Model: {TARGET_MODEL_PATH}")
    print(f"❄️ Schedule: Freeze First {NUM_FREEZE_EPISODES} Eps | Unfreeze Next {NUM_UNFREEZE_EPISODES} Eps")
    
    # 1. Load Data
    data, prices = load_all_data(timeframe="1D") 
    
    if data is None:
        return

    # 2. Environment (Live Params)
    env = VectorizedTradingEnv(data, prices, device=device, position_pct=0.10)
    env.initial_balance = 10000.0
    env.balance.fill_(10000.0)
    
    # 3. Initialize Agent
    num_features = data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )
    
    # 4. LOAD THE SMART BRAIN
    try:
        agent.load(TARGET_MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 5. Set Fine-Tuning Hyperparameters
    for sub_agent in agent.agents:
        sub_agent.epsilon = EPSILON_START
        sub_agent.epsilon_decay = DECAY
        sub_agent.epsilon_min = EPSILON_MIN

    # 6. Training Loop
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    
    dummy_text_ids_np = dummy_text_ids[0].cpu().numpy()
    dummy_text_mask_np = dummy_text_mask[0].cpu().numpy()
    
    writer = SummaryWriter(log_dir="logs/runs/gen8_freeze")

    for episode in range(START_EPISODE, TOTAL_EPISODES + 1):
        
        # --- FREEZE / UNFREEZE LOGIC ---
        if episode == START_EPISODE:
            print(f"🔒 Episode {episode}: Initializing FREEZE PHASE (1/{NUM_FREEZE_EPISODES}).")
            set_feature_extractors_trainable(agent, trainable=False)
        elif episode == FREEZE_UNTIL_EPISODE + 1:
            print(f"🔓 Episode {episode}: SWITCHING TO UNFREEZE PHASE (1/{NUM_UNFREEZE_EPISODES}).")
            set_feature_extractors_trainable(agent, trainable=True)
        # -------------------------------
        
        state = env.reset()
        total_reward = 0
        steps = env.total_steps
        
        pbar = tqdm(total=steps, desc=f"Ep {episode}/{TOTAL_EPISODES}")
        
        for step_idx in range(steps):
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            next_state, rewards, dones, _ = env.step(actions)
            
            # Train (Replay Buffer)
            sample_size = max(1, env.num_envs // 10)
            sample_indices = np.random.choice(env.num_envs, sample_size, replace=False)
            
            state_cpu = state[sample_indices].cpu().numpy()
            next_state_cpu = next_state[sample_indices].cpu().numpy()
            actions_cpu = actions[sample_indices].cpu().numpy()
            rewards_cpu = rewards[sample_indices].cpu().numpy()
            dones_cpu = dones[sample_indices].cpu().numpy()
            
            for i in range(sample_size):
                s_tuple = (state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                ns_tuple = (next_state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                agent.remember(s_tuple, actions_cpu[i], rewards_cpu[i], ns_tuple, dones_cpu[i])
            
            losses = agent.train_step()
            
            if step_idx % 100 == 0 and losses:
                 for name, loss in losses.items():
                        writer.add_scalar(f"Loss/{name}", loss, (episode * steps) + step_idx)

            state = next_state
            total_reward += rewards.sum().item()
            pbar.update(1)
            pbar.set_postfix({'Reward': f"{total_reward:.2f}"})
        
        pbar.close()
        writer.add_scalar("Reward/Episode", total_reward, episode)
        
        for sub_agent in agent.agents:
            sub_agent.epsilon = max(sub_agent.epsilon_min, sub_agent.epsilon * DECAY)
            
        print(f"📉 Epsilon: {agent.balanced.epsilon:.4f}")
        
        # Save Frequency
        if episode % 5 == 0 or episode == TOTAL_EPISODES:
            save_path = f"{NEW_PREFIX}{episode}"
            print(f"💾 Saving: {save_path}")
            agent.save(save_path)

if __name__ == "__main__":
    train()