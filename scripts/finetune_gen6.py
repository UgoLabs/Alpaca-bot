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
TARGET_MODEL_PATH = "models/swing_gen6_finetune_aggressive_update_ep350"  # Note: No suffix, agent.load adds it
NEW_PREFIX = "models/swing_gen7_refined_ep"
START_EPISODE = 351
TOTAL_EPISODES = 400  # Train for 50 more episodes
LEARNING_RATE_SCALE = 0.5  # Lower LR for fine-tuning (if supported, else we rely on optimizer state if saved, or just standard adam)
EPSILON_START = 0.05  # Low exploration, mostly exploit strict high performance
EPSILON_MIN = 0.01
DECAY = 0.99

# =============================================================================
# DATA LOADER (Copied from train_multimodal.py for standalone execution)
# =============================================================================
def load_all_data(timeframe="1D", train_end_date="2025-10-31"):
    """
    Load data with train/test split. 
    Extended cutoff to Oct 2025. 
    CRITICAL: We reserve Nov 2025 - Jan 2026 as a "Final Exam" (Hold-out set).
    This allows us to backtest Gen 7 on unseen data to prove it is actually better.
    """
    print(f"📥 Loading Historical Data ({timeframe})...")
    print(f"📅 Training cutoff: {train_end_date}")
    print(f"🛑 HOLD-OUT SET: Data form Nov 1, 2025 onwards is HIDDEN from training.")

    
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
    
    # User requested: Top 100, Tech Heavy, Diversified, Important Symbols
    PRIORITY_SYMBOLS = [
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 
        'AVGO', 'QCOM', 'INTC', 'CRM', 'ADBE', 'ORCL', 'IBM', 'NFLX',
        'PLTR', 'SNOW', 'U', 'COIN', 'SHOP', 'SQ', 'ROKU', 'NET', 'CRWD', 'PANW',
        'TSM', 'MU', 'LRCX', 'AMAT', 'KLAC',
        'SPY', 'QQQ', 'IWM', 'DIA', 'ARKK', 'SMH', 'XLK', 'XLF',
        'JPM', 'BAC', 'GS', 'V', 'MA', 'BLK',
        'WMT', 'COST', 'HD', 'NKE', 'SBUX', 'MCD', 'DIS',
        'JNJ', 'PFE', 'UNH', 'LLY', 'ABBV',
        'CAT', 'BA', 'GE', 'XOM', 'CVX',
        'MSTR', 'MARA', 'RIOT'
    ]
    
    # Categorize files could go here, but we are fine-tuning so we take all.
    selected_files = files 
    
    import random
    random.shuffle(selected_files)
    
    print(f"📊 Training on {len(selected_files)} Symbols")
    
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
            
            # Filter Date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] <= train_end_date]
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[df['Date'] <= train_end_date]
                
            # Add Indicators
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 500: # Less strict for fine tuning
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
            
            # Normalize
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
            # Pad
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

def train():
    import random # re-import
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting GEN 7 Fine-Tuning on {device}")
    
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
    print(f"📂 Loading Target Model: {TARGET_MODEL_PATH}...")
    try:
        agent.load(TARGET_MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 5. Set Fine-Tuning Hyperparameters
    print(f"🔧 Configuring Fine-Tuning params: Eps={EPSILON_START}, Decay={DECAY}")
    for sub_agent in agent.agents:
        sub_agent.epsilon = EPSILON_START
        sub_agent.epsilon_decay = DECAY
        sub_agent.epsilon_min = EPSILON_MIN
        # Optionally lower learning rate here if accessible, e.g.:
        # for param_group in sub_agent.optimizer.param_groups:
        #     param_group['lr'] *= LEARNING_RATE_SCALE

    # 6. Training Loop
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    
    writer = SummaryWriter(log_dir="logs/runs/gen7_finetune")
    
    # Pre-compute numpy versions for storage
    dummy_text_ids_np = dummy_text_ids[0].cpu().numpy()
    dummy_text_mask_np = dummy_text_mask[0].cpu().numpy()

    for episode in range(START_EPISODE, TOTAL_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        steps = env.total_steps
        
        pbar = tqdm(total=steps, desc=f"Fine-Tuning Ep {episode}/{TOTAL_EPISODES}")
        
        for step_idx in range(steps):
            # High-Performance Act
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            
            # Step
            next_state, rewards, dones, _ = env.step(actions)
            
            # Train (Replay Buffer)
            # Store 10% of transitions per step to reduce memory overhead
            sample_size = max(1, env.num_envs // 10)
            sample_indices = np.random.choice(env.num_envs, sample_size, replace=False)
            
            state_cpu = state[sample_indices].cpu().numpy()
            next_state_cpu = next_state[sample_indices].cpu().numpy()
            actions_cpu = actions[sample_indices].cpu().numpy()
            rewards_cpu = rewards[sample_indices].cpu().numpy()
            dones_cpu = dones[sample_indices].cpu().numpy()
            
            # Memorize line by line (matching agent interface)
            for i in range(sample_size):
                s_tuple = (state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                ns_tuple = (next_state_cpu[i], dummy_text_ids_np, dummy_text_mask_np)
                agent.remember(s_tuple, actions_cpu[i], rewards_cpu[i], ns_tuple, dones_cpu[i])
            
            # Learn
            losses = agent.train_step()
            
            # Log Loss
            if step_idx % 100 == 0:
                global_step = (episode * steps) + step_idx
                if losses:
                    for name, loss in losses.items():
                        writer.add_scalar(f"Loss/{name}", loss, global_step)

            state = next_state
            total_reward += rewards.sum().item()
            pbar.update(1)
            pbar.set_postfix({'Reward': f"{total_reward:.2f}", 'Loss': f"{list(losses.values())[0] if losses else 0:.4f}"})
        
        pbar.close()
        
        # Log Reward
        writer.add_scalar("Reward/Episode", total_reward, episode)
        
        # Decay Epsilon
        for sub_agent in agent.agents:
            sub_agent.epsilon = max(sub_agent.epsilon_min, sub_agent.epsilon * DECAY)
            
        print(f"📉 Epsilon decayed to {agent.balanced.epsilon:.4f}")
        
        # Save Checkpoint every 5 eps or if specifically requested
        if episode % 5 == 0:
            save_path = f"{NEW_PREFIX}{episode}"
            print(f"💾 Saving Finetuned Checkpoint: {save_path}")
            agent.save(save_path)

if __name__ == "__main__":
    train()
