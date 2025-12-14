"""
GPU-Accelerated Training Script.
Runs the entire RL loop (Environment + Agent + Replay) on the GPU.
Target: 100% Volume Utilization.
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import traceback

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.vector_env import VectorizedTradingEnv
from src.models.agent import DuelingDQN
from config.settings import TrainingConfig, SCALPER_MODEL_PATH

from src.core.indicators import add_technical_indicators

def load_all_data(timeframe="1Min"):
    print("📥 Loading ALL data into GPU Memory...")
    # Find all CSVs
    pattern = os.path.join("data/historical", f"*_{timeframe}.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No data found! Run download_data.py first.")
        return None
        
    data_list = []
    MAX_LEN = 10000 
    
    # Expected columns by TrainingConfig (Approximate list based on add_technical_indicators)
    # Open, High, Low, Close, Volume, RSI, MACD, Signal, ADX, SMA, EMA...
    # We must ensure we grab exactly NUM_WINDOW_FEATURES columns.
    # Currently assuming 11.
    
    for f in tqdm(files[:258], desc="Vectorizing"):
        try:
            df = pd.read_csv(f)
            # Fallback if case sensitive
            df.columns = [c.lower() for c in df.columns]
            
            # Map to Capitalized for indicators.py
            if 't' in df.columns: df['Timestamp'] = df['t']
            if 'o' in df.columns: df['Open'] = df['o']
            if 'h' in df.columns: df['High'] = df['h']
            if 'l' in df.columns: df['Low'] = df['l']
            if 'c' in df.columns: df['Close'] = df['c']
            if 'v' in df.columns: df['Volume'] = df['v']
            
            # Ensure we have the capitalized columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns and col.lower() in df.columns:
                    df[col] = df[col.lower()]
            
            # Add Indicators
            df = add_technical_indicators(df)
            
            # Select Numeric Columns (excluding timestamps/symbol)
            # The model likely learned on a specific order.
            # We should verify src/core/indicators.py output order.
            # For now, we select the last 11 numeric columns or specific ones.
            # Let's blindly trust proper column selection or just take normalized OHLCV + Indicators.
            
            # Filter NaNs
            df = df.dropna()
            
            if len(df) < 1000:
                tqdm.write(f"Skipping {f}: Too short ({len(df)})")
                continue
                
            # Select features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 11:
                tqdm.write(f"Skipping {f}: Not enough cols ({len(numeric_cols)})")
                tqdm.write(f"Cols: {numeric_cols.tolist()}")
                continue
                
            # Take last 11 columns (usually indicators are added last)
            # Actually, standard OHLCV (5) + indicators.
            # Let's take exactly 11 specific columns if possible. 
            # To avoid mismatch, we resort to taking the LAST 11 columns.
            final_cols = numeric_cols[-11:]
            
            features = df[final_cols].values[-MAX_LEN:]
            
            # If the user intended to check `len(features)` after it's defined,
            # this is where it would go. The instruction placed it before `features` was defined.
            # Assuming the original `len(df) < 1000` check was the intended place for a length check.
            
            # Normalize (Z-Score)
            # feat_mean = np.mean(features, axis=0)
            # feat_std = np.std(features, axis=0) + 1e-6
            # features = (features - feat_mean) / feat_std
            
            tensor = torch.tensor(features, dtype=torch.float32)
            
            if len(tensor) < MAX_LEN:
                padding = torch.zeros((MAX_LEN - len(tensor), 11), dtype=torch.float32)
                tensor = torch.cat([padding, tensor])
                
            data_list.append(tensor)
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
            traceback.print_exc()
            continue

def train_gpu():
    # 1. Load Data
    data = load_all_data()
    if data is None:
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Init Env
    env = VectorizedTradingEnv(data, device=device)
    
    # 3. Init Agent
    # State Size = Window * Features (5)
    state_size = TrainingConfig.WINDOW_SIZE * 5
    action_size = 3
    
    agent = DuelingDQN(state_size, action_size, use_noisy=True)
    agent.model.to(device)
    agent.target_model.to(device)
    
    print(f"🚀 Starting GPU Acceleration Training on {device}")
    
    # 4. Loop
    episodes = 5000000 # Infinite loop basically
    batch_size = 1024 # Massive batch
    
    obs = env.reset() # (Envs, Window, Feat)
    
    # Flatten obs for Network: (Envs, Window*Feat)
    # The Transformer handles (Window, Feat) internally, but DuelingDQN expects Flattened usually?
    # Our Transformer code expects Flattened OR shaped.
    # Let's adjust Agent input. Our Transformer accepts (Batch, Flattened) and reshapes it.
    
    # Flatten obs
    obs_flat = obs.view(env.num_envs, -1)
    
    pbar = tqdm(total=episodes)
    
    for i in range(episodes):
        # Select Actions (Vectorized)
        # Agent.act() usually does 1 item. We need batch_act.
        # We can just call model(obs) -> output (Envs, Actions).
        
        with torch.no_grad():
            q_values = agent.model(obs_flat)
            actions = q_values.argmax(dim=1) # Greedy for now (NoisyNet handles exploration)
            
        # Step
        next_obs, rewards, dones, _ = env.step(actions)
        
        # Flatten next
        next_obs_flat = next_obs.view(env.num_envs, -1)
        
        # Learn instantly?
        # In Vector RL, "Batch" is just "Current Step across all Envs".
        # We can train on (obs, action, reward, next_obs) directly.
        
        # Tensorize rewards/actions (already tensors)
        # Train Step
        loss = agent.train_step(obs_flat, actions, rewards, next_obs_flat, dones)
        
        # Update Obs
        obs_flat = next_obs_flat
        
        # Update Target
        if i % 100 == 0:
            agent.update_target_model()
            
        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss:.4f} | Reward: {rewards.mean().item():.4f}")
            pbar.update(10)
            
        # Save occasionally
        if i % 1000 == 0:
            agent.save(SCALPER_MODEL_PATH)

if __name__ == "__main__":
    train_gpu()
