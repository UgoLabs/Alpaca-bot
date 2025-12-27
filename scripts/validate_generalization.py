import torch
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ensemble_agent import EnsembleAgent
from src.environments.vector_env import VectorizedTradingEnv
from src.core.indicators import add_technical_indicators

# Symbols the model has NEVER seen
TEST_SYMBOLS = ['UBER', 'SQ', 'SHOP', 'ROKU', 'PINS', 'TWLO', 'COIN', 'PLTR']

def fetch_and_process_test_data(symbols):
    print(f"📥 Fetching test data for {len(symbols)} unseen symbols...")
    data_list = []
    price_list = []
    MAX_LEN = 5000
    
    for symbol in tqdm(symbols):
        try:
            # Fetch 2 years of daily data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y", interval="1d")
            
            if df.empty:
                continue
                
            # Standardize
            df = df.reset_index()
            # Ensure Date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df.set_index('Date', inplace=True)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Add Indicators
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 100:
                continue
                
            # Extract Raw Prices
            raw_close = df['Close'].values[-MAX_LEN:]
            
            # Extract Features
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
            
        except Exception as e:
            print(f"Error {symbol}: {e}")
            continue
            
    if not data_list:
        return None, None
        
    return torch.FloatTensor(np.array(data_list)), torch.FloatTensor(np.array(price_list))

def validate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧪 Starting Validation on {device}...")
    
    # 1. Load Data
    data, prices = fetch_and_process_test_data(TEST_SYMBOLS)
    if data is None:
        print("❌ Failed to load test data")
        return
        
    print(f"📊 Test Data Shape: {data.shape}")
    
    # 2. Initialize Environment
    env = VectorizedTradingEnv(data, prices, device=device)
    
    # 3. Initialize Agent
    num_features = data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )
    
    # 4. Load BEST Model
    model_path = "models/swing_best"
    print(f"📂 Loading model from {model_path}...")
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 5. Run Evaluation
    print("🚀 Running Evaluation Episode...")
    
    # Dummy Text
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    
    state = env.reset()
    total_reward = 0
    steps = env.total_steps
    
    # Tracking
    portfolio_values = []
    
    for _ in tqdm(range(steps)):
        # Act deterministically (eval_mode=True)
        actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask) # Note: batch_act uses epsilon, need to force greedy
        
        # Force greedy for validation
        # We can manually set epsilon to 0 temporarily
        for sub in agent.agents:
            sub.epsilon = 0.0
            
        # Step
        next_state, rewards, dones, infos = env.step(actions)
        
        total_reward += rewards.sum().item()
        state = next_state
        
    avg_reward = total_reward / env.num_envs
    print(f"\n🏆 Validation Results on Unseen Data:")
    print(f"   Total Reward (Sum): {total_reward:.4f}")
    print(f"   Average Reward per Symbol: {avg_reward:.4f}")
    
    if avg_reward > 0:
        print("\n✅ PASSED: Model generalizes to new data!")
    else:
        print("\n⚠️ WARNING: Model lost money on new data. Likely overfitting.")

if __name__ == "__main__":
    validate()
