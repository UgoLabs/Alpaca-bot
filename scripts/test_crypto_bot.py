import sys
import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Force UTF-8 for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
# Override Window Size for Crypto
TrainingConfig.WINDOW_SIZE = 60

from src.environments.vector_env import VectorizedTradingEnv
from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

def load_crypto_data(start_date: str, end_date: str, max_steps: int = 10000):
    print(f"📥 Loading Historical Crypto Data ({start_date} to {end_date})...")
    pattern = os.path.join("data/historical", "*_1D.csv")
    all_files = glob.glob(pattern)
    
    # Filter for likely crypto symbols
    watchlist_path = "config/watchlists/crypto_watchlist.txt"
    if os.path.exists(watchlist_path):
        with open(watchlist_path, 'r') as f:
            crypto_symbols = [line.strip().replace('/', '').replace('-', '') for line in f if line.strip()]
    else:
        crypto_symbols = ["BTC", "ETH", "SOL", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "XRP", "BNB"]
        
    files = []
    for f in all_files:
        filename = os.path.basename(f).upper()
        found = False
        for sym in crypto_symbols:
             if sym in filename:
                 found = True
                 break
        if found:
            files.append(f)
            
    if not files:
        print("❌ No crypto data found!")
        return None, None, [], 0
        
    print(f"📋 Found {len(files)} crypto data files for range.")
        
    data_list = []
    price_list = []
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for f in tqdm(files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            col_map = {'date': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 'adj close': 'Close'}
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
            
            if 'Date' not in df.columns:
                for c in df.columns:
                    if 'date' in c.lower() or 'time' in c.lower():
                        df['Date'] = df[c]
                        break
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            mask = (df['Date'] >= start_dt) & (df['Date'] < end_dt)
            df = df.loc[mask].copy()
            
            if len(df) < 60:
                continue

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col.lower() in df.columns:
                    df[col.capitalize()] = df[col.lower()]
            
            # Fill missing
            df = df.ffill().bfill()

            # Add Indicators
            df = add_technical_indicators(df)
            
            df.index = df['Date']
            day = df.index.dayofweek
            month = df.index.month
            
            df['sin_day'] = np.sin(2 * np.pi * day / 7)
            df['cos_day'] = np.cos(2 * np.pi * day / 7)
            df['sin_month'] = np.sin(2 * np.pi * month / 12)
            df['cos_month'] = np.cos(2 * np.pi * month / 12)
            df['sin_hour'] = 0.0
            df['cos_hour'] = 0.0

            df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
            df['volatility'] = df['log_ret'].rolling(window=20).std().fillna(0)
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 100:
                continue

            raw_close = df['Close'].values
            
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12',
                'atr', 'bb_width',
                'sin_day', 'cos_day', 'sin_month', 'cos_month',
                'volatility'
            ]
            
            valid_cols = [c for c in feature_cols if c in df.columns]
            features = df[valid_cols].values
            
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
            data_list.append(norm_features)
            price_list.append(raw_close)
            
        except Exception:
            continue
            
    if not data_list:
        return None, None, [], 0

    lengths = [len(x) for x in data_list]
    max_len = max(lengths)
    final_steps = max_len
    
    padded_data = []
    padded_prices = []
    
    for i in range(len(data_list)):
        d = data_list[i]
        p = price_list[i]
        curr_len = len(d)
        
        if curr_len < max_len:
            pad_len = max_len - curr_len
            d_pad = np.pad(d, ((0, pad_len), (0, 0)), mode='edge')
            p_pad = np.pad(p, (0, pad_len), mode='edge')
            padded_data.append(d_pad)
            padded_prices.append(p_pad)
        else:
            padded_data.append(d)
            padded_prices.append(p)
            
    data_tensor = torch.FloatTensor(np.array(padded_data))
    price_tensor = torch.FloatTensor(np.array(padded_prices))
    
    return data_tensor, price_tensor, files, final_steps

import matplotlib.pyplot as plt

def test(model_path: str = "models/crypto_best", start_date="2024-01-01", end_date="2030-01-01"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Crypto Bot Backtest on {device}...")
    
    # 1. Load Data
    data, prices, files, steps = load_crypto_data(start_date, end_date)
    
    if data is None:
        print("❌ Data failed to load.")
        return

    print(f"📊 Test Data: {data.shape} ({len(files)} symbols)")
    
    # 2. Init Env
    env = VectorizedTradingEnv(data, prices, device=device)
    env.transaction_cost_bps = 10.0
    env.slippage_bps = 5.0
    
    # 3. Init Agent
    num_features = data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )
    
    # Load Weights
    print(f"🔄 Loading weights from {model_path}")
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Set Evaluation Mode
    for sub_agent in agent.agents:
        sub_agent.epsilon = 0.0
        sub_agent.epsilon_min = 0.0

    # 4. Evaluation Loop
    state = env.reset()
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    
    total_reward = 0
    pbar = tqdm(total=steps, desc="Backtesting")
    
    # Tracking
    portfolio_history = []
    action_counts = {0:0, 1:0, 2:0}
    
    for step in range(steps):
        # Batch Action
        actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
        
        # Ensure actions is a tensor (it should be from batch_act)
        # env.step expects tensor if env.device is set
        
        next_state, rewards, dones, info = env.step(actions)
        
        # Log Actions
        act_cpu = actions.cpu().numpy()
        for a in act_cpu:
            action_counts[a] += 1
            
        # Log Equity (from info)
        if 'equity' in info:
            total_equity = info['equity'].sum().item()
            portfolio_history.append(total_equity)
        
        state = next_state
        total_reward += rewards.sum().item()
        
        pbar.update(1)
        
    pbar.close()
    
    print(f"🏁 Backtest Complete.")
    print(f"💰 Total Reward (Sum log-return): {total_reward:.2f}")
    
    # Stats
    # VectorEnv stores self.initial_balance as a float, but self.balance is a tensor with that value.
    # We want total initial equity.
    initial_equity = env.initial_balance * len(files) 
    final_equity = portfolio_history[-1] if portfolio_history else initial_equity
    pnl = final_equity - initial_equity
    pnl_pct = (pnl / initial_equity) * 100
    
    print(f"💵 Initial Portfolio: ${initial_equity:,.2f}")
    print(f"💵 Final Portfolio:   ${final_equity:,.2f}")
    print(f"📈 PnL:               ${pnl:,.2f} ({pnl_pct:.2f}%)")
    print(f"🤖 Actions:           Hold: {action_counts[0]}, Buy: {action_counts[1]}, Sell: {action_counts[2]}")
    
    if portfolio_history:
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_history, label='Portfolio Value')
        plt.title('Crypto Bot Equity Curve (Valuation)')
        plt.xlabel('Steps (Days)')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid()
        plt.savefig('crypto_backtest_results.png')
        print("📊 Saved equity curve to crypto_backtest_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/crypto_best", help="Path prefix for model")
    args = parser.parse_args()
    test(args.model)
