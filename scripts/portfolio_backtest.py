"""
Parallel Backtest of Full Portfolio
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import threading

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_trade_api as tradeapi
from config.settings import SwingTraderCreds, ALPACA_BASE_URL, SHARED_MODEL_PATH
from src.core.indicators import add_technical_indicators
from src.environments.swing_env import SwingTradingEnv
from src.models.agent import DuelingDQN
from src.core.state import get_state_size

# Global objects
api_lock = threading.Lock()
api = tradeapi.REST(SwingTraderCreds.API_KEY, SwingTraderCreds.API_SECRET, ALPACA_BASE_URL)
agent = None

def get_data_safe(symbol, days=365):
    """Thread-safe data fetching with retry/rate-limit handling"""
    end = datetime.now()
    start = end - timedelta(days=days)
    
    retries = 3
    for i in range(retries):
        try:
            # We don't lock API calls for REST generally, but if rate limit hits, we pause.
            bars = api.get_bars(symbol, '1Day', start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), limit=1000).df
            if bars.empty:
                return None
            return bars
        except Exception as e:
            if "429" in str(e):
                time.sleep(1 + (i*2)) # Backoff
            else:
                return None
    return None

def process_symbol(symbol):
    """Run backtest for a single symbol"""
    try:
        # 1. Fetch Data
        bars = get_data_safe(symbol)
        if bars is None or len(bars) < 50:
            return None

        # Prepare DataFrame
        df = bars.reset_index()
        # Rename lower case to Title Case if needed
        cols_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open','high','low','close','volume']}
        df = df.rename(columns=cols_map)
        
        # Add indicators
        df = add_technical_indicators(df)
        
        # 2. Run Simulation
        env = SwingTradingEnv(df, initial_balance=10000)
        state = env.reset()
        done = False
        
        while not done:
            # Use greedy action (epsilon=0.0) from SHARED AGENT
            # Note: PyTorch inference is thread-safe if no_grad is used (which act() does)
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state

        # 3. Collect Metrics
        metrics = env.get_metrics()
        if not metrics or metrics.get('total_trades', 0) == 0:
            return None # No trades
            
        metrics['symbol'] = symbol
        return metrics

    except Exception as e:
        # print(f"Error on {symbol}: {e}")
        return None

def run_portfolio_backtest():
    global agent
    
    # Load Portfolio
    params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_portfolio.txt")
    with open(params_path) as f:
        symbols = [line.strip().upper() for line in f if line.strip()]
    
    print(f"🚀 Starting Parallel Backtest on {len(symbols)} symbols...")
    
    # Load Agent ONCE
    print(f"🧠 Loading Model: {SHARED_MODEL_PATH}")
    state_size = get_state_size()
    agent = DuelingDQN(state_size, 3)
    agent.load(str(SHARED_MODEL_PATH))
    agent.model.eval() # Set to eval mode for deterministic behavior
    
    results = []
    start_time = time.time()
    
    # Run with ThreadPool
    # 8 workers is a good balance for REST API limits and CPU inference
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {executor.submit(process_symbol, sym): sym for sym in symbols}
        
        completed = 0
        total = len(symbols)
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            completed += 1
            if completed % 10 == 0:
                print(f"   Progress: {completed}/{total} ({(completed/total)*100:.1f}%)")
                
            res = future.result()
            if res:
                results.append(res)
    
    elapsed = time.time() - start_time
    
    # Report
    print("\n" + "="*60)
    print(f"📊 PORTFOLIO BACKTEST REPORT ({elapsed:.1f}s)")
    print("="*60)
    
    if not results:
        print("❌ No trades executed across the entire portfolio.")
        print("   (The model is currently extremely conservative/selective)")
        return

    # Convert to DataFrame for analysis
    res_df = pd.DataFrame(results)
    res_df.to_csv("backtest_summary.csv", index=False)
    print(f"💾 Saved results to backtest_summary.csv")
    
    res_df = res_df.sort_values(by='total_return', ascending=False)
    
    print(f"✅ Active Symbols: {len(res_df)}")
    print(f"💰 Average Return: {res_df['total_return'].mean()*100:+.2f}%")
    print(f"🎲 Total Trades:   {res_df['total_trades'].sum()}")
    print("-" * 60)
    print("🏆 TOP 10 PERFORMERS:")
    print(f"{'Symbol':<8} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Sharpe'}")
    
    for _, row in res_df.head(10).iterrows():
        print(f"{row['symbol']:<8} {row['total_return']*100:+.2f}%    {row['total_trades']:<8} {row['win_rate']*100:.0f}%       {row['sharpe']:.2f}")
    
    print("-" * 60)
    print("📉 BOTTOM 5 PERFORMERS:")
    for _, row in res_df.tail(5).iterrows():
        print(f"{row['symbol']:<8} {row['total_return']*100:+.2f}%    {row['total_trades']:<8} {row['win_rate']*100:.0f}%       {row['sharpe']:.2f}")
    print("="*60)

if __name__ == "__main__":
    run_portfolio_backtest()
