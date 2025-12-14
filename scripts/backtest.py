#!/usr/bin/env python
"""
Backtest the trained model on a specific symbol with custom parameters.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SwingTraderCreds, ALPACA_BASE_URL, SHARED_MODEL_PATH
from src.bots.swing_trader import SwingTraderBot
from src.core.indicators import add_technical_indicators
from src.environments.swing_env import SwingTradingEnv
from src.models.agent import DuelingDQN
from src.core.state import get_state_size

def run_backtest(symbol, days, timeframe, model_path):
    print(f"📉 Backtesting {symbol} for last {days} days ({timeframe})...")
    
    # 1. Fetch Data
    api = tradeapi.REST(SwingTraderCreds.API_KEY, SwingTraderCreds.API_SECRET, ALPACA_BASE_URL)
    end = datetime.now()
    start = end - timedelta(days=days)
    
    print("   Fetching data...")
    try:
        bars = api.get_bars(
            symbol, 
            timeframe, 
            start=start.strftime('%Y-%m-%d'), 
            end=end.strftime('%Y-%m-%d'), 
            limit=10000 if timeframe != '1Day' else 1000,
            feed='sip'
        ).df
        
        if bars.empty:
            print("   ❌ No data found.")
            return

        # Prepare DataFrame
        df = bars.reset_index()
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        
        # 2. Add Indicators
        df = add_technical_indicators(df)
        
        # 3. Load Agent
        print(f"   Loading model from {model_path}...")
        state_size = get_state_size()
        agent = DuelingDQN(state_size, 3)
        agent.load(model_path)
        
        # 4. Run Simulation
        env = SwingTradingEnv(df, initial_balance=10000)
        state = env.reset()
        done = False
        
        print("   Running simulation...")
        while not done:
            # Use greedy action (no epsilon)
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state

        # 5. Report
        metrics = env.get_metrics()
        
        if not metrics:
            print("\n" + "="*40)
            print(f"📊 BACKTEST RESULTS: {symbol}")
            print("="*40)
            print("❌ No trades executed.")
            # Still show Buy & Hold
            bnh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            print(f"🦁 Buy & Hold:      {bnh_return*100:+.2f}%")
            return

        print("\n" + "="*40)
        print(f"📊 BACKTEST RESULTS: {symbol}")
        print("="*40)
        print(f"💰 Initial Balance: ${env.initial_balance:,.2f}")
        print(f"💰 Final Balance:   ${env.net_worth:,.2f}")
        print(f"📈 Total Return:    {metrics['total_return']*100:+.2f}%")
        print(f"📉 Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
        print(f"⚖️ Sharpe Ratio:    {metrics['sharpe']:.2f}")
        print(f"🎲 Total Trades:    {metrics['total_trades']}")
        print(f"✅ Win Rate:        {metrics['win_rate']*100:.1f}%")
        print("="*40)
        
        # Buy & Hold comparison
        bnh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        print(f"🦁 Buy & Hold:      {bnh_return*100:+.2f}%")
        print("="*40)
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest a model on a single symbol')
    parser.add_argument('--symbol', type=str, default="SPY", help='Stock symbol')
    parser.add_argument('--days', type=int, default=365, help='Days of history')
    parser.add_argument('--timeframe', type=str, default="1Day", help='Timeframe (1Day, 5Min, etc)')
    parser.add_argument('--model', type=str, default=str(SHARED_MODEL_PATH), help='Path to model file')
    
    args = parser.parse_args()
    
    run_backtest(args.symbol, args.days, args.timeframe, args.model)
