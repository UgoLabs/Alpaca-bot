import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from backtest import Backtester
from datetime import datetime

# Configuration
MODEL_PATH = "models/SHARED_dqn_best.pth"
PORTFOLIO_FILE = "../my_portfolio.txt"
START_DATE = "2023-01-01"
END_DATE = "2024-12-08"
LOG_DIR = "logs/full_backtest"

def load_symbols():
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Portfolio file not found at {PORTFOLIO_FILE}")
        return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'AMZN', 'GOOGL', 'META']

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    symbols = load_symbols()
    
    # Remove duplicates
    symbols = list(set(symbols))
    print(f"Starting full backtest on {len(symbols)} symbols...")
    print(f"Model: {MODEL_PATH}")
    
    overall_results = []
    
    try:
        backtester = Backtester(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    for symbol in tqdm(symbols):
        try:
            results = backtester.run_backtest(symbol, START_DATE, END_DATE)
            if results:
                summary = {
                    'symbol': symbol,
                    'total_return': results['total_return'] * 100,
                    'sharpe': results['sharpe_ratio'],
                    'trades': results['total_trades'],
                    'win_rate': results['win_rate'] * 100,
                    'max_drawdown': results['max_drawdown'] * 100,
                    'buy_hold_return': results['buy_hold_return'] * 100
                }
                overall_results.append(summary)
        except Exception as e:
            # print(f"Error processing {symbol}: {e}") # Reduce noise
            pass

    # Analysis
    if not overall_results:
        print("No results generated.")
        return

    df = pd.DataFrame(overall_results)
    
    # Save raw results
    df.to_csv(f"{LOG_DIR}/summary.csv", index=False)
    
    print("\n" + "="*60)
    print("FULL PORTFOLIO BACKTEST RESULTS")
    print("="*60)
    print(f"Total Symbols Tested: {len(df)}")
    print(f"Average Return:       {df['total_return'].mean():.2f}%")
    print(f"Median Return:        {df['total_return'].median():.2f}%")
    print(f"Win Rate (Avg):       {df['win_rate'].mean():.2f}%")
    print(f"Sharpe Ratio (Avg):   {df['sharpe'].mean():.2f}")
    print(f"Trades per Symbol:    {df['trades'].mean():.1f}")
    print("-" * 60)
    print(f"Avg Buy & Hold Return:{df['buy_hold_return'].mean():.2f}%")
    print(f"Outperformance (Avg): {df['total_return'].mean() - df['buy_hold_return'].mean():.2f}%")
    print("="*60)
    
    # Best Performers
    print("\nTOP 5 PERFORMERS:")
    print(df.nlargest(5, 'total_return')[['symbol', 'total_return', 'trades', 'win_rate']].to_string(index=False))
    
    # Worst Performers
    print("\nBOTTOM 5 PERFORMERS:")
    print(df.nsmallest(5, 'total_return')[['symbol', 'total_return', 'trades', 'win_rate']].to_string(index=False))

if __name__ == "__main__":
    main()
