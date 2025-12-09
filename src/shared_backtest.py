"""
Shared Model Portfolio Backtest
Tests the SHARED model across the entire portfolio to simulate portfolio performance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from backtest import Backtester
# from utils import load_portfolio  # Removed faulty import

import sys
import io

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def load_portfolio(path="my_portfolio.txt"):
    """Load symbols from a text file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            symbols = []
            seen = set()
            for line in f:
                sym = line.strip()
                if sym and not sym.startswith('#') and sym not in seen:
                    seen.add(sym)
                    symbols.append(sym)
            return symbols
    return ['SPY', 'QQQ', 'IWM'] # Default fallback

def run_single_backtest(symbol, start, end, model_path):
    """Run backtest for a single symbol (helper for parallel execution)."""
    try:
        backtester = Backtester(model_path)
        # Suppress prints for single runs
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        results = backtester.run_backtest(symbol, start, end)
        
        sys.stdout = original_stdout
        if results:
            results['symbol'] = symbol
            return results
    except Exception as e:
        # print(f"Error {symbol}: {e}")
        return None
    return None

def main():
    parser = argparse.ArgumentParser(description='Backtest Shared Model on Full Portfolio')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date')
    parser.add_argument('--model', type=str, default='models/SHARED_dqn_best.pth', help='Path to shared model')
    parser.add_argument('--parallel', type=int, default=10, help='Parallel threads')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 PORTFOLIO BACKTEST: {args.model}")
    print(f"📅 Period: {args.start} to {args.end}")
    print(f"{'='*60}")
    
    # Load portfolio
    symbols = load_portfolio()
    print(f"📋 Loaded {len(symbols)} symbols from portfolio")
    
    results_list = []
    
    # Run backtests sequentially (safer for debugging)
    # with ThreadPoolExecutor(max_workers=args.parallel) as executor:
    #     futures = {executor.submit(run_single_backtest, sym, args.start, args.end, args.model): sym for sym in symbols}
        
    #     for future in tqdm(futures, total=len(symbols), desc="Backtesting"):
    #         res = future.result()
    #         if res:
    #             results_list.append(res)
    
    print("Running sequential backtest...")
    for sym in tqdm(symbols, desc="Backtesting"):
        res = run_single_backtest(sym, args.start, args.end, args.model)
        if res:
            results_list.append(res)
    
    # Aggregation
    if not results_list:
        print("❌ No results generated.")
        return
        
    df_results = pd.DataFrame(results_list)
    
    print(f"\n{'='*60}")
    print(f"📊 AGGREGATE RESULTS ({len(results_list)} symbols)")
    print(f"{'='*60}")
    
    avg_return = df_results['total_return_pct'].mean()
    med_return = df_results['total_return_pct'].median()
    avg_sharpe = df_results['sharpe_ratio'].mean()
    avg_win_rate = df_results['win_rate_pct'].mean()
    total_trades = df_results['total_trades'].sum()
    
    print(f"💰 Avg Return:      {avg_return:+.2f}%")
    print(f"📉 Median Return:   {med_return:+.2f}%")
    print(f"⚖️  Avg Sharpe:      {avg_sharpe:.2f}")
    print(f"✅ Avg Win Rate:    {avg_win_rate:.2f}%")
    print(f"🔢 Total Trades:    {total_trades}")
    
    # Top Performers
    print(f"\n🏆 TOP 5 PERFORMERS")
    print("-" * 30)
    top_5 = df_results.nlargest(5, 'total_return_pct')
    for _, row in top_5.iterrows():
        print(f"{row['symbol']:<6} {row['total_return_pct']:>7.2f}% (Sharpe: {row['sharpe_ratio']:.2f})")
    
    # Bottom Performers
    print(f"\n💩 BOTTOM 5 PERFORMERS")
    print("-" * 30)
    bot_5 = df_results.nsmallest(5, 'total_return_pct')
    for _, row in bot_5.iterrows():
        print(f"{row['symbol']:<6} {row['total_return_pct']:>7.2f}% (Sharpe: {row['sharpe_ratio']:.2f})")
        
    # Save CSV
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/portfolio_backtest_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed results saved to {csv_path}")

if __name__ == "__main__":
    main()
