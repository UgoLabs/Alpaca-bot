#!/usr/bin/env python3
"""
Run Hybrid DQN-LSTM Trading System
This script runs the hybrid trading system with short-term trading parameters.
"""

import os
import argparse
from datetime import datetime, timedelta
from hybrid_trading_system import HybridTradingSystem

def main():
    parser = argparse.ArgumentParser(description="Run Hybrid DQN-LSTM Trading System with short-term trading parameters")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest", 
                       help="Trading mode: backtest or live")
    parser.add_argument("--balance", type=float, default=5222.58, 
                       help="Initial balance")
    parser.add_argument("--symbols", type=str, nargs="+", 
                       default=["PLTR", "NET", "NVDA", "TSLA", "INTC", "MSTR", "SMCI", "META", "TSM", "NFLX"],
                       help="Stock symbols to trade")
    
    # Optional backtest parameters
    parser.add_argument("--days", type=int, default=30, 
                       help="Number of days to backtest (from today backwards)")
    parser.add_argument("--rebalance", type=int, default=1,
                       help="Rebalance interval in days (default: daily)")
    
    args = parser.parse_args()
    
    # Create the hybrid trading system with short-term trading parameters
    system = HybridTradingSystem(
        initial_balance=args.balance,
        max_positions=5,           # Hold max 5 positions
        position_size=0.2,         # 20% position size
        stop_loss=0.05,            # 5% stop loss
        trailing_stop=0.03,        # 3% trailing stop
        symbols=args.symbols
    )
    
    if args.mode == "backtest":
        # Calculate date range for backtest - more recent shorter timeframe
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        
        print(f"\n{'='*80}")
        print(f"HYBRID DQN-LSTM SHORT-TERM TRADING SYSTEM - BACKTEST")
        print(f"{'='*80}")
        print(f"Period: {start_date} to {end_date} ({args.days} days)")
        print(f"Initial balance: ${args.balance:.2f}")
        print(f"Symbols: {', '.join(args.symbols)}")
        print(f"Strategy: DQN primary with LSTM signal enhancement")
        print(f"Parameters: 20-day lookback, {args.rebalance}-day rebalance, 7-day max hold")
        print(f"{'='*80}\n")
        
        # Run backtest with the specified rebalance interval (default: daily)
        system.backtest(
            start_date=start_date,
            end_date=end_date,
            rebalance_interval=args.rebalance
        )
    
    elif args.mode == "live":
        print("Live trading mode not implemented yet.")
        # Future implementation would initialize live trading here
        # This would connect to Alpaca, fetch real-time data, and execute trades

if __name__ == "__main__":
    main() 