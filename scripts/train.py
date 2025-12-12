#!/usr/bin/env python
"""
Training Entry Point
Usage: python scripts/train.py --fresh --episodes 50
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import ParallelTrainer
from config.settings import MoneyScraperConfig

def main():
    parser = argparse.ArgumentParser(description='Train Dueling DQN Agent')
    parser.add_argument('--fresh', action='store_true', help='Start training from scratch')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per symbol')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Load watchlist
    watchlist_path = Path("config/watchlists/my_portfolio.txt")
    if not watchlist_path.exists():
        watchlist_path = Path("my_portfolio.txt")
        
    with open(watchlist_path) as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    print(f"🚀 Starting training on {len(symbols)} symbols")
    print(f"   Episodes: {args.episodes}")
    print(f"   Fresh: {args.fresh}")
    
    trainer = ParallelTrainer(
        episodes_per_symbol=args.episodes,
        parallel_jobs=args.workers,
        fresh=args.fresh
    )
    
    trainer.train(symbols)

if __name__ == "__main__":
    main()
