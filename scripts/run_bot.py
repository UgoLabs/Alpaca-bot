#!/usr/bin/env python
"""
Unified Bot Runner
Usage: python scripts/run_bot.py [money_scraper|swing_trader|day_trader]
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bots.money_scraper import MoneyScraperBot, main as run_money_scraper
from src.bots.swing_trader import SwingTraderBot, main as run_swing_trader
from src.bots.day_trader import DayTraderBot, main as run_day_trader


BOTS = {
    'money_scraper': run_money_scraper,
    'scraper': run_money_scraper,
    'swing_trader': run_swing_trader,
    'swing': run_swing_trader,
    'day_trader': run_day_trader,
    'day': run_day_trader,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_bot.py <bot_name>")
        print(f"Available bots: {', '.join(BOTS.keys())}")
        sys.exit(1)
    
    bot_name = sys.argv[1].lower()
    
    if bot_name not in BOTS:
        print(f"Unknown bot: {bot_name}")
        print(f"Available bots: {', '.join(BOTS.keys())}")
        sys.exit(1)
    
    print(f"Starting {bot_name}...")
    BOTS[bot_name]()


if __name__ == "__main__":
    main()
