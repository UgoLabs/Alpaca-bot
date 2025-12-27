"""
Script to download massive historical data to local SSD.
Leverages the 3.4TB storage for instant training access later.
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SwingTraderCreds, ALPACA_BASE_URL

# Configuration
DATA_DIR = "data/historical"
# TIMEFRAME = "1Min" # Now handled by argparse
START_DATE = "2020-01-01"  # 5+ Years of data
SYMBOLS_FILE = "config/watchlists/my_portfolio.txt"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_data():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Download Historical Data')
    parser.add_argument('--timeframe', type=str, default='1Min', help='Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)')
    args = parser.parse_args()
    
    TIMEFRAME = args.timeframe
    
    print(f"🚀 Initializing Massive Data Download Strategy")
    print(f"📂 Target: {os.path.abspath(DATA_DIR)}")
    print(f"⏱️  Timeframe: {TIMEFRAME}")
    print(f"📅 Range: {START_DATE} to Now")
    
    ensure_dir(DATA_DIR)
    
    # Load Symbols
    try:
        with open(SYMBOLS_FILE, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Could not find {SYMBOLS_FILE}")
        return

    print(f"📋 Found {len(symbols)} symbols to archive.")
    
    # Initialize API
    api = tradeapi.REST(SwingTraderCreds.API_KEY, SwingTraderCreds.API_SECRET, ALPACA_BASE_URL)
    
    end_date = datetime.now()
    
    # Progress bar
    pbar = tqdm(symbols, desc="Downloading", unit="sym")
    
    for symbol in pbar:
        file_path = os.path.join(DATA_DIR, f"{symbol}_{TIMEFRAME}.csv")
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"🚀 Initializing Parallel Download (20 Workers)")
    
    def download_symbol(symbol, api_instance, data_dir, timeframe, start_date_str, end_date_str):
        try:
            file_path = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
            
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
                return f"Skipped {symbol}"
                
            # Download
            bars = api_instance.get_bars(
                symbol,
                timeframe,
                start=start_date_str,
                end=end_date_str,
                adjustment='raw',
                feed='sip'
            ).df
            
            if bars.empty:
                return f"No Data {symbol}"
            
            bars.to_csv(file_path)
            return f"Done {symbol}"
            
        except Exception as e:
            return f"Error {symbol}: {e}"

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_symbol, sym, api, DATA_DIR, TIMEFRAME, START_DATE, end_date_str): sym for sym in symbols}
        
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Downloading"):
            result = future.result()
            # tqdm.write(result) # Optional: Print details
            
    print("\n✅ Parallel Download Complete!")

if __name__ == "__main__":
    download_data()
