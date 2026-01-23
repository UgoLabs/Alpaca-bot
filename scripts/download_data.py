"""
Script to download massive historical data to local SSD using yfinance.
Ensures 10 years of Daily data for Swing Trading.
"""
import os
import sys
import argparse
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
DATA_DIR = "data/historical_swing"
TIMEFRAME = "1d" 
PERIOD = "10y"
DEFAULT_SYMBOLS_FILE = "config/watchlists/my_portfolio.txt"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_data(watchlist_path=None):
    symbols_file = watchlist_path if watchlist_path else DEFAULT_SYMBOLS_FILE
    
    print(f"🚀 Initializing Massive Data Download Strategy (yfinance)")
    print(f"📂 Target: {os.path.abspath(DATA_DIR)}")
    print(f"⏱️  Timeframe: {TIMEFRAME}")
    print(f"📅 Period: {PERIOD}")
    
    ensure_dir(DATA_DIR)
    
    # Load Symbols
    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"❌ Could not find {symbols_file}")
        return

    print(f"📋 Found {len(symbols)} symbols in {symbols_file} to archive.")
    
    def download_symbol(symbol):
        try:
            file_path = os.path.join(DATA_DIR, f"{symbol}_1D.csv")
            
            # Always download to ensure freshness and full history
            # yfinance is fast enough
            
            df = yf.download(symbol, period=PERIOD, interval=TIMEFRAME, progress=False, auto_adjust=False)
            
            if df.empty:
                return f"No Data {symbol}"
            
            # Flatten MultiIndex if present (yfinance update)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Ensure index is named Date
            df.index.name = "Date"
            
            df.to_csv(file_path)
            return f"Done {symbol}"
            
        except Exception as e:
            return f"Error {symbol}: {e}"

    print(f"🚀 Initializing Parallel Download (4 Workers)")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_symbol, sym): sym for sym in symbols}
        
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Downloading"):
            result = future.result()
            
    print("\n✅ Download Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchlist", type=str, help="Path to watchlist file", default=None)
    args = parser.parse_args()
    
    download_data(args.watchlist)
