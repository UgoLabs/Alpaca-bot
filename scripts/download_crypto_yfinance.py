"""Download crypto data using yfinance for training.

Usage:
  python scripts/download_crypto_yfinance.py --lookback-years 4
"""

import os
import sys
import argparse
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join("data", "historical")
WATCHLIST_PATH = os.path.join("config", "watchlists", "crypto_watchlist.txt")

def download_data(lookback_years=4):
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(WATCHLIST_PATH):
        print(f"❌ Watchlist not found: {WATCHLIST_PATH}")
        return

    with open(WATCHLIST_PATH, "r") as f:
        symbols = [line.strip().replace('/', '-') for line in f if line.strip()]

    print(f"📥 Downloading {lookback_years} years of data for {len(symbols)} symbols...")

    for sym in tqdm(symbols):
        try:
            # Add -USD if missing (e.g. BTC -> BTC-USD)
            yf_sym = sym
            if not yf_sym.endswith("-USD"):
                yf_sym = f"{yf_sym}-USD"
            
            # yfinance download
            df = yf.download(yf_sym, period=f"{lookback_years}y", interval="1d", progress=False)
            
            if df.empty:
                print(f"⚠️ No data for {sym}")
                continue

            # Flatten MultiIndex columns (yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure minimal columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                print(f"⚠️ Missing columns for {sym}")
                continue

            # Save as _1Min.csv format (even though it's 1D, the trainer expects this naming for now or we update trainer)
            # Actually, let's save as _1D.csv and update trainer to look for that too.
            # But to keep compatible with existing train_crypto_bot which looks for _1Min, we will use that suffix OR update trainer.
            # Best practice: update trainer to support 1D. For now, let's stick to 1Min naming convention 
            # OR better: save as _1D.csv and I'll update the trainer script to look for it.
            
            save_path = os.path.join(DATA_DIR, f"{sym.replace('-USD', 'USD')}_1D.csv")
            df.to_csv(save_path)
            
        except Exception as e:
            print(f"❌ Error {sym}: {e}")

if __name__ == "__main__":
    download_data()
