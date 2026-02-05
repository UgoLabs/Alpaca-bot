
import os
import random
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import numpy as np

# Load env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

api = tradeapi.REST(
    os.getenv("SWING_API_KEY"),
    os.getenv("SWING_API_SECRET"),
    os.getenv("ALPACA_BASE_URL"),
    api_version='v2'
)

# 1. Load existing portfolio to exclude
portfolio_path = Path("config/watchlists/my_portfolio.txt")
exclude_symbols = set()
if portfolio_path.exists():
    with open(portfolio_path, 'r') as f:
        exclude_symbols = {line.strip() for line in f if line.strip()}

print(f"Avoiding {len(exclude_symbols)} symbols from current portfolio.")

# 2. Get all tradable assets from Alpaca
print("Fetching active assets from Alpaca...")
assets = api.list_assets(status='active', asset_class='us_equity')
tradable_assets = [a for a in assets if a.tradable and a.easy_to_borrow]

# 3. Filter candidates
candidates = []
for a in tradable_assets:
    if a.symbol not in exclude_symbols and "." not in a.symbol:
        candidates.append(a.symbol)

print(f"Found {len(candidates)} potential new symbols.")

# Take ALL valid candidates
selected_symbols = candidates
# limit for sanity if checking "every" - usually about 4000-6000
# If you really want ALL, comment out the slice.
# For reasonable execution time, let's process them all but in batches.

print(f"Selected {len(selected_symbols)} symbols for validation (ALL Tradeable).")

# 4. Download Data
save_dir = Path("data/historical_validation")
save_dir.mkdir(parents=True, exist_ok=True)

# Filter out already downloaded
existing_files = {f.stem.replace("_1D", "") for f in save_dir.glob("*_1D.csv")}
initial_count = len(selected_symbols)
selected_symbols = [s for s in selected_symbols if s not in existing_files]
print(f"Skipping {initial_count - len(selected_symbols)} already downloaded symbols. Downloading {len(selected_symbols)} new ones...")

print(f"Downloading data from YFinance to {save_dir}...")
success_count = 0

batch_size = 50
chunks = [selected_symbols[i:i + batch_size] for i in range(0, len(selected_symbols), batch_size)]

for chunk in tqdm(chunks, desc="Downloading Batches"):
    try:
        # bulk download is faster
        data = yf.download(chunk, start="2023-01-01", progress=False, group_by='ticker', auto_adjust=True)
        
        # Iterate through columns level 0 (Tickers) if multiple, or just handle single
        if len(chunk) == 1:
            # Handle single ticker case
            ticker = chunk[0]
            df = data
            if df.empty: continue
            
            # Format
            df = df.reset_index()
            # YF auto_adjust=True gives Open, High, Low, Close, Volume.
            # We need to ensure columns exist. 
            # Note: yfinance output structure changed recently for multi-index, but single is usually flat or simple.
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df.set_index('Date', inplace=True)
            
            df.to_csv(save_dir / f"{ticker}_1D.csv")
            success_count += 1
        else:
            for ticker in chunk:
                try:
                    df = data[ticker].copy()
                    if df.empty: continue
                    if df['Close'].isna().all(): continue # Skip empty

                    df = df.dropna(how='all')
                    
                    # Fix index
                    df = df.reset_index()
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                        df.set_index('Date', inplace=True)
                    
                    df.to_csv(save_dir / f"{ticker}_1D.csv")
                    success_count += 1
                except Exception:
                    continue
                    
    except Exception as e:
        print(f"Batch failed: {e}")

print(f"\nSuccessfully downloaded {success_count} new datasets.")
