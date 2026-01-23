
import os
import yfinance as yf
import pandas as pd
from tqdm import tqdm

def download_fix():
    # Load from Portfolio
    SYMBOLS_FILE = "config/watchlists/my_portfolio.txt"
    portfolio_symbols = []
    
    if os.path.exists(SYMBOLS_FILE):
        with open(SYMBOLS_FILE, 'r') as f:
            portfolio_symbols = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if not portfolio_symbols:
        print("❌ Could not load symbols from portfolio file!")
        # Fallback (Original Priority List kept just in case, but usually we want the file)
        portfolio_symbols = [
            'AAPL', 'MSFT', 'NVDA', 'AMD', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA'
        ]

    # Ensure dir exists
    DATA_DIR = "data/historical_swing"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"🚀 Downloading {len(portfolio_symbols)} Portfolio Symbols properly...")
    
    for symbol in tqdm(portfolio_symbols):
        try:
            df = yf.download(symbol, period="10y", interval="1d", progress=False, auto_adjust=False)
            
            # Flatten MultiIndex logic 
            if isinstance(df.columns, pd.MultiIndex):
                # Try to find 'Price' level or just take level 0
                df.columns = df.columns.get_level_values(0)
            
            df.index.name = "Date"
            
            # Sanity Check: Check columns are unique
            # If we get duplicate columns (Close, Close), keep the first
            df = df.loc[:, ~df.columns.duplicated()]
            
            if df.empty:
                continue

            output_path = os.path.join(DATA_DIR, f"{symbol}_1D.csv")
            df.to_csv(output_path)
            
        except Exception as e:
            print(f"FAILED {symbol}: {e}")

    print("✅ Portfolio Download Complete.")

if __name__ == "__main__":
    download_fix()
