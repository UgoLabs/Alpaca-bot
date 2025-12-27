import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.market_fetcher import MarketDataFetcher

def download_swing_data():
    fetcher = MarketDataFetcher()
    
    # 1. Load Watchlist
    watchlist_path = "config/watchlists/my_portfolio.txt"
    if os.path.exists(watchlist_path):
        with open(watchlist_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    else:
        print("⚠️ Watchlist not found. Using default list.")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "META", "NFLX", "SPY"]
        
    print(f"📋 Downloading 10 years of daily data for {len(symbols)} symbols...")
    
    output_dir = "data/historical_swing"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for symbol in tqdm(symbols):
        try:
            # Fetch 10 years of daily data
            df = fetcher.get_history_yfinance(symbol, period="10y", interval="1d")
            
            if not df.empty:
                # Save to CSV
                output_path = os.path.join(output_dir, f"{symbol}_1D.csv")
                df.to_csv(output_path)
                success_count += 1
        except Exception as e:
            print(f"❌ Failed to download {symbol}: {e}")
            
    print(f"✅ Download complete. Saved {success_count}/{len(symbols)} files to {output_dir}")

if __name__ == "__main__":
    download_swing_data()
