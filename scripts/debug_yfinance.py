
import yfinance as yf
import pandas as pd

def test_download():
    symbol = "AAPL"
    print(f"Downloading {symbol}...")
    df = yf.download(symbol, period="1mo", interval="1d", progress=False)
    
    print("Columns:", df.columns)
    if isinstance(df.columns, pd.MultiIndex):
        print("MultiIndex Levels:", df.columns.levels)
        print("Level 0:", df.columns.get_level_values(0))
        print("Level 1:", df.columns.get_level_values(1))
        
    print("\nHead:\n", df.head())
    
    # Test the flattening logic
    if isinstance(df.columns, pd.MultiIndex):
        # Existing logic in download_data.py
        df.columns = df.columns.get_level_values(0)
        
    print("\nAfter Flattening:\n", df.head())
    print("Columns:", df.columns)

if __name__ == "__main__":
    test_download()
