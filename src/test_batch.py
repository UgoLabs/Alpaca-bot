import alpaca_trade_api as tradeapi
import pandas as pd

API_KEY = "PKA7TFQVG5OB3YK6UEJ6ZFEGOH"
API_SECRET = "6ceJ8ZhknodD8iGM2NuMYTpxjr4BMgc5DaoD1xCagtbp"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)
symbols = ['NVDA', 'TSLA', 'SPY']

print("Testing batch fetch...")
try:
    bars = api.get_bars(symbols, '15Min', limit=10, feed='iex').df
    print(f"Received dataframe with shape: {bars.shape}")
    print(bars.head())
    
    print("\nIterating:")
    for symbol, group in bars.groupby(level=0):
        print(f" - {symbol}: {len(group)} bars")
        
except Exception as e:
    print(f"Error: {e}")
