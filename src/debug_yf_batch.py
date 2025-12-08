import yfinance as yf
import pandas as pd
import logging

# Suppress noise
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Test with a mix of symbols, including potential edge cases
symbols = ['NVDA', 'AMD', 'AAPL', 'MSFT', 'INVALID_SYM_TEST']

print(f"Testing yf.download with symbols: {symbols}")

try:
    # Mimic bot_standalone parameters
    data = yf.download(
        tickers=symbols,
        period='1y',
        interval='1d',
        group_by='ticker',
        auto_adjust=True,
        progress=False,
        threads=False
    )

    print(f"\nData Shape: {data.shape}")
    print(f"Data Empty? {data.empty}")
    print(f"Column Levels: {data.columns.nlevels}")
    print(f"Column Names: {data.columns}")
    
    if data.columns.nlevels > 1:
        print("\nMultiIndex detected. Level 0 values:")
        print(data.columns.get_level_values(0).unique())
        
    print("\nAttempting access:")
    for sym in symbols:
        try:
            if sym in data.columns.get_level_values(0):
                df = data[sym].copy()
                print(f"  ✅ Found {sym}: {len(df)} rows")
            else:
                print(f"  ❌ {sym} not in level 0 columns")
        except Exception as e:
            print(f"  ⚠️ Error accessing {sym}: {e}")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
