import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Load symbols
with open('../my_portfolio.txt', 'r') as f:
    symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]

print(f"Testing {len(symbols)} symbols from my_portfolio.txt\n")

valid = []
invalid = []

for i, symbol in enumerate(symbols):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(symbols)}")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='5d', interval='1d')
        
        if not df.empty and len(df) > 0:
            valid.append(symbol)
        else:
            invalid.append(symbol)
    except:
        invalid.append(symbol)

print(f"\n{'='*60}")
print(f"✅ VALID SYMBOLS: {len(valid)}/{len(symbols)}")
print(f"❌ INVALID SYMBOLS: {len(invalid)}")
print(f"{'='*60}")

if invalid:
    print(f"\nInvalid/Delisted symbols:")
    for sym in invalid:
        print(f"  - {sym}")

print(f"\nRecommendation: Remove invalid symbols from my_portfolio.txt")
