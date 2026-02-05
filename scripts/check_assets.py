
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

api = tradeapi.REST(
    os.getenv("SWING_API_KEY"),
    os.getenv("SWING_API_SECRET"),
    os.getenv("ALPACA_BASE_URL"),
    api_version='v2'
)

symbols = ["IR", "IRBO", "IRBT"]

print(f"Checking assets on {os.getenv('ALPACA_BASE_URL')}...")

for sym in symbols:
    print(f"\n--- {sym} ---")
    try:
        asset = api.get_asset(sym)
        print(f"Status: {asset.status}")
        print(f"Tradable: {asset.tradable}")
        print(f"Shortable: {asset.shortable}")
        print(f"Exchange: {asset.exchange}")
        print(f"Class: {asset.class_}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
