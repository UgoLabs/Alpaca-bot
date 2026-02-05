
import os
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

base_url = os.getenv("ALPACA_BASE_URL")
key_id = os.getenv("SWING_API_KEY")

print(f"Base URL: {base_url}")
print(f"Key Prefix: {key_id[:4] if key_id else 'None'}") # PK... = Paper, AK... = Live

api = tradeapi.REST(
    key_id,
    os.getenv("SWING_API_SECRET"),
    base_url,
    api_version='v2'
)

print("\nSearching for assets starting with 'IR'...")
try:
    # List active assets
    assets = api.list_assets(status='active', asset_class='us_equity')
    ir_assets = [a for a in assets if a.symbol.startswith('IR')]
    
    found_symbols = [a.symbol for a in ir_assets]
    print(f"Found {len(found_symbols)} assets starting with IR.")
    
    if "IRBT" in found_symbols:
        print("✅ IRBT is in the list!")
        asset = api.get_asset("IRBT")
        print(f"   Tradable: {asset.tradable}")
    else:
        print("❌ IRBT is NOT in the active asset list.")

    if "IRBO" in found_symbols:
        print("✅ IRBO is in the list!")
    else:
        print("❌ IRBO is NOT in the active asset list.")
        
    # Print a few examples
    print(f"Examples: {found_symbols[:10]}")

except Exception as e:
    print(f"Error: {e}")
