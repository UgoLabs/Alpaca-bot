import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import alpaca_trade_api as tradeapi

# Load .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

API_KEY = os.getenv("SWING_API_KEY")
API_SECRET = os.getenv("SWING_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    print("❌ Error: SWING_API_KEY or SWING_API_SECRET not found in .env")
    sys.exit(1)

def check_portfolio():
    print(f"Connecting to Alpaca ({BASE_URL})...")
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

    try:
        account = api.get_account()
        print("\n" + "="*40)
        print(f"ACCOUNT STATUS ({account.status})")
        print("="*40)
        print(f"Equity:       ${float(account.equity):,.2f}")
        print(f"Cash:         ${float(account.cash):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Day Trades:   {account.daytrade_count}")
        
        positions = api.list_positions()
        print("\n" + "="*40)
        print(f"POSITIONS ({len(positions)})")
        print("="*40)
        
        if not positions:
            print("No open positions.")
        else:
            print(f"{'SYMBOL':<8} {'QTY':<8} {'PRICE':<10} {'VALUE':<10} {'P/L ($)':<10} {'P/L (%)':<10}")
            print("-" * 60)
            for p in positions:
                pl_amt = float(p.unrealized_pl)
                pl_pct = float(p.unrealized_plpc) * 100
                print(f"{p.symbol:<8} {p.qty:<8} ${float(p.current_price):<9.2f} ${float(p.market_value):<9.2f} {pl_amt:<+9.2f} {pl_pct:<+8.2f}%")

    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    check_portfolio()
