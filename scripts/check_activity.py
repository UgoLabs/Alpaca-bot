
import alpaca_trade_api as tradeapi
import pandas as pd
import sys
import os

# Add root to path so we can import config
sys.path.append(os.getcwd())

from config.settings import SwingTraderCreds, ALPACA_BASE_URL
from datetime import datetime, timedelta
import pytz

def check_activity():
    print("Connecting to Alpaca to check recent activity...")
    api = tradeapi.REST(
        SwingTraderCreds.API_KEY,
        SwingTraderCreds.API_SECRET,
        ALPACA_BASE_URL,
        api_version='v2'
    )

    # Check orders from the last 24 hours
    now = datetime.now(pytz.utc)
    start_time = now - timedelta(hours=24)
    start_str = start_time.isoformat()

    print(f"Fetching filled orders since {start_str}...")
    try:
        orders = api.list_orders(status='filled', after=start_str, direction='desc')
        
        if not orders:
            print("No trades executed in the last 24 hours.")
        else:
            print(f"Found {len(orders)} trades:")
            print("-" * 60)
            print(f"{'TIME':<20} | {'SYMBOL':<6} | {'SIDE':<4} | {'QTY':<5} | {'PRICE':<10} | {'VALUE':<10}")
            print("-" * 60)
            
            total_buy = 0
            total_sell = 0
            
            for o in orders:
                # Convert to ET for readability if possible, or just keep string
                filled_at = pd.to_datetime(o.filled_at).tz_convert('US/Eastern')
                time_str = filled_at.strftime('%Y-%m-%d %H:%M')
                
                price = float(o.filled_avg_price)
                qty = float(o.filled_qty)
                val = price * qty
                
                print(f"{time_str:<20} | {o.symbol:<6} | {o.side.upper():<4} | {qty:<5} | ${price:<9.2f} | ${val:<9.2f}")
                
                if o.side == 'buy':
                    total_buy += val
                else:
                    total_sell += val
            
            print("-" * 60)
            print(f"Total Bought: ${total_buy:.2f}")
            print(f"Total Sold:   ${total_sell:.2f}")

    except Exception as e:
        print(f"Error fetching orders: {e}")

    # Check open positions
    print("\nCurrent Holdings:")
    try:
        positions = api.list_positions()
        if not positions:
            print("No open positions.")
        else:
            print(f"{'SYMBOL':<6} | {'QTY':<5} | {'ENTRY':<10} | {'CURRENT':<10} | {'P/L ($)':<10} | {'P/L (%)':<10}")
            print("-" * 70)
            for p in positions:
                pl = float(p.unrealized_pl)
                pl_pct = float(p.unrealized_plpc) * 100
                print(f"{p.symbol:<6} | {p.qty:<5} | ${float(p.avg_entry_price):<9.2f} | ${float(p.current_price):<9.2f} | ${pl:<9.2f} | {pl_pct:.2f}%")
    except Exception as e:
        print(f"Error fetching positions: {e}")

if __name__ == "__main__":
    check_activity()
