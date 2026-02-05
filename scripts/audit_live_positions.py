import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SwingTraderCreds, ALPACA_BASE_URL

def audit_account():
    print(f"🕵️ Auditing Swing Trader Account...")
    
    api = tradeapi.REST(
        str(SwingTraderCreds.API_KEY),
        str(SwingTraderCreds.API_SECRET),
        str(ALPACA_BASE_URL),
        api_version='v2'
    )
    
    # 1. Account Info
    account = api.get_account()
    print(f"\n💰 Account Status ({account.status}):")
    print(f"   Equity: ${float(account.equity):,.2f}")
    print(f"   Cash:   ${float(account.cash):,.2f}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Day Trade Count: {account.daytrade_count}")
    
    # 2. Positions
    positions = api.list_positions()
    print(f"\n📦 Open Positions ({len(positions)}):")
    if positions:
        pos_data = []
        for p in positions:
            pos_data.append({
                'Symbol': p.symbol,
                'Qty': p.qty,
                'Entry': float(p.avg_entry_price),
                'Current': float(p.current_price),
                'PnL': float(p.unrealized_pl),
                'PnL%': float(p.unrealized_plpc) * 100
            })
        df_pos = pd.DataFrame(pos_data)
        print(df_pos.to_string(index=False))
    else:
        print("   (No open positions)")
        
    # 3. Recent Activities (Fills)
    # Get last 50 activities
    print(f"\n📜 Recent Trade History (last 7 days):")
    activities = api.get_activities(activity_types='FILL', direction='desc') # Limit default is 50
    
    # Filter for last 7 days? Or just show all 50
    # activities is a list of Activity objects
    
    if activities:
        trade_data = []
        for a in activities[:20]: # Show last 20
            trade_data.append({
                'Time': a.transaction_time,
                'Symbol': a.symbol,
                'Side': a.side,
                'Qty': a.qty,
                'Price': a.price,
                'Order ID': a.order_id
            })
        df_trades = pd.DataFrame(trade_data)
        print(df_trades.to_string(index=False))
    else:
        print("   (No recent trades found)")

if __name__ == "__main__":
    audit_account()
