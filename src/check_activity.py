import alpaca_trade_api as api
from datetime import datetime

API_KEY = "PKA7TFQVG5OB3YK6UEJ6ZFEGOH"
API_SECRET = "6ceJ8ZhknodD8iGM2NuMYTpxjr4BMgc5DaoD1xCagtbp"
BASE_URL = "https://paper-api.alpaca.markets"

a = api.REST(API_KEY, API_SECRET, BASE_URL)

print("="*60)
print("PAPER ACCOUNT 2 - DAY TRADER")
print("="*60)

# Account
acc = a.get_account()
print(f"\n💰 ACCOUNT STATUS")
print(f"   Equity: ${float(acc.equity):,.2f}")
print(f"   Cash: ${float(acc.cash):,.2f}")
print(f"   Buying Power: ${float(acc.buying_power):,.2f}")

# Positions
print(f"\n📊 POSITIONS")
positions = a.list_positions()
if positions:
    for p in positions:
        pnl = float(p.unrealized_plpc) * 100
        color = "🟢" if pnl > 0 else "🔴"
        print(f"   {color} {p.symbol}: {p.qty} @ ${float(p.avg_entry_price):.2f} | P/L: {pnl:+.2f}%")
else:
    print("   None")

# Orders
print(f"\n📋 RECENT ORDERS (Last 10)")
orders = a.list_orders(status='all', limit=10)
if orders:
    for o in orders:
        time_str = str(o.created_at)[:19]
        print(f"   {time_str} | {o.side.upper():4s} {o.qty:>4s} {o.symbol:6s} | {o.status}")
else:
    print("   None")

print("\n" + "="*60)
