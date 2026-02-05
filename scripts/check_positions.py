"""Check current Alpaca positions"""
import alpaca_trade_api as tradeapi
from config.settings import SwingTraderCreds, ALPACA_BASE_URL

api = tradeapi.REST(str(SwingTraderCreds.API_KEY), str(SwingTraderCreds.API_SECRET), str(ALPACA_BASE_URL), api_version='v2')

positions = api.list_positions()
account = api.get_account()

print(f'Account Equity: ${float(account.equity):,.2f}')
print(f'Cash: ${float(account.cash):,.2f}')
print(f'Open Positions: {len(positions)}')
print()
print(f'{"Symbol":<8} {"Qty":>8} {"Entry":>10} {"Current":>10} {"PnL":>10} {"PnL%":>8}')
print('-' * 60)

total_pnl = 0
winners = 0
losers = 0
for p in sorted(positions, key=lambda x: float(x.unrealized_plpc), reverse=True):
    pnl = float(p.unrealized_pl)
    pnl_pct = float(p.unrealized_plpc) * 100
    total_pnl += pnl
    if pnl > 0:
        winners += 1
    else:
        losers += 1
    print(f'{p.symbol:<8} {float(p.qty):>8.2f} ${float(p.avg_entry_price):>9.2f} ${float(p.current_price):>9.2f} ${pnl:>9.2f} {pnl_pct:>7.1f}%')

print('-' * 60)
print(f'Total Unrealized PnL: ${total_pnl:,.2f}')
print(f'Winners: {winners} | Losers: {losers} | Win Rate: {winners/(winners+losers)*100:.1f}%')
