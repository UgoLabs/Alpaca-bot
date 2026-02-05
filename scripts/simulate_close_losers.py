"""Simulate closing losers"""
import alpaca_trade_api as tradeapi
from config.settings import SwingTraderCreds, ALPACA_BASE_URL

api = tradeapi.REST(str(SwingTraderCreds.API_KEY), str(SwingTraderCreds.API_SECRET), str(ALPACA_BASE_URL), api_version='v2')
positions = api.list_positions()
account = api.get_account()

current_equity = float(account.equity)
current_cash = float(account.cash)

print(f'📊 CURRENT STATE:')
print(f'   Equity: ${current_equity:,.2f}')
print(f'   Cash: ${current_cash:,.2f}')
print(f'   Positions: {len(positions)}')
print()

# Separate winners and losers
winners = []
losers = []
for p in positions:
    pnl = float(p.unrealized_pl)
    market_val = float(p.market_value)
    if pnl < 0:
        losers.append({'symbol': p.symbol, 'pnl': pnl, 'value': market_val, 'pnl_pct': float(p.unrealized_plpc)*100})
    else:
        winners.append({'symbol': p.symbol, 'pnl': pnl, 'value': market_val, 'pnl_pct': float(p.unrealized_plpc)*100})

# Sort losers by worst first
losers = sorted(losers, key=lambda x: x['pnl'])

print(f'🔴 LOSERS TO CLOSE ({len(losers)} positions):')
total_loss = 0
total_freed = 0
for l in losers:
    print(f"   {l['symbol']:<8} PnL: ${l['pnl']:>8.2f} ({l['pnl_pct']:>5.1f}%)  Value: ${l['value']:>10,.2f}")
    total_loss += l['pnl']
    total_freed += l['value']

print()
print(f'💸 IMPACT OF CLOSING ALL LOSERS:')
print(f'   Total Loss Realized: ${total_loss:,.2f}')
print(f'   Cash Freed Up: ${total_freed:,.2f}')
print(f'   New Cash Balance: ${current_cash + total_freed:,.2f}')
print(f'   Remaining Positions: {len(winners)}')
print()

# Winners summary
total_winner_pnl = sum(w['pnl'] for w in winners)
total_winner_value = sum(w['value'] for w in winners)
print(f'🟢 WINNERS KEPT ({len(winners)} positions):')
print(f'   Total Unrealized Gain: ${total_winner_pnl:,.2f}')
print(f'   Total Value: ${total_winner_value:,.2f}')
print()

new_equity = current_equity + total_loss  # equity drops by realized loss
print(f'📈 AFTER CLOSING LOSERS:')
print(f'   New Equity: ${new_equity:,.2f} (was ${current_equity:,.2f})')
print(f'   Net Change: ${total_loss:,.2f}')
print(f'   New Cash: ${current_cash + total_freed:,.2f} (was ${current_cash:,.2f})')
print(f'   Positions: {len(winners)} (was {len(positions)})')
print()
print(f'💡 VERDICT:')
if current_cash + total_freed > 0:
    print(f'   ✅ You would have POSITIVE cash: ${current_cash + total_freed:,.2f}')
    print(f'   ✅ No more margin! Ready to deploy fresh capital.')
else:
    print(f'   ⚠️ Still on margin: ${current_cash + total_freed:,.2f}')
