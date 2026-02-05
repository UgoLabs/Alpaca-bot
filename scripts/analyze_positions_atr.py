"""Analyze positions with 6x ATR guardrails"""
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from config.settings import SwingTraderCreds, ALPACA_BASE_URL

api = tradeapi.REST(str(SwingTraderCreds.API_KEY), str(SwingTraderCreds.API_SECRET), str(ALPACA_BASE_URL), api_version='v2')
positions = api.list_positions()

print(f'📊 ANALYZING 40 POSITIONS WITH 6x ATR GUARDRAILS\n')
print(f'{"Symbol":<8} {"PnL%":>8} {"ATR%":>8} {"Stop":>10} {"Target":>10} {"Status":<15}')
print('-' * 70)

stop_losses = []
take_profits = []
holding = []

for p in sorted(positions, key=lambda x: float(x.unrealized_plpc), reverse=True):
    symbol = p.symbol
    pnl_pct = float(p.unrealized_plpc) * 100
    entry = float(p.avg_entry_price)
    current = float(p.current_price)
    
    # Get historical data to calc ATR
    try:
        bars = api.get_bars(symbol, '1Day', limit=20).df
        if len(bars) < 14:
            atr_pct = 3.0  # default
        else:
            high = bars['high'].values
            low = bars['low'].values
            close = bars['close'].values
            tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
            atr = np.mean(tr[-14:])
            atr_pct = (atr / entry) * 100
    except:
        atr_pct = 3.0
    
    stop_pct = -6 * atr_pct
    target_pct = 6 * atr_pct
    
    if pnl_pct <= stop_pct:
        status = "❌ STOP LOSS"
        stop_losses.append(symbol)
    elif pnl_pct >= target_pct:
        status = "✅ TAKE PROFIT"
        take_profits.append(symbol)
    else:
        status = "⏳ Holding"
        holding.append(symbol)
    
    print(f'{symbol:<8} {pnl_pct:>7.1f}% {atr_pct:>7.1f}% {stop_pct:>9.1f}% {target_pct:>9.1f}% {status:<15}')

print('-' * 70)
print(f'\n📈 SUMMARY:')
print(f'   ❌ Would hit STOP LOSS: {len(stop_losses)} → {stop_losses}')
print(f'   ✅ Would hit TAKE PROFIT: {len(take_profits)} → {take_profits}')
print(f'   ⏳ Still holding: {len(holding)}')
print(f'\n💡 With MAX_POSITIONS=10, you would only have 10 of these trades.')
