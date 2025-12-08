import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

API_KEY = "PKA7TFQVG5OB3YK6UEJ6ZFEGOH"
API_SECRET = "6ceJ8ZhknodD8iGM2NuMYTpxjr4BMgc5DaoD1xCagtbp"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

symbol = 'NVDA'
print(f"\nTesting {symbol} manually...")

# Get data
bars = api.get_bars(symbol, '15Min', limit=100, feed='iex').df
print(f"Got {len(bars)} bars")

# Calculate signals (simplified)
close = bars['close']
volume = bars['volume']

# VWAP
current_date = bars.index[-1].date()
today_df = bars[bars.index.date == current_date].copy()
today_df['pv'] = today_df['close'] * today_df['volume']
vwap = today_df['pv'].cumsum().iloc[-1] / today_df['volume'].cumsum().iloc[-1]

# EMAs
ema_9 = close.ewm(span=9).mean().iloc[-1]
ema_21 = close.ewm(span=21).mean().iloc[-1]

# RSI
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss.replace(0, 1)
rsi = (100 - (100 / (1 + rs))).iloc[-1]

# Volume
vol_sma = volume.rolling(20).mean().iloc[-1]
vol_ratio = volume.iloc[-1] / vol_sma

# Current price
current_price = close.iloc[-1]
days_open = today_df['open'].iloc[0]

print(f"\nSignals:")
print(f"  Price: ${current_price:.2f}")
print(f"  VWAP: ${vwap:.2f} {'ABOVE' if current_price > vwap else 'BELOW'}")
print(f"  EMA9: {ema_9:.2f} vs EMA21: {ema_21:.2f} {'BULLISH' if ema_9 > ema_21 else 'BEARISH'}")
print(f"  RSI: {rsi:.1f} {'OK' if 40 < rsi < 70 else 'OUT OF RANGE'}")
print(f"  Volume Ratio: {vol_ratio:.2f}x {'HIGH' if vol_ratio > 1.3 else 'LOW'}")
print(f"  Day Open: ${days_open:.2f} {'GREEN' if current_price > days_open else 'RED'}")

score = 0
if current_price > vwap: score += 2
if ema_9 > ema_21: score += 2
if 40 < rsi < 70: score += 1
if vol_ratio > 1.3: score += 1
if current_price > days_open: score += 1

print(f"\nSCORE: {score}/7 (need 4 to buy)")
print(f"{'WOULD BUY!' if score >= 4 else 'HOLD'}")
