"""
Backtest with PURE AGENT BRAIN - No stops, no profit targets, unlimited positions
"""
import os, sys, glob, re, argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('model', help='Model path')
parser.add_argument('--start', default='2024-01-01', help='Start date')
parser.add_argument('--thresh', type=float, default=0.30, help='Confidence threshold')
parser.add_argument('--max-pos', type=int, default=0, help='Max positions (0=unlimited)')
args = parser.parse_args()

# Load data
data_dir = 'data/historical_swing'
start_date = args.start
files = glob.glob(os.path.join(data_dir, '*_1D.csv'))

market_data = {}
all_dates = set()

print('Loading symbols...')
for f in tqdm(files, desc='Loading'):
    try:
        df = pd.read_csv(f)
        symbol = os.path.basename(f).replace('_1D.csv', '')
        df.columns = [c.lower() for c in df.columns]
        col_map = {'o': 'Open', 'open': 'Open', 'h': 'High', 'high': 'High', 'l': 'Low', 'low': 'Low', 'c': 'Close', 'close': 'Close', 'v': 'Volume', 'volume': 'Volume'}
        for k, v in col_map.items():
            if k in df.columns: df[v] = df[k]
        if 'date' in df.columns: df['Date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'])
        else: continue
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df = df[df.index >= pd.to_datetime(start_date)]
        if df.empty: continue
        df = add_technical_indicators(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Select features based on model type
        if 'sharpe' in args.model.lower():
            feature_cols = [
                'sma_10', 'sma_20', 'sma_50', 'sma_200', 'atr', 'bb_width', 'bb_upper', 'bb_lower',
                'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_diff', 'adx', 'rsi',
                'stoch_k', 'stoch_d', 'williams_r', 'roc', 'bb_pband', 'volume_sma',
                'volume_ratio', 'obv', 'mfi', 'price_vs_sma20'
            ]
        else:
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12']
        
        if any(c not in df.columns for c in feature_cols): continue
        features = df[feature_cols].values
        mean, std = np.mean(features, axis=0), np.std(features, axis=0) + 1e-8
        norm_features = (features - mean) / std
        norm_features = np.clip(norm_features, -10, 10)  # Clip outliers
        norm_features = np.nan_to_num(norm_features, nan=0.0)
        market_data[symbol] = {'df': df, 'norm_features': norm_features, 'dates': df.index}
        all_dates.update(df.index)
    except: continue

valid_dates = sorted(list(all_dates))
print(f'Loaded {len(market_data)} symbols')

# Load Agent - detect feature count from model name
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'sharpe' in args.model.lower():
    feature_dim = 25
else:
    feature_dim = 11
agent = EnsembleAgent(time_series_dim=feature_dim, vision_channels=feature_dim, action_dim=3, device=device)
model_prefix = re.sub(r'_(aggressive|balanced|conservative)\.pth$', '', args.model).replace('.pth', '')
agent.load(model_prefix)
agent.set_eval()

# Sim params - AGENT BRAIN ONLY
cash = 10000.0
positions = {}
equity_curve = []
trade_log = []
window_size = 60
confidence_threshold = args.thresh

max_pos_str = f"Max {args.max_pos}" if args.max_pos > 0 else "Unlimited"
print(f'Simulating from {valid_dates[0].date()} to {valid_dates[-1].date()}...')
print(f'🧠 Settings: NO STOP LOSS, NO TAKE PROFIT, {max_pos_str} POSITIONS - Agent Brain Only')

for current_date in tqdm(valid_dates):
    current_equity = cash
    active_symbols, feed_tensors = [], []
    
    for symbol, data in market_data.items():
        df = data['df']
        if current_date not in df.index: continue
        try:
            curr_idx = df.index.get_loc(current_date)
            if curr_idx < window_size: continue
            input_window = data['norm_features'][curr_idx-window_size+1:curr_idx+1]
            if len(input_window) != window_size: continue
            active_symbols.append(symbol)
            feed_tensors.append(input_window)
        except: continue
    
    if not active_symbols:
        equity_curve.append({'Date': current_date, 'Equity': cash + sum(p['qty']*p['entry_price'] for p in positions.values())})
        continue
    
    # Batch inference
    batch = torch.tensor(np.array(feed_tensors), dtype=torch.float32, device=device)
    dummy_text = torch.zeros((len(active_symbols), 64), dtype=torch.long, device=device)
    with torch.no_grad():
        actions, confidences = agent.act(batch, dummy_text, dummy_text, return_q=True)
    
    signals = {}
    for i, sym in enumerate(active_symbols):
        df = market_data[sym]['df']
        row = df.loc[current_date]
        signals[sym] = {'action': actions[i], 'conf': confidences[i].item(), 'price': row['Close'], 'low': row['Low'], 'atr': row.get('atr', 1.0)}
    
    # SELLS - Agent brain only (no stops, no take profit)
    symbols_to_sell = []
    for sym, pos in positions.items():
        if sym not in signals:
            current_equity += pos['qty'] * pos['entry_price']
            continue
        sig = signals[sym]
        # ONLY AGENT SELL - confidence threshold
        if sig['action'] == 2 and sig['conf'] > confidence_threshold:
            symbols_to_sell.append((sym, sig['price'], 'AGENT_SELL'))
    
    for sym, price, reason in symbols_to_sell:
        pos = positions.pop(sym)
        proceeds = pos['qty'] * price
        cash += proceeds
        pnl = (price - pos['entry_price']) * pos['qty']
        pnl_pct = (price - pos['entry_price']) / pos['entry_price']
        trade_log.append({'Date': current_date, 'Symbol': sym, 'Side': 'SELL', 'Price': price, 'Reason': reason, 'PnL': pnl, 'PnL%': pnl_pct})
    
    # BUYS - Agent brain only
    opportunities = [(sym, sig) for sym, sig in signals.items() if sym not in positions and sig['action'] == 1 and sig['conf'] > confidence_threshold]
    opportunities.sort(key=lambda x: x[1]['conf'], reverse=True)
    
    # Check max positions limit
    max_positions = args.max_pos if args.max_pos > 0 else float('inf')
    
    holdings_value = sum(p['qty'] * signals.get(sym, {'price': p['entry_price']})['price'] for sym, p in positions.items())
    current_equity = cash + holdings_value
    
    # Equal weight sizing - use 2% of equity per position
    position_size = current_equity * 0.02
    
    for sym, sig in opportunities:
        if len(positions) >= max_positions: break  # Max positions check
        if cash < position_size * 0.5: break
        price = sig['price']
        qty = int(position_size / price)
        if qty <= 0: continue
        cost = qty * price
        if cost > cash: continue
        cash -= cost
        conf = sig['conf']
        positions[sym] = {'qty': qty, 'entry_price': price, 'symbol': sym}
        trade_log.append({'Date': current_date, 'Symbol': sym, 'Side': 'BUY', 'Price': price, 'Reason': f'AGENT_BUY Conf:{conf:.3f}'})
    
    # Record equity
    holdings_value = sum(p['qty'] * signals.get(sym, {'price': p['entry_price']})['price'] for sym, p in positions.items())
    equity_curve.append({'Date': current_date, 'Equity': cash + holdings_value})

# Final
final_equity = equity_curve[-1]['Equity']
trades_df = pd.DataFrame(trade_log)
sells = trades_df[trades_df['Side'] == 'SELL']
wins = (sells['PnL'] > 0).sum() if len(sells) > 0 else 0
total_sells = len(sells)

print(f'\n🏁 AGENT BRAIN ONLY - No Stops, No Profit Targets, Unlimited Positions')
print(f'💰 Final Equity: ${final_equity:,.2f} ({(final_equity/10000-1)*100:+.1f}%)')
print(f'📊 Total Trades: {len(trades_df)}')
if total_sells > 0:
    print(f'✅ Win Rate: {100*wins/total_sells:.1f}%')
else:
    print('No sells')
print(f'📦 Final Positions: {len(positions)}')
if len(sells) > 0:
    if wins > 0:
        print(f'💚 Avg Win: ${sells[sells["PnL"]>0]["PnL"].mean():.2f}')
    if (total_sells-wins) > 0:
        print(f'💔 Avg Loss: ${sells[sells["PnL"]<0]["PnL"].mean():.2f}')

# Save
trades_df.to_csv('logs/trades_agent_brain_only.csv', index=False)
pd.DataFrame(equity_curve).to_csv('logs/equity_agent_brain_only.csv', index=False)
print('\n📁 Saved to logs/trades_agent_brain_only.csv')
