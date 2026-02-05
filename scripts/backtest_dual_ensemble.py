"""
Backtest combining TWO models as an ensemble:
- sharpe_gen3_ep2 (25 features)
- swing_gen7_refined_ep380 (11 features)
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
parser.add_argument('--start', default='2024-01-01', help='Start date')
parser.add_argument('--thresh', type=float, default=0.30, help='Confidence threshold')
parser.add_argument('--guards', action='store_true', help='Use guardrails (stops/profit)')
args = parser.parse_args()

# Feature definitions
FEATURES_25 = [
    'sma_10', 'sma_20', 'sma_50', 'sma_200', 'atr', 'bb_width', 'bb_upper', 'bb_lower',
    'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_diff', 'adx', 'rsi',
    'stoch_k', 'stoch_d', 'williams_r', 'roc', 'bb_pband', 'volume_sma',
    'volume_ratio', 'obv', 'mfi', 'price_vs_sma20'
]
FEATURES_11 = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12']

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
        
        # Check both feature sets exist
        if any(c not in df.columns for c in FEATURES_25): continue
        if any(c not in df.columns for c in FEATURES_11): continue
        
        # Normalize 25 features
        feat25 = df[FEATURES_25].values
        mean25, std25 = np.mean(feat25, axis=0), np.std(feat25, axis=0) + 1e-8
        norm25 = np.clip((feat25 - mean25) / std25, -10, 10)
        norm25 = np.nan_to_num(norm25, nan=0.0)
        
        # Normalize 11 features
        feat11 = df[FEATURES_11].values
        mean11, std11 = np.mean(feat11, axis=0), np.std(feat11, axis=0) + 1e-8
        norm11 = (feat11 - mean11) / std11
        
        market_data[symbol] = {
            'df': df, 
            'norm_25': norm25, 
            'norm_11': norm11,
            'dates': df.index
        }
        all_dates.update(df.index)
    except: continue

valid_dates = sorted(list(all_dates))
print(f'Loaded {len(market_data)} symbols')

# Load BOTH agents
device = 'cuda' if torch.cuda.is_available() else 'cpu'

agent_sharpe = EnsembleAgent(time_series_dim=25, vision_channels=25, action_dim=3, device=device)
agent_sharpe.load('models/sharpe_gen3_ep20')
agent_sharpe.set_eval()
print('✅ Loaded sharpe_gen3_ep20')

agent_swing = EnsembleAgent(time_series_dim=11, vision_channels=11, action_dim=3, device=device)
agent_swing.load('models/swing_gen7_refined_ep380')
agent_swing.set_eval()
print('✅ Loaded swing_gen7_refined_ep380')

# Sim params
cash = 10000.0
positions = {}
equity_curve = []
trade_log = []
window_size = 60
confidence_threshold = args.thresh
max_positions = 9999 if not args.guards else 10

print(f'Simulating from {valid_dates[0].date()} to {valid_dates[-1].date()}...')
if args.guards:
    print('🛡️ Settings: WITH GUARDRAILS (Stop Loss, Take Profit, Max 10 Positions)')
else:
    print('🧠 Settings: DUAL ENSEMBLE - No Stops, No Profit Targets')

for current_date in tqdm(valid_dates):
    current_equity = cash
    active_symbols = []
    feed_25 = []
    feed_11 = []
    
    for symbol, data in market_data.items():
        df = data['df']
        if current_date not in df.index: continue
        try:
            idx = df.index.get_loc(current_date)
            if idx < window_size: continue
            
            window_25 = data['norm_25'][idx-window_size+1:idx+1]
            window_11 = data['norm_11'][idx-window_size+1:idx+1]
            
            if len(window_25) != window_size: continue
            
            active_symbols.append(symbol)
            feed_25.append(window_25)
            feed_11.append(window_11)
        except: continue
    
    # Check stops/profits for existing positions
    for sym in list(positions.keys()):
        if sym not in market_data: continue
        df = market_data[sym]['df']
        if current_date not in df.index: continue
        row = df.loc[current_date]
        pos = positions[sym]
        current_equity += pos['qty'] * row['Close']
        
        if args.guards:
            pnl_pct = (row['Close'] - pos['entry']) / pos['entry']
            # Stop loss at -8%
            if pnl_pct < -0.08:
                cash += pos['qty'] * row['Close']
                trade_log.append({'date': current_date, 'symbol': sym, 'action': 'SELL', 'reason': 'STOP_LOSS', 'pnl': pnl_pct})
                del positions[sym]
                continue
            # Take profit at +15%
            if pnl_pct > 0.15:
                cash += pos['qty'] * row['Close']
                trade_log.append({'date': current_date, 'symbol': sym, 'action': 'SELL', 'reason': 'TAKE_PROFIT', 'pnl': pnl_pct})
                del positions[sym]
                continue
    
    if not active_symbols:
        equity_curve.append({'Date': current_date, 'Equity': current_equity})
        continue
    
    # Run BOTH models
    batch_25 = torch.FloatTensor(np.array(feed_25)).to(device)
    batch_11 = torch.FloatTensor(np.array(feed_11)).to(device)
    dummy_text_25 = torch.zeros((len(active_symbols), 64), dtype=torch.long).to(device)
    dummy_text_11 = torch.zeros((len(active_symbols), 64), dtype=torch.long).to(device)
    
    with torch.no_grad():
        actions_sharpe, conf_sharpe = agent_sharpe.act(batch_25, dummy_text_25, dummy_text_25, return_q=True)
        actions_swing, conf_swing = agent_swing.act(batch_11, dummy_text_11, dummy_text_11, return_q=True)
    
    # Combine signals - require BOTH models to agree or take weighted vote
    signals = {}
    for i, sym in enumerate(active_symbols):
        a_sharpe = actions_sharpe[i].item() if hasattr(actions_sharpe[i], 'item') else actions_sharpe[i]
        a_swing = actions_swing[i].item() if hasattr(actions_swing[i], 'item') else actions_swing[i]
        c_sharpe = conf_sharpe[i].item() if hasattr(conf_sharpe[i], 'item') else conf_sharpe[i]
        c_swing = conf_swing[i].item() if hasattr(conf_swing[i], 'item') else conf_swing[i]
        
        # Agreement: both say same thing
        if a_sharpe == a_swing:
            final_action = a_sharpe
            final_conf = (c_sharpe + c_swing) / 2
        else:
            # Disagreement: weighted by confidence
            # Give slight edge to sharpe model (newer, better returns)
            if c_sharpe * 1.2 > c_swing:
                final_action = a_sharpe
                final_conf = c_sharpe * 0.7
            else:
                final_action = a_swing
                final_conf = c_swing * 0.7
        
        df = market_data[sym]['df']
        row = df.loc[current_date]
        signals[sym] = {
            'action': final_action,
            'conf': final_conf,
            'price': row['Close'],
            'sharpe': a_sharpe,
            'swing': a_swing
        }
    
    # Process SELLS first
    for sym in list(positions.keys()):
        if sym not in signals: continue
        sig = signals[sym]
        if sig['action'] == 2 and sig['conf'] > confidence_threshold:
            pos = positions[sym]
            pnl = (sig['price'] - pos['entry']) / pos['entry']
            cash += pos['qty'] * sig['price']
            trade_log.append({
                'date': current_date, 'symbol': sym, 'action': 'SELL', 
                'reason': f'ENSEMBLE (S:{sig["sharpe"]}/W:{sig["swing"]})', 
                'pnl': pnl, 'conf': sig['conf']
            })
            del positions[sym]
    
    # Process BUYS
    buy_signals = [(sym, sig) for sym, sig in signals.items() 
                   if sig['action'] == 1 and sig['conf'] > confidence_threshold and sym not in positions]
    buy_signals.sort(key=lambda x: x[1]['conf'], reverse=True)
    
    for sym, sig in buy_signals:
        if len(positions) >= max_positions: break
        if cash < 100: break
        
        alloc = cash / min(5, max_positions - len(positions))
        qty = alloc / sig['price']
        positions[sym] = {'qty': qty, 'entry': sig['price']}
        cash -= alloc
        trade_log.append({
            'date': current_date, 'symbol': sym, 'action': 'BUY',
            'reason': f'ENSEMBLE (S:{sig["sharpe"]}/W:{sig["swing"]})',
            'conf': sig['conf']
        })
    
    # Update equity
    current_equity = cash + sum(pos['qty'] * market_data[sym]['df'].loc[current_date]['Close'] 
                                 for sym, pos in positions.items() 
                                 if current_date in market_data[sym]['df'].index)
    equity_curve.append({'Date': current_date, 'Equity': current_equity})

# Final calc
final_equity = cash
for sym, pos in positions.items():
    try:
        last_price = market_data[sym]['df']['Close'].iloc[-1]
        final_equity += pos['qty'] * last_price
    except: pass

# Results
print(f'\n🏁 DUAL ENSEMBLE: sharpe_gen3_ep2 + swing_gen7_ep380')
print(f'💰 Final Equity: ${final_equity:,.2f} ({(final_equity/10000-1)*100:+.1f}%)')
print(f'📊 Total Trades: {len(trade_log)}')

wins = [t for t in trade_log if t['action'] == 'SELL' and t.get('pnl', 0) > 0]
losses = [t for t in trade_log if t['action'] == 'SELL' and t.get('pnl', 0) <= 0]
if wins or losses:
    print(f'✅ Win Rate: {len(wins)/(len(wins)+len(losses))*100:.1f}%')
print(f'📦 Final Positions: {len(positions)}')

# Save
df_log = pd.DataFrame(trade_log)
df_log.to_csv('logs/trades_dual_ensemble.csv', index=False)
print(f'\n📁 Saved to logs/trades_dual_ensemble.csv')
