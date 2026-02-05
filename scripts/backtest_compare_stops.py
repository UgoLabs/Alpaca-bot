
import os, sys, glob, re, argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

def run_simulation(model_path, start_date, use_hard_stop=False, stop_pct=0.05):
    print(f"\nSTARTING SIMULATION: Hard Stop = {use_hard_stop} ({stop_pct*100}%)")
    
    # Load data
    data_dir = 'data/historical_swing'
    files = glob.glob(os.path.join(data_dir, '*_1D.csv'))
    market_data = {}
    all_dates = set()

    # Feature columns (Hardcoded for standard ensemble)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12']

    for f in tqdm(files, desc='Loading Data'):
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
            
            if any(c not in df.columns for c in feature_cols): continue
            features = df[feature_cols].values
            mean, std = np.mean(features, axis=0), np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            norm_features = np.clip(norm_features, -10, 10)
            norm_features = np.nan_to_num(norm_features, nan=0.0)
            market_data[symbol] = {'df': df, 'norm_features': norm_features}
            all_dates.update(df.index)
        except: continue

    valid_dates = sorted(list(all_dates))
    
    # Load Agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_dim = 11
    agent = EnsembleAgent(time_series_dim=feature_dim, vision_channels=feature_dim, action_dim=3, device=device)
    model_prefix = re.sub(r'_(aggressive|balanced|conservative)\.pth$', '', model_path).replace('.pth', '')
    agent.load(model_prefix)
    agent.set_eval()

    # Sim State
    cash = 10185.0  # Current Equity roughly
    positions = {}
    trade_log = []
    equity_curve = []
    window_size = 60
    confidence_threshold = 0.60 # Current Live Threshold

    for current_date in tqdm(valid_dates, desc='Simulating'):
        current_equity = cash
        active_symbols, feed_tensors = [], []
        
        # Prepare Batch
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
            continue
        
        # Inference
        batch = torch.tensor(np.array(feed_tensors), dtype=torch.float32, device=device)
        dummy_text = torch.zeros((len(active_symbols), 64), dtype=torch.long, device=device)
        with torch.no_grad():
            actions, confidences = agent.act(batch, dummy_text, dummy_text, return_q=True)
        
        signals = {}
        for i, sym in enumerate(active_symbols):
            df = market_data[sym]['df']
            row = df.loc[current_date]
            signals[sym] = {'action': actions[i], 'conf': confidences[i].item(), 'price': row['Close'], 'low': row['Low']}
        
        # --- SELLS ---
        symbols_to_sell = []
        for sym, pos in positions.items():
            if sym not in signals: continue
            sig = signals[sym]
            
            # 1. HARD STOP CHECK
            if use_hard_stop:
                stop_price = pos['entry_price'] * (1 - stop_pct)
                if sig['low'] <= stop_price:
                    symbols_to_sell.append((sym, stop_price, 'STOP_LOSS')) # Assume filled at stop
                    continue

            # 2. AGENT SELL
            if sig['action'] == 2 and sig['conf'] > confidence_threshold:
                symbols_to_sell.append((sym, sig['price'], 'AGENT_SELL'))
        
        for sym, price, reason in symbols_to_sell:
            if sym not in positions: continue # Already sold?
            pos = positions.pop(sym)
            proceeds = pos['qty'] * price
            cash += proceeds
            pnl = (price - pos['entry_price']) * pos['qty']
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            trade_log.append({'Date': current_date, 'Symbol': sym, 'Side': 'SELL', 'Price': price, 'Reason': reason, 'PnL': pnl, 'PnL%': pnl_pct})

        # --- BUYS ---
        opportunities = [(sym, sig) for sym, sig in signals.items() if sym not in positions and sig['action'] == 1 and sig['conf'] > confidence_threshold]
        opportunities.sort(key=lambda x: x[1]['conf'], reverse=True)
        
        # Max Positions (Matches current live config approx 40)
        max_positions = 40 
        
        holdings_val = sum(p['qty'] * signals.get(sym, {'price': p['entry_price']})['price'] for sym, p in positions.items())
        current_equity = cash + holdings_val
        
        # Size: 2.5% per trade (1/40th)
        position_size = current_equity * 0.025
        
        for sym, sig in opportunities:
            if len(positions) >= max_positions: break
            if cash < position_size: break
            
            price = sig['price']
            qty = int(position_size / price)
            if qty <= 0: continue
            cost = qty * price
            cash -= cost
            positions[sym] = {'qty': qty, 'entry_price': price}
            trade_log.append({'Date': current_date, 'Symbol': sym, 'Side': 'BUY', 'Price': price, 'Reason': f"Conf:{sig['conf']:.2f}"})

        # Record Daily Equity
        holdings_val = sum(p['qty'] * signals.get(sym, {'price': p['entry_price']})['price'] for sym, p in positions.items())
        equity_curve.append({'Date': current_date, 'Equity': cash + holdings_val})

    # Stats
    final_eq = equity_curve[-1]['Equity']
    ret = (final_eq - 10185.0) / 10185.0
    
    trades = pd.DataFrame(trade_log)
    sells = trades[trades['Side'] == 'SELL']
    win_rate = (len(sells[sells['PnL'] > 0]) / len(sells)) * 100 if len(sells) > 0 else 0
    
    # Calculate Drawdown
    eq_df = pd.DataFrame(equity_curve)
    eq_df['Peak'] = eq_df['Equity'].cummax()
    eq_df['Drawdown'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak']
    max_dd = eq_df['Drawdown'].min() * 100

    print("-" * 40)
    print(f"RESULTS (Hard Stop={use_hard_stop})")
    print("-" * 40)
    print(f"Final Equity:   ${final_eq:,.2f} ({ret*100:+.2f}%)")
    print(f"Max Drawdown:   {max_dd:.2f}%")
    print(f"Win Rate:       {win_rate:.1f}%")
    print(f"Total Trades:   {len(trades)}")
    return final_eq, max_dd, win_rate

if __name__ == "__main__":
    # Use the known good model path from config
    model_path = "models/swing_gen7_refined_ep380_balanced.pth"
    
    print("Running Comparative Backtest (2024-Present)...")
    
    # Run 1: Current Config (Brain Only)
    eq1, dd1, wr1 = run_simulation(model_path, '2024-01-01', use_hard_stop=False)
    
    # Run 2: Hard Stop (5%)
    eq2, dd2, wr2 = run_simulation(model_path, '2024-01-01', use_hard_stop=True, stop_pct=0.05)

    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Metric':<15} {'Brain Only (Current)':<20} {'Hard Stop (5%)':<20}")
    print("-" * 60)
    print(f"{'Final Equity':<15} ${eq1:<19,.2f} ${eq2:<19,.2f}")
    print(f"{'Return':<15} {(eq1/10185-1)*100:<+19.1f}% {(eq2/10185-1)*100:<+19.1f}%")
    print(f"{'Max Drawdown':<15} {dd1:<19.2f}% {dd2:<19.2f}%")
    print(f"{'Win Rate':<15} {wr1:<19.1f}% {wr2:<19.1f}%")
