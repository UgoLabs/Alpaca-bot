
import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

def load_aligned_data(data_dir, start_date=None, end_date=None):
    """
    Loads all CSVs and aligns them to a common date index.
    Returns:
        market_data: dict {symbol: dataframe}
        valid_dates: sorted list of dates
    """
    print(f"📥 Loading Portfolio Data from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "*_1D.csv"))
    
    market_data = {}
    all_dates = set()
    
    # Pre-compute indicators for speed
    for f in tqdm(files, desc="Loading Symbols"):
        try:
            df = pd.read_csv(f)
            symbol = os.path.basename(f).replace("_1D.csv", "")
            
            # Standardize
            df.columns = [c.lower() for c in df.columns]
            col_map = {
                'o': 'Open', 'open': 'Open',
                'h': 'High', 'high': 'High',
                'l': 'Low', 'low': 'Low',
                'c': 'Close', 'close': 'Close',
                'v': 'Volume', 'volume': 'Volume',
                'adj close': 'Close' # yfinance often gives this
            }
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
                    
            if 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                continue
                
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Date Filter
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            if df.empty:
                continue
                
            # Add Indicators
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Feature Columns check
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12']
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                # Debug only first few failures
                # if len(market_data) < 3:
                #    print(f"Missing cols for {symbol}: {missing}. Columns: {df.columns.tolist()}")
                continue
            
            # Normalize Features needed for Agent
            features = df[feature_cols].values
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
            # Store normalized features in DF for easy access
            # We store as a list of arrays because DataFrame cells can hold objects
            # Or better, just keep a parallel numpy array map
            
            market_data[symbol] = {
                'df': df,
                'norm_features': norm_features,
                'dates': df.index
            }
            all_dates.update(df.index)
            
        except Exception:
            continue
            
    print(f"✅ Loaded {len(market_data)} symbols.")
    return market_data, sorted(list(all_dates))

def run_portfolio_backtest(
    model_path,
    data_dir="data/historical_swing",
    start_date="2024-01-01",
    end_date=None,
    initial_cash=10000.0,
    max_positions=10,
    stop_atr_mult=6.0,
    profit_atr_mult=3.0,
    confidence_threshold=0.40  # With temp=0.01, ~30% of signals pass this
):
    # 1. Setup Device & Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Portfolio Backtest | Device: {device} | Model: {os.path.basename(model_path)}")
    print(f"⚙️  Settings: Max Pos={max_positions} | Stop={stop_atr_mult}x | Profit={profit_atr_mult}x | Conf Threshold={confidence_threshold}")
    
    # Load Dummy Data to init agent size
    agent = EnsembleAgent(time_series_dim=11, vision_channels=11, action_dim=3, device=device)
    # model_path should be the prefix like "models/ensemble_ep200" (without _balanced.pth etc)
    # If user passes full path, strip the suffix
    import re
    model_prefix = re.sub(r'_(aggressive|balanced|conservative)\.pth$', '', model_path)
    model_prefix = model_prefix.replace('.pth', '')  # fallback if no suffix match
    agent.load(model_prefix)
    agent.set_eval()
    
    # 2. Load Data
    market_data, valid_dates = load_aligned_data(data_dir, start_date, end_date)
    
    # 3. Simulation State
    cash = initial_cash
    positions = {} # {symbol: {qty, entry_price, highest_price, stop_price}}
    equity_curve = []
    trade_log = []
    
    window_size = 60
    
    print(f"📅 Simulating from {valid_dates[0].date()} to {valid_dates[-1].date()}...")
    
    for current_date in tqdm(valid_dates):
        # A. Update Holdings & Check Exits (Stops/Targets/Signals)
        current_equity = cash
        
        # Prepare Batch Inference for potential entries and current holdings
        active_symbols = []
        feed_tensors = []
        
        # 1. Identify which symbols have data for today
        # We need at least window_size history up to today
        
        for symbol, data in market_data.items():
            df = data['df']
            if current_date not in df.index:
                continue
                
            # Get integer location of current date
            try:
                # Optimized lookup?
                curr_idx = df.index.get_loc(current_date)
                if curr_idx < window_size:
                    continue
                    
                # Extract Window
                # stored 'norm_features' handles normalization
                input_window = data['norm_features'][curr_idx-window_size+1 : curr_idx+1]
                
                if len(input_window) != window_size:
                    continue
                    
                active_symbols.append(symbol)
                feed_tensors.append(input_window)
                
            except KeyError:
                continue
        
        if not active_symbols:
            equity_curve.append({'Date': current_date, 'Equity': current_equity})
            continue
            
        # 2. Run Inference in One Batch
        batch_tensor = torch.FloatTensor(np.array(feed_tensors)).to(device)
        dummy_text = torch.zeros((len(active_symbols), 64), dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Get Q-values for relative confidence
            actions, confidences = agent.act(batch_tensor, dummy_text, dummy_text, return_q=True)
            
        # Map back to symbols
        signals = {} # symbol -> {action, conf, price, atr}
        
        for i, sym in enumerate(active_symbols):
            df = market_data[sym]['df']
            curr_row = df.loc[current_date]
            price = curr_row['Close']
            
            # Simple ATR calc (using High-Low as approximation for speed or pre-calc?)
            # Actually we can compute rolling ATR in load_aligned_data but for backtest speed let's approx margin
            # Or better, re-calc proper ATR
            # For this script, let's assume 'atr' column exists from pre-processing
            # but 'indicators.py' might verify column names
            atr = (curr_row['High'] - curr_row['Low']) if 'atr' not in curr_row else curr_row['atr']
            # Re-read indicators if adding technicals didn't add 'atr' column explicitly?
            # It usually adds 'atr' or we can infer.
            
            signals[sym] = {
                'action': actions[i].item(), # 0:HOLD, 1:BUY, 2:SELL
                'conf': confidences[i].item(),
                'price': price,
                'high': curr_row['High'],
                'low': curr_row['Low'],
                'atr': atr
            }

        # B. Process Exits (SELL signals or Stops)
        symbols_to_sell = []
        
        for sym, pos in positions.items():
            if sym not in signals:
                # No data today, skip (market closed for this ticker?)
                # Update equity with last known price
                current_equity += pos['qty'] * pos['entry_price'] # Estimate
                continue
            
            sig = signals[sym]
            curr_price = sig['price']
            
            # Update Trailing Stop logic
            if curr_price > pos['highest_price']:
                pos['highest_price'] = curr_price
                # Trailing Stop: Highest High - ATR * Mult
                # We use the ATR from the signal day for dynamic width? Or usage at entry?
                # Standard Chandelier uses current ATR.
                new_stop = pos['highest_price'] - (sig['atr'] * stop_atr_mult)
                pos['stop_price'] = max(pos['stop_price'], new_stop)
            
            # Check Stop Loss
            if sig['low'] < pos['stop_price']:
                symbols_to_sell.append((sym, pos['stop_price'], "STOP_LOSS"))
                continue
                
            # Check Profit Take
            if curr_price > pos['entry_price'] + (sig['atr'] * profit_atr_mult):
                 symbols_to_sell.append((sym, curr_price, "TAKE_PROFIT"))
                 continue
                 
            # Check Agent Signal (SELL + Confidence)
            # CONFIDENCE FIX: Only sell if confidence > threshold
            if sig['action'] == 2 and sig['conf'] > confidence_threshold:
                symbols_to_sell.append((sym, curr_price, "AGENT_SELL"))
                
        # Execute Sells
        for sym, price, reason in symbols_to_sell:
            pos = positions.pop(sym)
            proceeds = pos['qty'] * price
            cash += proceeds
            
            # Log
            pnl = (price - pos['entry_price']) * pos['qty']
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            trade_log.append({
                'Date': current_date, 'Symbol': sym, 'Side': 'SELL', 
                'Price': price, 'Reason': reason, 'PnL': pnl, 'PnL%': pnl_pct
            })
            
        # C. Process Entries (BUY signals)
        # 1. Rank opportunities
        # Only look at BUY signals (Action=1) that we don't own
        opportunities = []
        for sym, sig in signals.items():
            if sym not in positions and sig['action'] == 1 and sig['conf'] > confidence_threshold:
                opportunities.append((sym, sig))
                
        # Sort by Confidence Descending
        opportunities.sort(key=lambda x: x[1]['conf'], reverse=True)
        
        # 2. Fill Slots
        open_slots = max_positions - len(positions)
        
        # Position Sizing: Equal Weight of current equity? 
        # Or Equal Weight of Initial Capital? 
        # "Sniper" usually implies Fixed Fractional or Max Allocation.
        # Let's use: (Total Equity / Max Positions) as target size.
        
        # Recalculate Equity for sizing
        holdings_value = sum(p['qty'] * signals[p['symbol']]['price'] for p in positions.values() if p['symbol'] in signals)
        # Note: If symbol missing data today, use entry price (approx)
        holdings_value += sum(p['qty'] * p['entry_price'] for p in positions.values() if p['symbol'] not in signals)
        
        total_equity = cash + holdings_value
        target_size = total_equity / max_positions
        
        for sym, sig in opportunities[:open_slots]:
            if cash < target_size:
                # Not enough cash to fill full slot, take what we can?
                # Or skip? Let's take what we can if > 0
                max_buy = cash
            else:
                max_buy = target_size
                
            if max_buy < 10.0: # Minimum trade check
                continue
                
            buy_price = sig['price']
            qty = max_buy / buy_price
            
            cash -= (qty * buy_price)
            
            # Initial Stop
            stop_price = buy_price - (sig['atr'] * stop_atr_mult)
            
            positions[sym] = {
                'symbol': sym,
                'qty': qty,
                'entry_price': buy_price,
                'highest_price': buy_price,
                'stop_price': stop_price,
                'date': current_date
            }
            
            trade_log.append({
                'Date': current_date, 'Symbol': sym, 'Side': 'BUY', 
                'Price': buy_price, 'Reason': f"Conf: {sig['conf']:.3f}", 'PnL': 0, 'PnL%': 0
            })

        # Record Daily Equity
        # Re-sum exact equity
        final_holdings_val = 0
        for sym, pos in positions.items():
            price = signals[sym]['price'] if sym in signals else pos['entry_price']
            final_holdings_val += pos['qty'] * price
            
        equity_curve.append({'Date': current_date, 'Equity': cash + final_holdings_val})

    # 4. Reporting
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    trades_df = pd.DataFrame(trade_log)
    
    print("\n🏁 Backtest Complete.")
    print(f"💰 Final Equity: ${equity_df['Equity'].iloc[-1]:.2f}")
    if not trades_df.empty:
        print(f"📊 Total Trades: {len(trades_df)}")
        print(f"✅ Win Rate: {len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100:.1f}%")
        print(trades_df['Reason'].value_counts())
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['Equity'], label='Strategy')
    plt.title(f"Sniper Strategy: {os.path.basename(model_path)}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/backtest_portfolio_{os.path.basename(model_path)}.png")
    print(f"📈 Plot saved to plots/backtest_portfolio_{os.path.basename(model_path)}.png")

    # CSV Export
    trades_df.to_csv(f"logs/trades_{os.path.basename(model_path)}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model file")
    parser.add_argument("--start", default="2024-01-01", help="Start Date")
    args = parser.parse_args()
    
    run_portfolio_backtest(args.model, start_date=args.start)
