"""Backtest with monthly deposits + margin trading"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

# Feature columns
FEATURES_11 = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12']

# Sharpe Model Features (25)
FEATURES_25 = [
    'sma_10', 'sma_20', 'sma_50', 'sma_200', 
    'atr', 'bb_width', 'bb_upper', 'bb_lower', 
    'ema_12', 'ema_26', 
    'macd', 'macd_signal', 'macd_diff', 
    'adx', 'rsi', 'stoch_k', 'stoch_d', 
    'williams_r', 'roc', 'bb_pband', 
    'volume_sma', 'volume_ratio', 'obv', 'mfi', 'price_vs_sma20'
]

# === SIMULATION PARAMETERS (MATCHING LIVE AUDIT) ===
# Audit: Equity $10k, Cash -$9k -> Margin Used
# Audit: 40 Positions -> ~2.5% Allocation
STARTING_EQUITY = 10000.0   # 10k Start
MONTHLY_DEPOSIT = 0.0       # No Deposits
MONTHS_TO_SIMULATE = 6      # 6 Month Projection
MARGIN_MULTIPLIER = 1.0     # Cash Only (No Margin)
CONFIDENCE_THRESHOLD = 0.50 # Buy: Standard Entry
SELL_CONFIDENCE = 0.20      # Sell: Standard Exit
POSITION_SIZE_PCT = 0.05    # 5% (20 Positions)
USE_STOPS = True            # Enable Trailing Stops

# Exit Settings from Live Config/Training Config
STOP_ATR_MULT = 4.0
PROFIT_ATR_MULT = 5.0

print(f"SIMULATION PARAMETERS:")
print(f"   Starting Equity: ${STARTING_EQUITY:,.2f}")
print(f"   Monthly Deposit: ${MONTHLY_DEPOSIT:,.2f}")
print(f"   Total Deposits Over 2 Years: ${MONTHLY_DEPOSIT * MONTHS_TO_SIMULATE:,.2f}")
print(f"   Margin: {MARGIN_MULTIPLIER}x")
print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = 'models/swing_gen8_all_last'
if len(sys.argv) > 1:
    model_path = sys.argv[1]
print(f"\nLoading model: {model_path}")

# Detect Feature Set
if "sharpe" in model_path:
    print("ℹ️ Detected SHARPE model (Using 25 Features)")
    feature_list = FEATURES_25
    ts_dim = 25
else:
    print("ℹ️ Detected SWING model (Using 11 Features)")
    feature_list = FEATURES_11
    ts_dim = 11

agent = EnsembleAgent(time_series_dim=ts_dim, vision_channels=ts_dim, action_dim=3, device=device)
agent.load(model_path)
agent.set_eval()

# Load historical data
data_dir = "data/historical_swing"
print(f"\nLoading data from {data_dir}...")
files = glob.glob(os.path.join(data_dir, "*_1D.csv"))

# --- (Exclusion logic removed - data already isolated in folder) ---
print(f"ℹ️ Testing on {len(files)} 'Unseen' validation symbols")
# ---------------------------------------------------------

market_data = {}
all_dates = set()

for f in tqdm(files, desc="Loading"):
    try:
        df = pd.read_csv(f)
        symbol = os.path.basename(f).replace("_1D.csv", "")
        
        df.columns = [c.lower() for c in df.columns]
        col_map = {'o': 'Open', 'open': 'Open', 'h': 'High', 'high': 'High',
                   'l': 'Low', 'low': 'Low', 'c': 'Close', 'close': 'Close',
                   'v': 'Volume', 'volume': 'Volume', 't': 'Date', 'date': 'Date', 'timestamp': 'Date'}
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])
        # Ensure timezone-naive
        if df['Date'].dt.tz is not None:
             df['Date'] = df['Date'].dt.tz_localize(None)
             
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        df = add_technical_indicators(df)
        df = df.dropna()
        
        # Ensure all features exist
        for c in feature_list:
            if c not in df.columns:
                df[c] = 0.0

        if len(df) < 100:
            continue
            
        # Normalize features
        norm_features = df[feature_list].values.copy()
        for j in range(norm_features.shape[1]):
            col_mean = norm_features[:, j].mean()
            col_std = norm_features[:, j].std() + 1e-8
            norm_features[:, j] = (norm_features[:, j] - col_mean) / col_std
        
        market_data[symbol] = {'df': df, 'norm_features': norm_features, 'dates': df.index}
        all_dates.update(df.index)
    except:
        continue

print(f"Loaded {len(market_data)} symbols")
valid_dates = sorted(list(all_dates))

# Filter to requested duration
cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=int(MONTHS_TO_SIMULATE * 30.5)))
valid_dates = [d for d in valid_dates if d >= cutoff_date]
print(f"Date range: {valid_dates[0].date()} to {valid_dates[-1].date()}")

# Track monthly deposit dates
first_date = valid_dates[0]
deposit_dates = []
for m in range(MONTHS_TO_SIMULATE):
    deposit_date = first_date + pd.DateOffset(months=m)
    # Find nearest trading day
    nearest = min(valid_dates, key=lambda x: abs(x - deposit_date))
    deposit_dates.append(nearest)
deposit_dates = set(deposit_dates)

# Initialize
cash = STARTING_EQUITY
positions = {}
equity_curve = []
trade_log = []
wins = 0
losses = 0
total_deposited = STARTING_EQUITY
window_size = 60

print(f"\nRunning backtest with monthly ${MONTHLY_DEPOSIT:,.0f} deposits...")

for current_date in tqdm(valid_dates):
    # Monthly deposit
    if current_date in deposit_dates and current_date != first_date:
        cash += MONTHLY_DEPOSIT
        total_deposited += MONTHLY_DEPOSIT
    
    # Calculate current equity (with margin buying power)
    position_value = 0
    for sym, pos in positions.items():
        if sym in market_data and current_date in market_data[sym]['df'].index:
            price = market_data[sym]['df'].loc[current_date, 'Close']
            position_value += pos['qty'] * price
    
    equity = cash + position_value
    # buying_power = max(0, cash) * MARGIN_MULTIPLIER  # Incorrect for deep margin
    # Simplified Buying Power for Margin Account
    total_buying_power = equity * MARGIN_MULTIPLIER
    used_buying_power = position_value
    buying_power = max(0.0, total_buying_power - used_buying_power)
    
    equity_curve.append({
        'Date': current_date, 
        'Equity': equity, 
        'Cash': cash,
        'Positions': len(positions),
        'Deposited': total_deposited
    })
    
    # Prepare batch for inference
    active_symbols = []
    feed_tensors = []
    
    for symbol, data in market_data.items():
        df = data['df']
        if current_date not in df.index:
            continue
        try:
            curr_idx = df.index.get_loc(current_date)
            if curr_idx < window_size:
                continue
            input_window = data['norm_features'][curr_idx-window_size+1:curr_idx+1]
            if len(input_window) != window_size:
                continue
            active_symbols.append(symbol)
            feed_tensors.append(input_window)
        except:
            continue
    
    if not active_symbols:
        continue
    
    # Run inference
    batch_tensor = torch.FloatTensor(np.array(feed_tensors)).to(device)
    dummy_text = torch.zeros((len(active_symbols), 64), dtype=torch.long).to(device)
    
    with torch.no_grad():
        actions, confidences = agent.act(batch_tensor, dummy_text, dummy_text, return_q=True)
    
    # Handle batch_size=1 edge case (agent returns scalars instead of tensors)
    if isinstance(actions, int):
        actions = torch.tensor([actions], device=device)
        confidences = torch.tensor([confidences], device=device)

    # Build signals dict
    signals = {}
    for i, sym in enumerate(active_symbols):
        df = market_data[sym]['df']
        curr_row = df.loc[current_date]
        atr = curr_row.get('atr', curr_row['High'] - curr_row['Low'])
        signals[sym] = {
            'action': actions[i].item(),
            'conf': confidences[i].item(),
            'price': curr_row['Close'],
            'high': curr_row['High'],
            'low': curr_row['Low'],
            'atr': atr
        }
    
    # Process SELLS - Agent + Stops
    symbols_to_sell = []
    for sym, pos in positions.items():
        if sym not in signals:
            continue
        sig = signals[sym]
        price = sig['price']
        
        # Update Highest Price & Stop for Trailing Stop
        if USE_STOPS:
            if sig['high'] > pos['highest_price']:
                pos['highest_price'] = sig['high']
                # Trailing Stop Update
                new_stop = pos['highest_price'] - (pos['atr'] * STOP_ATR_MULT)
                pos['stop_price'] = max(pos.get('stop_price', 0), new_stop)

        # 1. Agent Sell (Asymmetric Threshold)
        # Sell if Action=2 AND Confidence > 0.3 (Easy Exit)
        if sig['action'] == 2 and sig['conf'] > SELL_CONFIDENCE:
            symbols_to_sell.append((sym, price, 'AGENT_SELL'))
        
        # 2. Stop Loss (Trailing)
        elif USE_STOPS and sig['low'] < pos['stop_price']:
             exit_price = min(price, pos['stop_price']) 
             symbols_to_sell.append((sym, exit_price, 'STOP_LOSS'))

        # 3. Take Profit
        elif USE_STOPS and sig['high'] > pos['entry_price'] + (pos['atr'] * PROFIT_ATR_MULT):
             exit_price = pos['entry_price'] + (pos['atr'] * PROFIT_ATR_MULT)
             symbols_to_sell.append((sym, exit_price, 'TAKE_PROFIT'))
    
    for sym, sell_price, reason in symbols_to_sell:
        pos = positions[sym]
        pnl = (sell_price - pos['entry_price']) * pos['qty']
        pnl_pct = (sell_price / pos['entry_price'] - 1) * 100
        
        cash += pos['qty'] * sell_price
        
        trade_log.append({
            'date': current_date,
            'symbol': sym,
            'action': 'SELL',
            'reason': reason,
            'price': sell_price,
            'qty': pos['qty'],
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
        
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        
        del positions[sym]
    
    # Process BUYS (using margin)
    buy_signals = [(sym, sig) for sym, sig in signals.items() 
                   if sig['action'] == 1 and sig['conf'] > CONFIDENCE_THRESHOLD and sym not in positions]
    buy_signals = sorted(buy_signals, key=lambda x: x[1]['conf'], reverse=True)
    
    # Recalculate buying power after sells
    # buying_power = max(0, cash) * MARGIN_MULTIPLIER
    used_buying_power = sum(pos['qty'] * market_data[s]['df']['Close'].loc[current_date] 
                           for s, pos in positions.items() if s in market_data and current_date in market_data[s]['df'].index)
    buying_power = max(0.0, (equity * MARGIN_MULTIPLIER) - used_buying_power)
    
    for sym, sig in buy_signals:
        # Enforce Max 20 Positions
        if len(positions) >= 20:
            break

        if buying_power < 100:  # Min buying power
            break
            
        price = sig['price']
        # Position Size = 10% of Total Buying Power (Equity * Margin)
        # This allows 10 positions at 2x leverage
        position_size = (equity * MARGIN_MULTIPLIER) * POSITION_SIZE_PCT
        qty = int(position_size / price)
        cost = qty * price
        
        if qty > 0 and cost <= buying_power:
            cash -= cost  # Can go negative (margin)
            buying_power -= cost # Simple deduction
            
            positions[sym] = {
                'qty': qty,
                'entry_price': price,
                'atr': sig['atr'],
                'highest_price': price, # Track high for trailing stop
                'stop_price': price - (sig['atr'] * STOP_ATR_MULT) # Initial Stop
            }
            
            trade_log.append({
                'date': current_date,
                'symbol': sym,
                'action': 'BUY',
                'price': price,
                'qty': qty,
                'pnl': 0,
                'pnl_pct': 0
            })

# Final equity
final_position_value = 0
for sym, pos in positions.items():
    if sym in market_data:
        last_price = market_data[sym]['df']['Close'].iloc[-1]
        final_position_value += pos['qty'] * last_price

final_equity = cash + final_position_value
total_return = (final_equity / total_deposited - 1) * 100
pure_gain = final_equity - total_deposited

# Results
print(f"\n{'='*70}")
print(f"BACKTEST RESULTS (With ${MONTHLY_DEPOSIT:,.0f}/month Deposits + {MARGIN_MULTIPLIER}x Margin)")
print(f"{'='*70}")
print(f"Period: {valid_dates[0].date()} to {valid_dates[-1].date()} ({MONTHS_TO_SIMULATE} months)")
print()
print(f"CAPITAL:")
print(f"   Starting Equity: ${STARTING_EQUITY:,.2f}")
print(f"   Monthly Deposits: ${MONTHLY_DEPOSIT:,.2f} x {MONTHS_TO_SIMULATE-1} months")
print(f"   Total Deposited: ${total_deposited:,.2f}")
print()
print(f"RESULTS:")
print(f"   Final Equity: ${final_equity:,.2f}")
print(f"   Pure Trading Gain: ${pure_gain:,.2f}")
print(f"   Return on Deposits: {total_return:+.1f}%")
print()
print(f"TRADING STATS:")
print(f"   Total Trades: {len([t for t in trade_log if t['action'] == 'SELL'])}")
print(f"   Wins: {wins} | Losses: {losses}")
if wins + losses > 0:
    print(f"   Win Rate: {wins/(wins+losses)*100:.1f}%")
print(f"   Final Positions: {len(positions)}")
print(f"   Final Cash: ${cash:,.2f}")
print()

# Max drawdown
eq_df = pd.DataFrame(equity_curve)
eq_df['peak'] = eq_df['Equity'].cummax()
eq_df['drawdown'] = (eq_df['Equity'] / eq_df['peak'] - 1) * 100
max_dd = eq_df['drawdown'].min()
print(f"Max Drawdown: {max_dd:.1f}%")

# Comparison
print()
print(f"{'='*70}")
print(f"COMPARISON:")
print(f"{'='*70}")
no_trading_value = total_deposited  # Just deposits, no trading
spy_approx_return = 0.25  # ~25% over 2 years historical average
spy_value = total_deposited * (1 + spy_approx_return)

print(f"   Just Savings (no trading): ${no_trading_value:,.2f}")
print(f"   SPY Buy & Hold (~25% 2yr): ${spy_value:,.2f}")
print(f"   Your Strategy:             ${final_equity:,.2f}")
print()
print(f"   You made ${final_equity - no_trading_value:,.2f} MORE than just saving")
print(f"   You made ${final_equity - spy_value:,.2f} MORE than SPY buy & hold")

# Save
model_base = os.path.basename(model_path)
eq_df.to_csv(f'logs/equity_{model_base}.csv', index=False)
trades_df = pd.DataFrame(trade_log)
trades_df.to_csv(f'logs/trades_{model_base}.csv', index=False)

print(f"\nSaved equity curve to logs/equity_{model_base}.csv")
print(f"Saved trade log to logs/trades_{model_base}.csv")