"""Backtest using the existing portfolio backtest, starting with current 40 positions"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import alpaca_trade_api as tradeapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators
from config.settings import SwingTraderCreds, ALPACA_BASE_URL

# Feature columns
FEATURES_11 = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12']

# Get current positions
api = tradeapi.REST(str(SwingTraderCreds.API_KEY), str(SwingTraderCreds.API_SECRET), str(ALPACA_BASE_URL), api_version='v2')
live_positions = api.list_positions()
account = api.get_account()

print(f"Loading {len(live_positions)} live positions...")
starting_portfolio = {}
for p in live_positions:
    starting_portfolio[p.symbol] = {
        'qty': int(p.qty),
        'entry_price': float(p.avg_entry_price)
    }

current_equity = float(account.equity)
current_cash = float(account.cash)
print(f"   Current Equity: ${current_equity:,.2f}")
print(f"   Current Cash: ${current_cash:,.2f}")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = 'models/swing_gen7_refined_ep380'
print(f"\nLoading model: {model_path}")

agent = EnsembleAgent(time_series_dim=11, vision_channels=11, action_dim=3, device=device)
agent.load(model_path)
agent.set_eval()

# Load historical data
data_dir = "data/historical_swing"
print(f"\nLoading data from {data_dir}...")
files = glob.glob(os.path.join(data_dir, "*_1D.csv"))

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
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        df = add_technical_indicators(df)
        df = df.dropna()
        
        if len(df) < 100:
            continue
            
        # Normalize features
        feature_cols = FEATURES_11
        norm_features = df[feature_cols].values.copy()
        for j in range(norm_features.shape[1]):
            col_mean = norm_features[:, j].mean()
            col_std = norm_features[:, j].std() + 1e-8
            norm_features[:, j] = (norm_features[:, j] - col_mean) / col_std
        
        market_data[symbol] = {'df': df, 'norm_features': norm_features, 'dates': df.index}
        all_dates.update(df.index)
    except:
        continue

print(f"[OK] Loaded {len(market_data)} symbols")
valid_dates = sorted(list(all_dates))

# Filter to last 2 years only (out-of-sample)
from datetime import datetime, timedelta
cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=730))
valid_dates = [d for d in valid_dates if d >= cutoff_date]
print(f"Date range (2 years only): {valid_dates[0].date()} to {valid_dates[-1].date()}")

# Backtest settings
# CLEAN START: Assume we sold everything and paid off debt
# Starting Equity = $10,185.51 (User's actual Net Liquidation Value)
# Starting Cash = $10,185.51
# Positions = Empty
INITIAL_EQUITY = current_equity  # $10,185.51
INITIAL_CASH = current_equity    # All cash start
LEVERAGE = 2.0
# Using 0.40 threshold set below
POSITION_SIZE_PCT = 0.05
window_size = 60

positions = {} # Clean start

print(f"\nStarting Backtest (CLEAN START with 2x Margin)")
print(f"   Starting Equity: ${INITIAL_EQUITY:,.2f}")
print(f"   Starting Cash: ${INITIAL_CASH:,.2f}")
print(f"   Leverage: {LEVERAGE}x")

# Settings
CONFIDENCE_THRESHOLD = 0.40

# Tracking
equity_curve = []
trade_log = []
wins = 0
losses = 0

cash = INITIAL_CASH

print(f"\nRunning backtest...")
for current_date in tqdm(valid_dates):
    # Calculate current equity
    current_equity = cash
    portfolio_value = 0
    for sym, pos in positions.items():
        if sym in market_data and current_date in market_data[sym]['df'].index:
            price = market_data[sym]['df'].loc[current_date, 'Close']
            val = pos['qty'] * price
            current_equity += val
            portfolio_value += val
            
    equity_curve.append({'Date': current_date, 'Equity': current_equity})
    
    # Calculate Buying Power (Reg T Margin)
    # Buying Power = Equity * Leverage - Portfolio Value
    # E.g. $10k Equity * 2 = $20k BP. If holding $5k stocks, BP = $15k.
    buying_power = current_equity * LEVERAGE
    available_to_spend = buying_power - portfolio_value
    
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
    
    # Build signals dict
    signals = {}
    for i, sym in enumerate(active_symbols):
        df = market_data[sym]['df']
        curr_row = df.loc[current_date]
        signals[sym] = {
            'action': actions[i].item(),
            'conf': confidences[i].item(),
            'price': curr_row['Close']
        }
    
    # Process SELLS
    symbols_to_sell = []
    for sym, pos in positions.items():
        if sym not in signals:
            continue
        sig = signals[sym]
        # Sell if agent says sell with confidence
        if sig['action'] == 2 and sig['conf'] > CONFIDENCE_THRESHOLD:
            symbols_to_sell.append(sym)
    
    for sym in symbols_to_sell:
        pos = positions[sym]
        sell_price = signals[sym]['price']
        pnl = (sell_price - pos['entry_price']) * pos['qty']
        pnl_pct = (sell_price / pos['entry_price'] - 1) * 100
        
        proceeds = pos['qty'] * sell_price
        cash += proceeds
        # Buying power updates automatically next loop via equity calc
        
        trade_log.append({
            'date': current_date,
            'symbol': sym,
            'action': 'SELL',
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
    
    # Process BUYS - Use 2x Margin Logic
    buy_signals = [(sym, sig) for sym, sig in signals.items() 
                   if sig['action'] == 1 and sig['conf'] > CONFIDENCE_THRESHOLD and sym not in positions]
    buy_signals = sorted(buy_signals, key=lambda x: x[1]['conf'], reverse=True)
    
    for sym, sig in buy_signals[:5]:  # Allow up to 5 buys per day if margin allows
        price = sig['price']
        
        # Position sizing: 5% of Total Buying Power (aggressive)
        # Or 5% of Equity (conservative)?
        # Let's use 5% of Buying Power to utilize the leverage.
        # Buying Power is approx Equity * 2. So this is ~10% of Equity per trade.
        target_pos_size = (current_equity * LEVERAGE) * POSITION_SIZE_PCT
        
        qty = int(target_pos_size / price)
        cost = qty * price
        
        # Check buying power constraint (and minimum cash safety?)
        # Alpaca allows cash to go negative up to -Equity.
        if qty > 0 and available_to_spend >= cost:
            cash -= cost
            available_to_spend -= cost # Update intra-day BP
            
            positions[sym] = {
                'qty': qty,
                'entry_price': price
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
final_equity = cash
for sym, pos in positions.items():
    if sym in market_data:
        last_price = market_data[sym]['df']['Close'].iloc[-1]
        final_equity += pos['qty'] * last_price

total_return = (final_equity / INITIAL_EQUITY - 1) * 100

# Results
print(f"\n{'='*60}")
print(f"BACKTEST RESULTS (Clean Start, 2x Margin, Conf={CONFIDENCE_THRESHOLD})")
print(f"{'='*60}")
print(f"Period: {valid_dates[0].date()} to {valid_dates[-1].date()}")
print(f"Starting Equity: ${INITIAL_EQUITY:,.2f}")
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return:+.1f}%")
print()
print(f"Total Trades: {len(trade_log)}")
sells = [t for t in trade_log if t['action'] == 'SELL']
buys = [t for t in trade_log if t['action'] == 'BUY']
print(f"Sells: {len(sells)} | Buys: {len(buys)}")
print(f"Wins: {wins} | Losses: {losses}")
if wins + losses > 0:
    print(f"Win Rate: {wins/(wins+losses)*100:.1f}%")
print()
print(f"Final Positions: {len(positions)}")
print(f"Final Cash: ${cash:,.2f} {'[OFF MARGIN!]' if cash >= 0 else '[USING MARGIN]'}")

# Max drawdown
eq_df = pd.DataFrame(equity_curve)
eq_df['peak'] = eq_df['Equity'].cummax()
eq_df['drawdown'] = (eq_df['Equity'] / eq_df['peak'] - 1) * 100
max_dd = eq_df['drawdown'].min()
print(f"Max Drawdown: {max_dd:.1f}%")

# Save trade log
trade_df = pd.DataFrame(trade_log)
trade_df.to_csv(f'logs/trades_clean_margin_{CONFIDENCE_THRESHOLD}.csv', index=False)
print(f"\nTrade log saved to logs/trades_clean_margin_{CONFIDENCE_THRESHOLD}.csv")
