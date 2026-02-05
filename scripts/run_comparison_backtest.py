"""
Comparison Backtest Script
Compare Model 380 (swing_gen7_refined_ep380) vs Model 900 (ensemble_ep900)

Parameters:
- Initial Equity: $10,000
- Stop Loss: 4%
- Take Profit: 6%
- Period: Last 6 months (2025-07-28 to 2026-01-28)
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators


def load_market_data(data_dir, start_date, end_date):
    """Load and prepare market data for backtesting."""
    print(f"📥 Loading data from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "*_1D.csv"))
    
    market_data = {}
    all_dates = set()
    
    for f in tqdm(files, desc="Loading Symbols"):
        try:
            df = pd.read_csv(f)
            symbol = os.path.basename(f).replace("_1D.csv", "")
            
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            col_map = {
                'o': 'Open', 'open': 'Open',
                'h': 'High', 'high': 'High',
                'l': 'Low', 'low': 'Low',
                'c': 'Close', 'close': 'Close',
                'v': 'Volume', 'volume': 'Volume',
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
            # df = df[df.index >= pd.to_datetime(start_date)] # Moved below
            df = df[df.index <= pd.to_datetime(end_date)]
                
            if df.empty or len(df) < 70:  # Need enough history
                continue
                
            # Add Indicators
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Filter Start Date AFTER indicators are calculated
            df = df[df.index >= pd.to_datetime(start_date)]
            
            if df.empty:
                 continue
            
            # Feature columns - must match training script (25 features)
            feature_cols = [
                'sma_10', 'sma_20', 'sma_50', 'sma_200', 
                'atr', 'bb_width', 'bb_upper', 'bb_lower', 
                'ema_12', 'ema_26', 
                'macd', 'macd_signal', 'macd_diff', 
                'adx', 'rsi', 'stoch_k', 'stoch_d', 
                'williams_r', 'roc', 'bb_pband', 
                'volume_sma', 'volume_ratio', 'obv', 'mfi', 'price_vs_sma20'
            ]
            
            # Fill missing features with 0
            for c in feature_cols:
                if c not in df.columns:
                    df[c] = 0.0
            
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                continue
            
            # Normalize
            features = df[feature_cols].values
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            norm_features = (features - mean) / std
            
            market_data[symbol] = {
                'df': df,
                'norm_features': norm_features,
                'dates': df.index
            }
            all_dates.update(df.index)
            
        except Exception as e:
            continue
    
    valid_dates = sorted(all_dates)
    print(f"✅ Loaded {len(market_data)} symbols | {len(valid_dates)} trading days")
    return market_data, valid_dates


def run_backtest(
    model_path,
    model_name,
    data_dir="data/historical_swing",
    start_date="2025-07-28",
    end_date="2026-01-28",
    initial_cash=10000.0,
    max_positions=10,
    stop_loss_pct=0.04,    # 4% stop loss
    take_profit_pct=0.06,  # 6% take profit
    confidence_threshold=0.40
):
    """Run backtest with fixed percentage stop loss and take profit."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"🚀 BACKTEST: {model_name}")
    print(f"{'='*80}")
    print(f"📊 Model: {model_path}")
    print(f"💰 Initial Capital: ${initial_cash:,.2f}")
    print(f"🛑 Stop Loss: {stop_loss_pct*100:.1f}%")
    print(f"🎯 Take Profit: {take_profit_pct*100:.1f}%")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🖥️  Device: {device}")
    
    # Load Agent - 25 features to match training
    num_features = 25
    agent = EnsembleAgent(time_series_dim=num_features, vision_channels=num_features, action_dim=3, device=device)
    import re
    model_prefix = re.sub(r'_(aggressive|balanced|conservative)\.pth$', '', model_path)
    model_prefix = model_prefix.replace('.pth', '')
    
    try:
        agent.load(model_prefix)
        agent.set_eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load Data
    market_data, valid_dates = load_market_data(data_dir, start_date, end_date)
    
    if not valid_dates:
        print("❌ No data available for the specified period")
        return None
    
    # Simulation State
    cash = initial_cash
    positions = {}
    equity_curve = []
    trade_log = []
    window_size = 60
    
    print(f"📅 Simulating from {valid_dates[0].date()} to {valid_dates[-1].date()}...")
    
    for current_date in tqdm(valid_dates, desc="Backtesting"):
        current_equity = cash
        
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
            price = curr_row['Close']
            
            signals[sym] = {
                'action': actions[i].item(),
                'conf': confidences[i].item(),
                'price': price,
                'high': curr_row['High'],
                'low': curr_row['Low'],
            }

        # Process Exits
        symbols_to_sell = []
        
        for sym, pos in positions.items():
            if sym not in signals:
                current_equity += pos['qty'] * pos['entry_price']
                continue
            
            sig = signals[sym]
            curr_price = sig['price']
            low_price = sig['low']
            high_price = sig['high']
            
            # Calculate stop and target prices
            stop_price = pos['entry_price'] * (1 - stop_loss_pct)
            target_price = pos['entry_price'] * (1 + take_profit_pct)
            
            # Check Stop Loss (using low of the day)
            if low_price <= stop_price:
                symbols_to_sell.append((sym, stop_price, "STOP_LOSS"))
                continue
                
            # Check Take Profit (using high of the day)
            if high_price >= target_price:
                symbols_to_sell.append((sym, target_price, "TAKE_PROFIT"))
                continue
                 
            # Check Agent Sell Signal
            if sig['action'] == 2 and sig['conf'] > confidence_threshold:
                symbols_to_sell.append((sym, curr_price, "AGENT_SELL"))
                
        # Execute Sells
        for sym, price, reason in symbols_to_sell:
            pos = positions.pop(sym)
            proceeds = pos['qty'] * price
            cash += proceeds
            
            pnl = (price - pos['entry_price']) * pos['qty']
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            trade_log.append({
                'Date': current_date, 'Symbol': sym, 'Side': 'SELL', 
                'Price': price, 'Reason': reason, 'PnL': pnl, 'PnL%': pnl_pct
            })
            
        # Process Entries
        opportunities = []
        for sym, sig in signals.items():
            if sym not in positions and sig['action'] == 1 and sig['conf'] > confidence_threshold:
                opportunities.append((sym, sig))
                
        opportunities.sort(key=lambda x: x[1]['conf'], reverse=True)
        
        open_slots = max_positions - len(positions)
        
        # Position Sizing
        holdings_value = sum(
            p['qty'] * signals[p['symbol']]['price'] 
            for p in positions.values() if p['symbol'] in signals
        )
        total_equity = cash + holdings_value
        target_size = total_equity / max_positions
        
        for sym, sig in opportunities[:open_slots]:
            if cash < target_size:
                max_buy = cash
            else:
                max_buy = target_size
                
            if max_buy < 10.0:
                continue
                
            buy_price = sig['price']
            qty = max_buy / buy_price
            
            cash -= (qty * buy_price)
            
            positions[sym] = {
                'symbol': sym,
                'qty': qty,
                'entry_price': buy_price,
                'date': current_date
            }
            
            trade_log.append({
                'Date': current_date, 'Symbol': sym, 'Side': 'BUY', 
                'Price': buy_price, 'Reason': f"Conf: {sig['conf']:.3f}", 'PnL': 0, 'PnL%': 0
            })

        # Record Equity
        final_holdings_val = 0
        for sym, pos in positions.items():
            price = signals[sym]['price'] if sym in signals else pos['entry_price']
            final_holdings_val += pos['qty'] * price
            
        equity_curve.append({'Date': current_date, 'Equity': cash + final_holdings_val})

    # Results
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    trades_df = pd.DataFrame(trade_log)
    
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = (final_equity - initial_cash) / initial_cash * 100
    
    # Calculate metrics
    equity_values = equity_df['Equity'].values
    daily_returns = np.diff(equity_values) / equity_values[:-1]
    sharpe = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity_values)
    drawdown = (peak - equity_values) / peak
    max_dd = np.max(drawdown) * 100
    
    # Win Rate
    sell_trades = trades_df[trades_df['Side'] == 'SELL'] if not trades_df.empty else pd.DataFrame()
    wins = len(sell_trades[sell_trades['PnL'] > 0]) if not sell_trades.empty else 0
    total_trades = len(sell_trades) if not sell_trades.empty else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 RESULTS: {model_name}")
    print(f"{'='*50}")
    print(f"💰 Final Equity: ${final_equity:,.2f}")
    print(f"📈 Total Return: {total_return:+.2f}%")
    print(f"📉 Max Drawdown: {max_dd:.2f}%")
    print(f"📊 Sharpe Ratio: {sharpe:.2f}")
    print(f"🎯 Win Rate: {win_rate:.1f}% ({wins}/{total_trades})")
    
    if not trades_df.empty:
        print(f"\n📋 Trade Breakdown:")
        print(trades_df['Reason'].value_counts().to_string())
    
    return {
        'model_name': model_name,
        'equity_df': equity_df,
        'trades_df': trades_df,
        'final_equity': final_equity,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'total_trades': total_trades
    }


def main():
    # Configuration
    INITIAL_CASH = 10000.0
    STOP_LOSS = 0.04    # 4%
    TAKE_PROFIT = 0.06  # 6%
    MAX_POSITIONS = 5
    START_DATE = "2024-01-28"  # 2 YEAR backtest
    END_DATE = "2026-01-28"
    
    # Test Ens900 vs Ens1400 with 5 positions
    models = [
        ("models/ensemble_ep900_balanced.pth", "Ens900_5pos"),
        ("models/ensemble_ep1400", "Ens1400_5pos"),
    ]
    
    results = []
    
    for model_path, model_name in models:
        result = run_backtest(
            model_path=model_path,
            model_name=model_name,
            start_date=START_DATE,
            end_date=END_DATE,
            initial_cash=INITIAL_CASH,
            max_positions=MAX_POSITIONS,
            stop_loss_pct=STOP_LOSS,
            take_profit_pct=TAKE_PROFIT,
        )
        if result:
            results.append(result)
    
    if len(results) >= 2:
        # Plot Comparison
        plt.figure(figsize=(14, 8))
        
        for res in results:
            plt.plot(res['equity_df'].index, res['equity_df']['Equity'], 
                    label=f"{res['model_name']} ({res['total_return']:+.1f}%)", linewidth=2)
        
        plt.axhline(y=INITIAL_CASH, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        plt.title(f'Ens900 vs Ens1400: 5 Max Positions / 4% SL / 6% TP\n{START_DATE} to {END_DATE}', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/backtest_swing380_sl_comparison.png', dpi=150)
        print(f"\n📈 Comparison plot saved to plots/backtest_swing380_sl_comparison.png")
        
        # Summary Table
        print(f"\n{'='*80}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<35} {'Final $':<12} {'Return':<10} {'Max DD':<10} {'Sharpe':<8} {'Win Rate':<10}")
        print(f"{'-'*80}")
        for res in results:
            print(f"{res['model_name']:<35} ${res['final_equity']:<11,.2f} {res['total_return']:>+7.2f}% {res['max_drawdown']:>7.2f}% {res['sharpe']:>7.2f} {res['win_rate']:>7.1f}%")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
