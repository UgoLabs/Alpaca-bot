"""
Comprehensive Grid Search Backtest
- Models: swing_gen7_refined_ep380 + ensemble models (900-1400)
- Stop Loss: ATR-based (3x, 4x) and percentage-based (4%, 5%)
- Take Profit: 4%, 5%, 6%
- Max Positions: 10, 15, 40
- Period: Last 6 months
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
from itertools import product
import warnings
warnings.filterwarnings('ignore')

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
    
    for f in tqdm(files, desc="Loading Symbols", leave=False):
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
            df = df[df.index >= pd.to_datetime(start_date)]
            df = df[df.index <= pd.to_datetime(end_date)]
                
            if df.empty or len(df) < 70:
                continue
                
            # Add Indicators
            df = add_technical_indicators(df)
            
            # Calculate ATR
            high = df['High']
            low = df['Low']
            close = df['Close']
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean().bfill()
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Feature columns
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12']
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


def run_single_backtest(
    agent,
    market_data,
    valid_dates,
    device,
    initial_cash=10000.0,
    max_positions=10,
    stop_loss_pct=0.04,
    take_profit_pct=0.06,
    use_atr_stop=False,
    atr_stop_mult=3.0,
    confidence_threshold=0.40
):
    """Run a single backtest with given parameters."""
    
    cash = initial_cash
    positions = {}
    equity_curve = []
    trade_log = []
    window_size = 60
    
    for current_date in valid_dates:
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
            atr = curr_row['atr'] if 'atr' in curr_row else price * 0.02
            
            signals[sym] = {
                'action': actions[i].item(),
                'conf': confidences[i].item(),
                'price': price,
                'high': curr_row['High'],
                'low': curr_row['Low'],
                'atr': atr,
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
            
            # Calculate stop price (ATR-based or percentage-based)
            if use_atr_stop:
                stop_price = pos['entry_price'] - (pos['entry_atr'] * atr_stop_mult)
            else:
                stop_price = pos['entry_price'] * (1 - stop_loss_pct)
            
            target_price = pos['entry_price'] * (1 + take_profit_pct)
            
            # Check Stop Loss
            if low_price <= stop_price:
                symbols_to_sell.append((sym, stop_price, "STOP_LOSS"))
                continue
                
            # Check Take Profit
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
                'entry_atr': sig['atr'],
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

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    if equity_df.empty:
        return None
        
    equity_df = equity_df.set_index('Date')
    trades_df = pd.DataFrame(trade_log)
    
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = (final_equity - initial_cash) / initial_cash * 100
    
    equity_values = equity_df['Equity'].values
    daily_returns = np.diff(equity_values) / equity_values[:-1]
    sharpe = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
    
    peak = np.maximum.accumulate(equity_values)
    drawdown = (peak - equity_values) / peak
    max_dd = np.max(drawdown) * 100
    
    sell_trades = trades_df[trades_df['Side'] == 'SELL'] if not trades_df.empty else pd.DataFrame()
    wins = len(sell_trades[sell_trades['PnL'] > 0]) if not sell_trades.empty else 0
    total_trades = len(sell_trades) if not sell_trades.empty else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'equity_df': equity_df
    }


def main():
    # Configuration
    INITIAL_CASH = 10000.0
    START_DATE = "2025-07-28"
    END_DATE = "2026-01-28"
    DATA_DIR = "data/historical_swing"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Load data once
    market_data, valid_dates = load_market_data(DATA_DIR, START_DATE, END_DATE)
    
    # Models to test
    models = [
        ("models/swing_gen7_refined_ep380", "Swing380"),
    ]
    
    # Add ensemble models from 900 to 1400 (every 100)
    for ep in [900, 1000, 1100, 1200, 1300, 1400]:
        models.append((f"models/ensemble_ep{ep}", f"Ens{ep}"))
    
    # Parameter combinations
    param_grid = {
        'max_positions': [10, 15, 40],
        'take_profit_pct': [0.04, 0.05, 0.06],
        'stop_config': [
            {'use_atr_stop': False, 'stop_loss_pct': 0.04, 'atr_stop_mult': None, 'label': 'SL4%'},
            {'use_atr_stop': False, 'stop_loss_pct': 0.05, 'atr_stop_mult': None, 'label': 'SL5%'},
            {'use_atr_stop': True, 'stop_loss_pct': None, 'atr_stop_mult': 3.0, 'label': 'ATR3x'},
            {'use_atr_stop': True, 'stop_loss_pct': None, 'atr_stop_mult': 4.0, 'label': 'ATR4x'},
        ]
    }
    
    # Results storage
    all_results = []
    
    print(f"\n{'='*100}")
    print(f"🔬 COMPREHENSIVE GRID SEARCH BACKTEST")
    print(f"{'='*100}")
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"💰 Initial Capital: ${INITIAL_CASH:,.2f}")
    print(f"📊 Models: {len(models)}")
    print(f"⚙️  Max Positions: {param_grid['max_positions']}")
    print(f"🎯 Take Profit: {[f'{x*100}%' for x in param_grid['take_profit_pct']]}")
    print(f"🛑 Stop Configs: {[x['label'] for x in param_grid['stop_config']]}")
    
    total_combinations = len(models) * len(param_grid['max_positions']) * len(param_grid['take_profit_pct']) * len(param_grid['stop_config'])
    print(f"📈 Total Combinations: {total_combinations}")
    print(f"{'='*100}\n")
    
    # Run grid search
    combo_idx = 0
    
    for model_path, model_name in models:
        print(f"\n🔄 Loading {model_name}...")
        
        # Load agent
        agent = EnsembleAgent(time_series_dim=11, vision_channels=11, action_dim=3, device=device)
        import re
        model_prefix = re.sub(r'_(aggressive|balanced|conservative)\.pth$', '', model_path)
        model_prefix = model_prefix.replace('.pth', '')
        
        try:
            agent.load(model_prefix)
            agent.set_eval()
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue
        
        for max_pos in param_grid['max_positions']:
            for tp_pct in param_grid['take_profit_pct']:
                for stop_cfg in param_grid['stop_config']:
                    combo_idx += 1
                    
                    config_label = f"{model_name} | Pos:{max_pos} | TP:{int(tp_pct*100)}% | {stop_cfg['label']}"
                    print(f"  [{combo_idx}/{total_combinations}] {config_label}...", end=" ")
                    
                    result = run_single_backtest(
                        agent=agent,
                        market_data=market_data,
                        valid_dates=valid_dates,
                        device=device,
                        initial_cash=INITIAL_CASH,
                        max_positions=max_pos,
                        stop_loss_pct=stop_cfg['stop_loss_pct'] or 0.04,
                        take_profit_pct=tp_pct,
                        use_atr_stop=stop_cfg['use_atr_stop'],
                        atr_stop_mult=stop_cfg['atr_stop_mult'] or 3.0,
                    )
                    
                    if result:
                        result['model'] = model_name
                        result['max_positions'] = max_pos
                        result['take_profit'] = f"{int(tp_pct*100)}%"
                        result['stop_config'] = stop_cfg['label']
                        result['config_label'] = config_label
                        all_results.append(result)
                        
                        print(f"Return: {result['total_return']:+.1f}% | Sharpe: {result['sharpe']:.2f} | WR: {result['win_rate']:.0f}%")
                    else:
                        print("❌ No trades")
    
    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Model': r['model'],
            'Max Pos': r['max_positions'],
            'Take Profit': r['take_profit'],
            'Stop': r['stop_config'],
            'Final $': r['final_equity'],
            'Return %': r['total_return'],
            'Max DD %': r['max_drawdown'],
            'Sharpe': r['sharpe'],
            'Win Rate %': r['win_rate'],
            'Trades': r['total_trades']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by Return
    summary_df = summary_df.sort_values('Return %', ascending=False)
    
    # Save results
    os.makedirs('logs', exist_ok=True)
    summary_df.to_csv('logs/grid_search_results.csv', index=False)
    
    # Print Top 20
    print(f"\n{'='*120}")
    print("🏆 TOP 20 CONFIGURATIONS BY RETURN")
    print(f"{'='*120}")
    print(summary_df.head(20).to_string(index=False))
    
    # Print Top 10 by Sharpe
    print(f"\n{'='*120}")
    print("📊 TOP 10 CONFIGURATIONS BY SHARPE RATIO")
    print(f"{'='*120}")
    sharpe_sorted = summary_df.sort_values('Sharpe', ascending=False)
    print(sharpe_sorted.head(10).to_string(index=False))
    
    # Print Summary by Model
    print(f"\n{'='*120}")
    print("📈 AVERAGE PERFORMANCE BY MODEL")
    print(f"{'='*120}")
    model_summary = summary_df.groupby('Model').agg({
        'Return %': 'mean',
        'Sharpe': 'mean',
        'Win Rate %': 'mean',
        'Max DD %': 'mean'
    }).round(2).sort_values('Return %', ascending=False)
    print(model_summary.to_string())
    
    # Plot top 5 equity curves
    print(f"\n📊 Generating comparison plot...")
    top_5 = summary_df.head(5)
    
    plt.figure(figsize=(16, 10))
    
    for idx, row in top_5.iterrows():
        # Find the result with this config
        for r in all_results:
            if (r['model'] == row['Model'] and 
                r['max_positions'] == row['Max Pos'] and 
                r['take_profit'] == row['Take Profit'] and 
                r['stop_config'] == row['Stop']):
                
                label = f"{r['model']} | Pos:{r['max_positions']} | TP:{r['take_profit']} | {r['stop_config']} ({r['total_return']:+.1f}%)"
                plt.plot(r['equity_df'].index, r['equity_df']['Equity'], label=label, linewidth=2)
                break
    
    plt.axhline(y=INITIAL_CASH, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    plt.title('Top 5 Configurations - Equity Curves', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/grid_search_top5.png', dpi=150)
    print(f"✅ Plot saved to plots/grid_search_top5.png")
    
    print(f"\n✅ Full results saved to logs/grid_search_results.csv")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
