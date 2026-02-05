import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
# Override Window Size for Swing Trading (Must match training)
TrainingConfig.WINDOW_SIZE = 60

from src.agents.ensemble_agent import EnsembleAgent
from scripts.backtest_swing import load_swing_data  # Reuse data loader

class PortfolioSimulator:
    def __init__(self, tickers, prices, data, model, device, 
                 start_cash=10000.0, max_positions=10, 
                 stop_loss_mult=4.0, profit_take_mult=6.0,
                 transaction_cost=0.001): # 0.1% per trade
        
        self.tickers = tickers
        self.prices = prices # [N, T]
        self.data = data     # [N, T, F]
        self.model = model
        self.device = device
        
        self.start_cash = start_cash
        self.cash = start_cash
        self.max_positions = max_positions
        self.stop_mult = stop_loss_mult
        self.profit_mult = profit_take_mult
        self.tc = transaction_cost
        
        # Portfolio State
        # {ticker_index: {'entry_price': float, 'qty': float, 'stop': float}}
        self.positions = {} 
        self.equity_curve = []
        self.detailed_log = []
        self.trade_history = []

    def run(self):
        num_tickers, num_steps, num_features = self.data.shape
        
        # Dummy Text for Model (Text disabled for pure Swing Backtest)
        seq_len = 64
        dummy_text_ids = torch.zeros((num_tickers, seq_len), dtype=torch.long).to(self.device)
        dummy_text_mask = torch.ones((num_tickers, seq_len), dtype=torch.long).to(self.device)

        # Pre-calculate ATR for all steps (Approximate using cached data)
        # Using feature column 5 (ATR/Price) assuming normalization matches state.py
        # state.py: norm_atr = window['atr'].values / window['Close'].values
        # But features are normalized (Z-Score). We can't easily recover raw ATR.
        # We need to approximate ATR from the Z-scored feature or assume a % volatility.
        # BETTER: Recompute ATR on the fly? No, too slow.
        # Workaround: Use a % Stop Loss based on volatility proxy or just fixed % if raw ATR lost.
        # Wait, backtest_swing.py passes 'atr_values' tensor! I should use that.
        
        pass 

def run_simulation(model_path, data_dir="data/historical_swing", 
                   positions_list=[10, 15], 
                   stop_list=[2.5, 3.0], 
                   profit_list=[5.0, 5.5, 6.0],
                   initial_equity=10000.0, start_date=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Grid Search Simulation on {device}...")
    
    # 1. Load Data
    data_tensor, price_tensor, stop_tensor, atr_tensor, tickers = load_swing_data(
        data_dir=data_dir, 
        test_start_date=start_date
    )
    
    if data_tensor is None:
        print("❌ Failed to load data.")
        return

    data_tensor = data_tensor.to(device)
    price_tensor = price_tensor.to(device)
    atr_tensor = atr_tensor.to(device)
    
    # 2. Load Model
    num_features = data_tensor.shape[2]
    agent = EnsembleAgent(num_features, num_features, 3, device=device)
    
    load_path = model_path
    for suffix in ['_balanced.pth', '_aggressive.pth', '_conservative.pth']:
        if load_path.endswith(suffix):
            load_path = load_path.replace(suffix, "")
            break
            
    agent.load(load_path)
    for sa in agent.agents:
        sa.epsilon = 0.0
        sa.policy_net.eval()
        
    print(f"✅ Data & Model Loaded. (Tickers: {len(tickers)}, Days: {data_tensor.shape[1]})")

    # 3. Simulate Scenarios
    results = {}
    
    # Create grid
    scenarios = list(itertools.product(positions_list, stop_list, profit_list))
    print(f"📋 Running {len(scenarios)} scenarios...")
    
    print("🧠 Pre-calculating Model Inference...")
    N, T, F = data_tensor.shape
    all_actions = np.zeros((N, T), dtype=int)
    all_confs = np.zeros((N, T), dtype=float)
    
    # Dummy text
    dummy_ids = torch.zeros((N, 64), dtype=torch.long).to(device)
    dummy_mask = torch.ones((N, 64), dtype=torch.long).to(device)

    # Batch inference
    for t in tqdm(range(60, T), desc="Inference"):
        start = t - 60
        end = t
        state_slice = data_tensor[:, start:end, :]
        
        with torch.no_grad():
            actions, confs = agent.act(state_slice, dummy_ids, dummy_mask, eval_mode=True, return_q=True)
            
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(confs, torch.Tensor):
            confs = confs.cpu().numpy()

        all_actions[:, t] = actions
        all_confs[:, t] = confs
        
    cpu_prices = price_tensor.cpu().numpy()
    cpu_atrs = atr_tensor.cpu().numpy()
    
    print("\n🏎️  Running Strategy Loops...")
    
    for (max_pos, stop_mult, profit_mult) in scenarios:
        key = f"Pos:{max_pos}_Stop:{stop_mult}_Prof:{profit_mult}"
        
        cash = initial_equity
        equity_curve = []
        positions = {} 
        trades_count = 0
        total_volume = 0
        
        for t in range(60, T):
            curr_prices = cpu_prices[:, t]
            curr_atrs = cpu_atrs[:, t]
            day_actions = all_actions[:, t]
            day_confs = all_confs[:, t]
            valid = (curr_prices > 0)
            
            # A) Check Exits
            to_remove = []
            for idx, pos in positions.items():
                price = curr_prices[idx]
                atr = curr_atrs[idx]
                
                if price <= 0: continue 
                
                if price > pos['highest']:
                    pos['highest'] = price
                    
                stop_price = pos['highest'] - (atr * stop_mult)
                profit_price = pos['entry'] + (atr * profit_mult)
                
                model_sell = (day_actions[idx] == 2) and (day_confs[idx] > 0.40)
                
                reason = None
                if price < stop_price: reason = "Stop Loss"
                elif price > profit_price: reason = "Profit Take"
                elif model_sell: reason = "Model Sell"
                
                if reason:
                    proceeds = pos['qty'] * price
                    cash += proceeds
                    trades_count += 1
                    total_volume += proceeds
                    to_remove.append(idx)
                    
            for idx in to_remove:
                del positions[idx]
                
            # B) Check Entries
            candidates = []
            for idx in range(N):
                if not valid[idx]: continue
                if idx in positions: continue
                
                if day_actions[idx] == 1 and day_confs[idx] > 0.40:
                    candidates.append((idx, day_confs[idx]))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            open_slots = max_pos - len(positions)
            equity = cash + sum(p['qty'] * cpu_prices[i, t] for i, p in positions.items())
            
            # Avoid division by zero
            target_size = equity / max_pos if max_pos > 0 else 0
            
            for idx, conf in candidates:
                if open_slots <= 0: break
                if cash < target_size: break
                if target_size <= 0: break
                
                price = curr_prices[idx]
                if price <= 0: continue

                qty = target_size / price
                cost = qty * price
                
                if cost > cash: 
                    qty = cash / price
                    cost = cash
                
                cash -= cost
                positions[idx] = {
                    'entry': price,
                    'highest': price,
                    'qty': qty
                }
                open_slots -= 1
                trades_count += 1
                total_volume += cost
            
            current_equity = cash + sum(p['qty'] * cpu_prices[i, t] for i, p in positions.items())
            equity_curve.append(current_equity)
            
        final_equity = equity_curve[-1]
        ret = (final_equity - initial_equity) / initial_equity * 100
        
        results[key] = {
            'param_pos': max_pos,
            'param_stop': stop_mult,
            'param_prof': profit_mult,
            'final_equity': final_equity,
            'return': ret,
            'trades': trades_count,
            'vol': total_volume,
            'curve': equity_curve
        }

    print("\n📊 GRID SEARCH RESULTS (Start: $10,000)")
    print("="*100)
    print(f"{'POSITIONS':<10} | {'STOP':<6} | {'PROFIT':<8} | {'FINAL Equity':<15} | {'RETURN':<10} | {'TRADES':<10}")
    print("-" * 100)
    
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]['return'], reverse=True)
    
    for k in sorted_keys:
        res = results[k]
        print(f"{res['param_pos']:<10} | {res['param_stop']:<6} | {res['param_prof']:<8} | ${res['final_equity']:.2f}       | {res['return']:+.1f}%    | {res['trades']}")
    print("="*100)
    
    # Plot top 3
    plt.figure(figsize=(12, 7))
    for k in sorted_keys[:3]:
        res = results[k]
        plt.plot(res['curve'], label=f"Pos:{res['param_pos']} S:{res['param_stop']} P:{res['param_prof']} ({res['return']:.1f}%)")
        
    plt.title("Top 3 Configurations")
    plt.xlabel("Days")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/simulation_grid_results.png")
    print("💾 Plot saved to plots/simulation_grid_results.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/swing_gen7_refined_ep380_balanced.pth")
    parser.add_argument("--start_date", type=str, default=None, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    run_simulation(args.model, start_date=args.start_date)
