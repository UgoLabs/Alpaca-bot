import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_comparison_backtest import run_backtest

def main():
    INITIAL_CASH = 10000.0
    END_DATE = "2026-01-28"
    
    # 1. Standard Testing (Unbound)
    models_unbound = [
        ("Sharpe_Gen2_Ep50", "models/sharpe_gen2_ep50"),
        ("Sharpe_Gen2_Ep30", "models/sharpe_gen2_ep30"),
    ]
    
    # 2. Wrapped Testing (Hard Stop Loss)
    # We test the aggressive candidates with a safety net
    models_wrapped = []
    
    all_results = []
    
    # --- PHASE 1: UNBOUND (Pure Agent) ---
    MAX_POS = 100
    SL_UNBOUND = 1000.0
    TP_UNBOUND = 1000.0

    periods = [
        ("6 Months", "2025-07-28"),
        ("1 Year",   "2025-01-28"),
        ("2 Years",  "2024-01-28"),
    ]

    for model_name, model_path in models_unbound:
        # Check if ensemble files exist (EnsembleAgent.load adds _balanced.pth etc.)
        balanced_path = f"{model_path}_balanced.pth"
        if not os.path.exists(balanced_path):
            print(f"❌ Model not found: {balanced_path}")
            continue

        print(f"\n{'#'*80}")
        print(f"🧪 TESTING MODEL: {model_name} (UNBOUND)")
        print(f"🔧 Settings: MaxPos={MAX_POS}, SL=DISABLED, TP=DISABLED")
        print(f"{'#'*80}")
        
        for period_name, start_date in periods:
            try:
                res = run_backtest(
                    model_path=model_path,
                    model_name=f"{model_name}_{period_name}",
                    start_date=start_date,
                    end_date=END_DATE,
                    initial_cash=INITIAL_CASH,
                    max_positions=MAX_POS,
                    stop_loss_pct=SL_UNBOUND,
                    take_profit_pct=TP_UNBOUND,
                )
                if res and res['total_trades'] > 0:
                    res['period'] = period_name
                    res['model'] = model_name
                    res['type'] = "Unbound"
                    all_results.append(res)
            except Exception as e:
                print(f"⚠️ Failed to run backtest for {period_name}: {e}")

    # --- PHASE 2: WRAPPED (Hard Stop Loss) ---
    for model_name, model_path, sl_val in models_wrapped:
        if not os.path.exists(model_path):
            continue

        print(f"\n{'#'*80}")
        print(f"🛡️  TESTING WRAPPED MODEL: {model_name}")
        print(f"🔧 Settings: Hard SL={sl_val*100}%")
        print(f"{'#'*80}")

        for period_name, start_date in periods:
             try:
                res = run_backtest(
                    model_path=model_path,
                    model_name=f"{model_name}_{period_name}",
                    start_date=start_date,
                    end_date=END_DATE,
                    initial_cash=INITIAL_CASH,
                    max_positions=MAX_POS,
                    stop_loss_pct=sl_val,
                    take_profit_pct=TP_UNBOUND, # Keep TP disabled to let winners run
                )
                if res and res['total_trades'] > 0:
                    res['period'] = period_name
                    res['model'] = model_name
                    res['type'] = f"SL {sl_val*100}%"
                    all_results.append(res)
             except Exception as e:
                print(f"⚠️ Failed to run backtest: {e}")

    # Comparative Results Table
    if all_results:
        print(f"\n{'='*120}")
        print(f"📊 MODEL COMPARISON: UNBOUND vs WRAPPED")
        print(f"{'='*120}")
        print(f"{'Model':<15} {'Type':<12} {'Period':<10} {'Return':<10} {'Final $':<12} {'Max DD':<10} {'Trades':<8} {'Win Rate':<10}")
        print(f"{'-'*120}")
        
        # Sort by Return (descending)
        all_results.sort(key=lambda x: x['total_return'], reverse=True)

        for res in all_results:
            model = res['model']
            m_type = res['type']
            period = res['period']
            ret = res['total_return']
            final = res['final_equity']
            dd = res['max_drawdown']
            trades = res['total_trades']
            wr = res['win_rate']
            
            print(f"{model:<15} {m_type:<12} {period:<10} {ret:>+7.2f}%   ${final:<11,.2f} {dd:>7.2f}%   {trades:<8} {wr:>7.1f}%")
        print(f"{'='*120}")

if __name__ == "__main__":
    main()
