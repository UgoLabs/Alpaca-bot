import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import run_backtest from the script
from scripts.backtest_swing import run_backtest

def run_ab_test():
    """
    Head-to-Head Backtest:
    Config A: Ep 380 (Stop 4.0 / Profit 6.0)
    Config B: Ep 380 (Stop 3.0 / Profit 5.0)
    Date Range: Last 6 Months
    """
    
    # Define Date Range (Last 6 months from Jan 26, 2026)
    end_date = datetime(2026, 1, 26)
    start_date = end_date - timedelta(days=180) # Approx 6 months
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"⚔️ Starting A/B Test: {start_str} to {end_str}")
    
    model_path = "models/swing_gen7_refined_ep380"
    
    # -------------------------------------------------------------------------
    # CONFIG A: Current Champion (4.0 / 6.0)
    # -------------------------------------------------------------------------
    print("\n🟢 Running CONFIG A (Stop 4.0 / Profit 6.0)...")
    
    metrics_a = run_backtest(
        model_path=model_path,
        test_generalization=False, # Use existing historical_swing data
        use_trailing_stop=True,
        atr_mult=4.0,
        use_profit_take=True,
        profit_atr_mult=6.0,
        test_start_date=start_str,
        test_end_date=end_str,
        visualize=False
    )
    
    # -------------------------------------------------------------------------
    # CONFIG B: Challenger (3.0 / 5.0)
    # -------------------------------------------------------------------------
    print("\n🔵 Running CONFIG B (Stop 3.0 / Profit 5.0)...")
    
    metrics_b = run_backtest(
        model_path=model_path,
        test_generalization=False,
        use_trailing_stop=True,
        atr_mult=3.0,
        use_profit_take=True,
        profit_atr_mult=5.0,
        test_start_date=start_str,
        test_end_date=end_str,
        visualize=False
    )
    
    if not metrics_a or not metrics_b:
        print("❌ A/B Test Failed: One or both runs returned no metrics.")
        return

    # -------------------------------------------------------------------------
    # COMPARISON REPORT
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print(f"{'METRIC':<25} | {'CONFIG A (4/6)':<20} | {'CONFIG B (3/5)':<20} | {'DIFF'}")
    print("-" * 80)
    
    keys = ['total_reward', 'avg_reward_per_symbol', 'win_rate', 'trades']
    
    for k in keys:
        val_a = metrics_a.get(k, 0.0)
        val_b = metrics_b.get(k, 0.0)
        diff = val_a - val_b
        
        # Formatting
        if k == 'win_rate':
            txt_a = f"{val_a:.2%}"
            txt_b = f"{val_b:.2%}"
            txt_diff = f"{diff:.2%}"
        elif k == 'trades':
            txt_a = f"{int(val_a)}"
            txt_b = f"{int(val_b)}"
            txt_diff = f"{int(diff)}"
        else:
            txt_a = f"{val_a:.4f}"
            txt_b = f"{val_b:.4f}"
            txt_diff = f"{diff:.4f}"
            
        print(f"{k.replace('_', ' ').title():<25} | {txt_a:<20} | {txt_b:<20} | {txt_diff}")
        
    print("="*80)
    
    # Conclusion
    if metrics_a['avg_reward_per_symbol'] > metrics_b['avg_reward_per_symbol']:
        print("\n🏆 WINNER: Config A (Stop 4.0 / Profit 6.0) is superior.")
    else:
        print("\n🏆 WINNER: Config B (Stop 3.0 / Profit 5.0) is superior.")

if __name__ == "__main__":
    run_ab_test()
