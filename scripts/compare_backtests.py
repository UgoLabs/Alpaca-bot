import subprocess
import re
import sys

def run_backtest(model_path, scenario_name, atr_mult, profit_mult):
    print(f"\n🚀 Running: {scenario_name}")
    print(f"   Model: {model_path}")
    print(f"   Parameters: Stop={atr_mult}x ATR, Profit={profit_mult}x ATR")
    
    cmd = [
        sys.executable, "scripts/backtest_swing.py",
        model_path,
        "--use-trailing-stop",
        "--atr-mult", str(atr_mult),
        "--use-profit-take",
        "--profit-atr-mult", str(profit_mult),
        # "--visualize" # Skip visualization to save time
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        output = result.stdout
        
        # Parse Reward
        total_reward = "N/A"
        avg_reward = "N/A"
        
        # Look for "Total Reward (Sum): 123.4567"
        match_total = re.search(r"Total Reward \(Sum\): ([-\d\.]+)", output)
        if match_total:
            total_reward = float(match_total.group(1))
            
        return total_reward
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    models = [
        ("Ep 330", "models/swing_gen6_finetune_aggressive_update_ep330"),
        ("Ep 350", "models/swing_gen6_finetune_aggressive_update_ep350")
    ]
    
    scenarios = [
        ("Loose (User Pref)", 6.0, 3.0),
        ("Tight (Sniper)", 3.0, 4.0)
    ]
    
    results = []
    
    print("📊 STARTING BACKTEST COMPARISON")
    print("=" * 40)
    
    for model_name, model_path in models:
        for scen_name, atr, profit in scenarios:
            full_name = f"{model_name} - {scen_name}"
            reward = run_backtest(model_path, full_name, atr, profit)
            results.append({
                "name": full_name,
                "reward": reward
            })
            print(f"   👉 Result: Total Reward = {reward}")
    
    print("\n\n🏆 FINAL LEADERBOARD")
    print("=" * 40)
    # Sort by reward descending (handle N/A)
    results.sort(key=lambda x: x['reward'] if isinstance(x['reward'], float) else -999999, reverse=True)
    
    for i, res in enumerate(results):
        print(f"{i+1}. {res['name']:<30} | Reward: {res['reward']}")

if __name__ == "__main__":
    main()
