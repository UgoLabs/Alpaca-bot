import os
import subprocess
import pandas as pd
import glob
import re

def run_batch_backtest():
    model_dir = "models"
    # Find all sharpe_gen1 prefix patterns
    pattern = os.path.join(model_dir, "sharpe_gen1_ep*_balanced.pth")
    files = glob.glob(pattern)
    
    prefixes = []
    for f in files:
        # Extract prefix: models/sharpe_gen1_epX
        match = re.search(r"(sharpe_gen1_ep\d+)", f)
        if match:
            prefixes.append(match.group(1))
            
    # Sort by episode number
    def get_ep(p):
        m = re.search(r"ep(\d+)", p)
        return int(m.group(1)) if m else 0
        
    prefixes = sorted(list(set(prefixes)), key=get_ep)
    
    # Filter for ep 1 to 70 as requested
    prefixes = [p for p in prefixes if get_ep(p) <= 70]
    
    results = []
    python_exe = r"C:\Users\okwum\Alpaca-bot\.venv\Scripts\python.exe"
    
    print(f"📊 Starting Batch Backtest for {len(prefixes)} checkpoints...")
    
    for prefix in prefixes:
        full_prefix = f"models/{prefix}"
        print(f"\n--- Testing {prefix} ---")
        
        # Run the backtest script
        # We use a smaller test count for speed in batch mode
        cmd = [
            python_exe, 
            "scripts/backtest_swing.py", 
            full_prefix, 
            "--generalization", 
            "--test-count", "10", 
            "--period", "1y"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            output = result.stdout
            
            # Parse metrics from output
            # Look for "Average Reward per Symbol: X"
            reward_match = re.search(r"Average Reward per Symbol: ([\d.-]+)", output)
            reward = float(reward_match.group(1)) if reward_match else 0.0
            
            # Simple Action distribution parsing
            hold_match = re.search(r"Hold: ([\d.]+)%", output)
            buy_match = re.search(r"Buy:\s+([\d.]+)%", output)
            sell_match = re.search(r"Sell:\s+([\d.]+)%", output)
            
            hold_pct = float(hold_match.group(1)) if hold_match else 0.0
            buy_pct = float(buy_match.group(1)) if buy_match else 0.0
            sell_pct = float(sell_match.group(1)) if sell_match else 0.0
            
            summary = {
                "checkpoint": prefix,
                "episode": get_ep(prefix),
                "avg_reward": reward,
                "hold_pct": hold_pct,
                "buy_pct": buy_pct,
                "sell_pct": sell_pct
            }
            results.append(summary)
            print(f"✅ Reward: {reward:.2f} | Buy: {buy_pct}% | Sell: {sell_pct}%")
            
        except Exception as e:
            print(f"❌ Error testing {prefix}: {e}")
            
    # Save results
    if results:
        df = pd.DataFrame(results)
        output_file = "logs/batch_backtest_sharpe_gen1.csv"
        df.to_csv(output_file, index=False)
        print(f"\n📁 Batch Backtest Complete! Results saved to {output_file}")
        print(df)

if __name__ == "__main__":
    run_batch_backtest()
