import os
import glob
import subprocess
import re
import sys

def main():
    models_dir = "models"
    # Pattern to match: swing_gen6_finetune_ep<Num>_*.pth or swing_gen6_finetune_aggressive_update_ep<Num>_*.pth
    # We want to extract the "prefix" that can be passed to backtest_swing.py
    # backtest_swing.py expects a path that does NOT have the _balanced/_aggressive suffix.
    
    files = glob.glob(os.path.join(models_dir, "*.pth"))
    
    unique_prefixes = set()
    
    for f in files:
        basename = os.path.basename(f)
        
        # Regex to find ep number and the prefix
        # Example 1: swing_gen6_finetune_ep550_aggressive.pth -> Prefix: swing_gen6_finetune_ep550, Ep: 550
        # Example 2: swing_gen6_finetune_aggressive_update_ep340_balanced.pth -> Prefix: swing_gen6_finetune_aggressive_update_ep340, Ep: 340
        
        match = re.search(r'(.*_ep(\d+))_(aggressive|balanced|conservative)\.pth$', basename)
        if match:
            prefix = match.group(1)
            ep_num = int(match.group(2))
            
            # Backtest specific episodes:
            # User request: 380, 895, 900, 905
            if ep_num in [380, 895, 900, 905]:
                full_prefix_path = os.path.join(models_dir, prefix).replace("\\", "/")
                # Ensure we don't accidentally add duplicates if logic changes
                unique_prefixes.add((ep_num, full_prefix_path))

    # Sort by episode number
    sorted_prefixes = sorted(list(unique_prefixes), key=lambda x: x[0])
    
    if not sorted_prefixes:
        print("No models found with episode >= 380.")
        return

    print(f"Found {len(sorted_prefixes)} models to backtest.")
    
    results = []

    for ep, prefix_path in sorted_prefixes:
        print(f"\n{'='*50}")
        print(f"Testing Episode {ep}: {prefix_path}")
        print(f"{'='*50}")
        
        # Updated to "Gen 7 Refined" parameters + Hold-Out Period
        cmd = [
            sys.executable, "scripts/backtest_swing.py",
            prefix_path,
            "--use-trailing-stop",
            "--atr-mult", "3.0",
            "--use-profit-take",
            "--profit-atr-mult", "5.0",
            "--position-pct", "0.1",
            "--test-start-date", "2025-08-01",
            "--test-end-date", "2026-01-23"
        ]
        
        # Run command and capture output
        try:
            # We run without visualize to speed things up
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding='utf-8',
                bufsize=1
            )
            
            output_lines = []
            if process.stdout:
                for line in process.stdout:
                    print(line, end='') # Stream to console
                    output_lines.append(line)
            
            process.wait()
            output = "".join(output_lines)
            
            # Simple parsing of rewards for summary
            total_reward = "N/A"
            buy_rate = "N/A"
            
            for line in output.splitlines():
                if "Total Reward (Sum):" in line:
                    total_reward = line.split(":")[-1].strip()
                if "Buy:" in line and "Sell:" in line: # Action Distribution line
                    # Example: Action Distribution: Hold: 31.9% | Buy: 65.7% | Sell: 2.4%
                    parts = line.split("|")
                    for p in parts:
                        if "Buy:" in p:
                            buy_rate = p.split(":")[-1].strip()
            
            # print(output) # Already streamed
            results.append({
                "Episode": ep,
                "Model": os.path.basename(prefix_path),
                "Reward": total_reward,
                "Buy Rate": buy_rate
            })
            
        except Exception as e:
            print(f"Error running backtest for {prefix_path}: {e}")

    print("\n\n" + "="*60)
    print("BATCH BACKTEST SUMMARY")
    print("="*60)
    print(f"{'Episode':<10} | {'Reward':<15} | {'Buy Rate':<15} | {'Model':<30}")
    print("-" * 75)
    for res in results:
        print(f"{res['Episode']:<10} | {res['Reward']:<15} | {res['Buy Rate']:<15} | {res['Model']:<30}")
    print("="*60)

if __name__ == "__main__":
    main()
