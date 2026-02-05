
import subprocess
import sys
import os

# Configuration
start_ep = 50
end_ep = 50
model_prefix = "models/sharpe_gen4_ep"
script_path = "scripts/backtest_with_deposits.py"
python_executable = sys.executable

print(f"🚀 Starting Batch Backtest for Sharpe Gen 4 (Ep {start_ep}-{end_ep})")

for i in range(start_ep, end_ep + 1):
    model_path = f"{model_prefix}{i}"
    # Check for the balanced model file
    if not os.path.exists(f"{model_path}_balanced.pth"):
        print(f"⏩ Skipping Ep {i} (File not found: {model_path}_balanced.pth)")
        continue

    print(f"\n▶️ Running Backtest for Episode {i}...")
    try:
        subprocess.run([python_executable, script_path, model_path], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Backtest failed for Episode {i}")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        break
