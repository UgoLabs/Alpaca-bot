"""Debug script to inspect Q-values from Sharpe model."""
import os
import sys
import torch
import numpy as np
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ensemble_agent import EnsembleAgent
from src.core.indicators import add_technical_indicators

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model - test sharpe_gen3_ep1 with 25 features
model_path = "models/sharpe_gen3_ep1"
agent = EnsembleAgent(time_series_dim=25, vision_channels=25, action_dim=3, device=device)
agent.load(model_path)
agent.set_eval()

# Load one symbol for testing
test_file = glob.glob("data/historical_swing/*_1D.csv")[0]
df = pd.read_csv(test_file)

# Standardize columns
df.columns = [c.lower() for c in df.columns]
col_map = {
    'o': 'Open', 'open': 'Open',
    'h': 'High', 'high': 'High',
    'l': 'Low', 'low': 'Low',
    'c': 'Close', 'close': 'Close',
    'v': 'Volume', 'volume': 'Volume'
}
for k, v in col_map.items():
    if k in df.columns:
        df[v] = df[k]

df = add_technical_indicators(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Feature columns - 25 for sharpe models
feature_cols = [
    'sma_10', 'sma_20', 'sma_50', 'sma_200', 'atr', 'bb_width', 'bb_upper', 'bb_lower',
    'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_diff', 'adx', 'rsi',
    'stoch_k', 'stoch_d', 'williams_r', 'roc', 'bb_pband', 'volume_sma',
    'volume_ratio', 'obv', 'mfi', 'price_vs_sma20'
]

# Normalize
features = df[feature_cols].values
mean = np.mean(features, axis=0)
std = np.std(features, axis=0) + 1e-8
norm_features = (features - mean) / std

# Get 60-bar window from end
window_size = 60
if len(norm_features) >= window_size:
    window = norm_features[-window_size:]
    
    # Make tensor
    batch = torch.FloatTensor(window).unsqueeze(0).to(device)
    dummy_text = torch.zeros((1, 64), dtype=torch.long).to(device)
    
    # Get Q-values from each sub-agent
    print(f"\n📊 Q-Values from {os.path.basename(test_file)}:")
    print("=" * 50)
    
    for i, (name, sub_agent) in enumerate(zip(['Aggressive', 'Conservative', 'Balanced'], agent.agents)):
        with torch.no_grad():
            q_values = sub_agent.policy_net(batch, dummy_text, dummy_text)
            q_np = q_values.cpu().numpy()[0]
            action = q_values.argmax().item()
            
            print(f"\n{name}:")
            print(f"  Q[HOLD]=  {q_np[0]:.6f}")
            print(f"  Q[BUY]=   {q_np[1]:.6f}")
            print(f"  Q[SELL]=  {q_np[2]:.6f}")
            print(f"  Action:   {['HOLD', 'BUY', 'SELL'][action]}")
            
    # Now test ensemble action
    action, conf = agent.act(batch, dummy_text, dummy_text, return_q=True)
    print(f"\n🎯 ENSEMBLE: {['HOLD', 'BUY', 'SELL'][action]} (confidence={conf:.4f})")
    
else:
    print(f"Not enough data: {len(norm_features)} rows")
