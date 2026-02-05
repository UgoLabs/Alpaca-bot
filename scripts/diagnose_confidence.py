import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Set up paths
ROOT_DIR = Path("C:/Users/okwum/Alpaca-bot")
sys.path.append(str(ROOT_DIR))

from src.agents.ensemble_agent import EnsembleAgent
from src.data.pipeline import MultiModalDataPipeline
from config.settings import SwingTraderCreds, SwingTraderConfig, SWING_MODEL_PATH

# Load .env
load_dotenv(ROOT_DIR / '.env')

def diagnose_symbol(symbol):
    print(f"\n DIAGNOSING {symbol}...")
    
    # 1. Initialize Pipeline
    creds = SwingTraderCreds
    pipeline = MultiModalDataPipeline(window_size=60, creds=creds, feed='sip')
    
    # 2. Fetch Data
    print("    Fetching live data...")
    data = pipeline.fetch_and_process(symbol, timeframe='1D')
    
    if data is None:
        print("    Failed to fetch data.")
        return

    ts_state, text_ids, text_mask, current_price = data
    ts_state = ts_state.reshape(60, 11) # Reshape
    
    # 3. Load Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = EnsembleAgent(time_series_dim=11, vision_channels=11, action_dim=3, device=device)
    
    # Load Model
    model_path = str(SWING_MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"    Model not found: {model_path}")
        return
        
    print(f"    Loading Model: {os.path.basename(model_path)}")
    agent.load(model_path.replace("_balanced.pth", ""))
    
    # 4. Run Inference (Get RAW probabilities)
    print("    Running Inference...")
    
    # Manually run the forward pass to see logits if possible, or use act() with return_q
    # The act method returns (action, confidence). 
    # Let's peek under the hood if we can, or just trust the output.
    
    action, confidence = agent.act(ts_state, text_ids, text_mask, eval_mode=True, return_q=True)
    
    actions = ["HOLD", "BUY", "SELL"]
    print(f"\n    DECISION: {actions[action]}")
    print(f"    CONFIDENCE: {confidence:.4f}")
    print(f"    Current Price: ${current_price:.2f}")
    
    if action == 2: # SELL
        if confidence > 0.60:
            print("    Verdict: Model WANTS to sell and PASSES threshold (>0.60).")
        else:
            print(f"    Verdict: Model WANTS to sell but FAILS threshold ({confidence:.4f} < 0.60).")
    elif action == 0: # HOLD
        print("    Verdict: Model says HOLD. (It does not see a reason to exit yet).")
    elif action == 1: # BUY
        print("    Verdict: Model says BUY. (It actually likes this stock).")

if __name__ == "__main__":
    # Test the losers
    for sym in ["QBTS", "UMC", "MRNA"]:
        diagnose_symbol(sym)
