
import sys
import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Force UTF-8 for Windows
if sys.platform == 'win32':
    # type: ignore
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
# Override Window Size for Swing Trading
TrainingConfig.WINDOW_SIZE = 60

from src.environments.vector_env import VectorizedTradingEnv  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.core.indicators import add_technical_indicators  # noqa: E402
from scripts.backtest_swing import load_swing_data, run_backtest # Reuse existing functions


def run_comparative_backtest():
    """
    Runs backtests for Gen 7 models (Episodes 365-390) 
    using the PRECISE parameters from our verified profitable config.
    """
    
    # === CONFIGURATION (MATCHING LIVE SETTINGS) ===
    MODELS_TO_TEST = [
        "models/swing_gen7_refined_ep365",
        "models/swing_gen7_refined_ep370",
        "models/swing_gen7_refined_ep375",
        "models/swing_gen7_refined_ep380",
        "models/swing_gen7_refined_ep385",
        "models/swing_gen7_refined_ep390"
    ]
    
    # 1. HOLD-OUT TEST PERIOD
    TEST_START = "2025-08-01"
    TEST_END = "2026-01-23" # Today

    # Scenarios to test: (Description, Stop Mul, Profit Mul)
    SCENARIOS = [
        ("Winner (Stop 3.0, Profit 5.0, 10% Size)", 3.0, 5.0, 0.10)
    ]

    print("="*60)
    print(f"🚀 RUNNING GEN 7 REFINED CHECKPOINT ANALYSIS")
    print(f"📅 Period: {TEST_START} to {TEST_END}")
    print("="*60)

    for label, stop_mult, profit_mult, pos_pct in SCENARIOS:
        print(f"\n👉 Testing Scenario: {label}")
        
        # 2. MATCHING LIVE PARAMS
        PARAMS = {
            "use_trailing_stop": True,
            "atr_period": 14,
            "atr_mult": stop_mult,        
            "use_profit_take": True,
            "profit_atr_mult": profit_mult,  
            "position_pct": pos_pct,     
            "test_start_date": TEST_START,
            "test_end_date": TEST_END,
            "visualize": False,
        }
        
        for model_path in MODELS_TO_TEST:
            # print(f"👉 Testing: {os.path.basename(model_path)}")
            run_backtest(model_path, **PARAMS)

if __name__ == "__main__":
    run_comparative_backtest()
