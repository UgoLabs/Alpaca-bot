"""
Simplified Paper Trading Deployment Script
Uses the same imports as parallel_train.py which works successfully
"""
import os
import sys

# Suppress the alpha_vantage import error
import warnings
warnings.filterwarnings('ignore')

# Try importing, catch the alpha_vantage error and continue anyway
try:
    from paper_trade import PaperTrader
except ModuleNotFoundError as e:
    if 'alpha_vantage' in str(e):
        print(f"Warning: {e}")
        print("Attempting to continue without alpha_vantage...")
        # Mock the module
        sys.modules['alpha_vantage'] = type(sys)('alpha_vantage')
        sys.modules['alpha_vantage.sectorperformance'] = type(sys)('sectorperformance')
        sys.modules['alpha_vantage.timeseries'] = type(sys)('timeseries')
        from paper_trade import PaperTrader
    else:
        raise

if __name__ == "__main__":
    print("Starting Paper Trading Bot...")
    print("="*60)
    
    trader = PaperTrader(model_path='models/live_model.pth')
    
    # Run once to test, then you can run in loop
    print("\nRunning initial scan...")
    trader.run_once()
    
    print("\n" + "="*60)
    print("Initial scan complete!")
    print("To run continuously, use: trader.run_loop(interval_minutes=15)")
