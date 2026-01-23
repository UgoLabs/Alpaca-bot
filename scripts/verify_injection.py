
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add root to path
sys.path.append(os.getcwd())

from src.data.pipeline import MultiModalDataPipeline

def verify_injection():
    print("🧪 Starting Data Injection Verification...")
    
    # 1. Initialize Pipeline
    pipeline = MultiModalDataPipeline(window_size=60)
    symbol = "AAPL"
    
    # 2. Load Initial Data (No Injection)
    print(f"   📂 Loading initial local data for {symbol}...")
    df_initial = pipeline._load_local_csv(symbol)
    
    if df_initial.empty:
        print(f"   ❌ Error: No local data found for {symbol}. Cannot verify.")
        return
        
    last_date_initial = df_initial.index[-1]
    print(f"      Initial Last Date: {last_date_initial}")
    
    # 3. Create Fake Live Candle (Future Date)
    # Ensure it's definitely in the future relative to the CSV
    fake_date = last_date_initial + timedelta(days=1)
    # Normalize to avoid time mismatch issues if CSV has times
    fake_date = fake_date.normalize()
    
    # If the CSV has specific time zone, we might need to handle that, 
    # but _load_local_csv does .normalize() in the injection logic, 
    # so we should provide a timestamp that resembles Alpaca's.
    
    print(f"   💉 Injecting Fake Candle for date: {fake_date}...")
    
    pipeline.live_daily_candles[symbol] = {
        'date': fake_date, # Timestamp
        'open': 200.0,
        'high': 210.0,
        'low': 190.0,
        'close': 205.0,
        'volume': 1234567
    }
    
    # 4. Clear Cache to force reload from disk + injection
    # The pipeline caches the result of _load_local_csv in self._csv_cache
    if symbol in pipeline._csv_cache:
        print("   🧹 Clearing internal cache to force re-processing...")
        del pipeline._csv_cache[symbol]
        
    # 5. Load Data Again (With Injection)
    df_injected = pipeline._load_local_csv(symbol)
    last_date_injected = df_injected.index[-1]
    last_close_injected = df_injected['Close'].iloc[-1]
    
    print(f"      Injected Last Date: {last_date_injected}")
    print(f"      Injected Last Close: {last_close_injected}")
    
    # 6. Verify
    is_date_match = (last_date_injected == fake_date)
    is_price_match = (last_close_injected == 205.0)
    
    if is_date_match and is_price_match:
        print("\n✅ VERIFICATION SUCCESS: Live candle successfully injected into historical data!")
    else:
        print("\n❌ VERIFICATION FAILED: Injected candle not found.")
        print(f"   Expected Date: {fake_date}, Found: {last_date_injected}")
        print(f"   Expected Close: 205.0, Found: {last_close_injected}")

if __name__ == "__main__":
    verify_injection()
