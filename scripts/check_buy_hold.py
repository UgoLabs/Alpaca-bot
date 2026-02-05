import yfinance as yf
from datetime import datetime

start = "2024-01-28"
end = "2026-01-28"

print(f"📊 Benchmarking Buy & Hold (SPY) vs Your Bot (+52%)...")
print(f"📅 Period: {start} to {end}")

try:
    spy = yf.download("SPY", start=start, end=end, progress=False)
    if len(spy) > 0:
        start_price = spy['Close'].iloc[0].item()
        end_price = spy['Close'].iloc[-1].item()
        
        return_pct = ((end_price - start_price) / start_price) * 100
        
        print(f"\n📈 S&P 500 (SPY) Result:")
        print(f"   - Start Price: ${start_price:.2f}")
        print(f"   - End Price:   ${end_price:.2f}")
        print(f"   - Total Return: {return_pct:+.2f}%")
        
        bot_return = 52.0  # From your Swing380 results
        diff = bot_return - return_pct
        
        print(f"\n🏆 VERDICT:")
        if diff > 0:
            print(f"   ✅ Your Bot BEAT the market by {diff:+.2f}%!")
        else:
            print(f"   ❌ Buy & Hold BEAT your bot by {abs(diff):.2f}%.")
            
    else:
        print("❌ Could not download SPY data.")

except Exception as e:
    print(f"Error: {e}")
