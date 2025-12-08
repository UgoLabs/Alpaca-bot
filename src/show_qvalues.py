"""Print raw Q-values to diagnose the issue"""
from bot_standalone import PaperTrader, add_technical_indicators, normalize_state
import numpy as np
import torch

trader = PaperTrader('models/live_model.pth')

# Test one symbol in detail
symbol = 'NVDA'
print(f"\n{'='*60}")
print(f"Detailed Q-value analysis for {symbol}")
print(f"{'='*60}\n")

df = trader.get_bars(symbol)
if df is None:
    print("No data!")
else:
    df = add_technical_indicators(df)
    current_step = len(df) - 1
    
    # Show latest indicators
    latest = df.iloc[-1]
    print(f"Latest data ({df.index[-1].date()}):")
    print(f"  Close: ${latest['close']:.2f}")
    print(f"  RSI: {latest['rsi']:.1f}")
    print(f"  ADX: {latest['adx']:.1f}")
    print(f"  Regime: {latest['regime']}")
    print(f"  Trend Strength: {latest['trend_strength']}")
    print(f"  Momentum 5d: {latest['momentum_5d']*100:.2f}%")
    print(f"  SMA Cross: {latest['sma_cross']}")
    
    # Build state
    market_state = normalize_state(df, current_step)
    portfolio_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    state = np.concatenate((market_state, portfolio_state))
    
    print(f"\nState vector:")
    print(f"  Market state size: {len(market_state)}")
    print(f"  Total state size: {len(state)}")
    print(f"  First 5 features: {state[:5]}")
    print(f"  Last 10 features: {state[-10:]}")
    
    # Get Q-values
    trader.agent.model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = trader.agent.model(state_tensor).cpu().numpy()[0]
    
    print(f"\nRaw Q-values:")
    print(f"  HOLD: {q_values[0]:.6f}")
    print(f"  BUY:  {q_values[1]:.6f}")
    print(f"  SELL: {q_values[2]:.6f}")
    
    action = np.argmax(q_values)
    print(f"\nChosen action: {['HOLD', 'BUY', 'SELL'][action]}")
    
    # Analysis
    if q_values[0] > q_values[1] and q_values[0] > q_values[2]:
        diff_buy = q_values[0] - q_values[1]
        diff_sell = q_values[0] - q_values[2]
        print(f"\nWhy HOLD?")
        print(f"  HOLD beats BUY by: {diff_buy:.6f}")
        print(f"  HOLD beats SELL by: {diff_sell:.6f}")
        
        if abs(diff_buy) < 0.001:
            print(f"  ⚠️  Very close! Might trade with small changes")
        elif diff_buy > 1.0:
            print(f"  ✅ Strong preference for HOLD (model is confident)")
