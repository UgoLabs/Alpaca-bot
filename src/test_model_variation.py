from bot_standalone import PaperTrader
import random

print("Testing if model outputs vary...")
trader = PaperTrader('models/live_model.pth')

# Test 5 random symbols
test_symbols = random.sample(trader.symbols, min(5, len(trader.symbols)))

results = {}
for symbol in test_symbols:
    try:
        action = trader.get_model_action(symbol)
        results[symbol] = action
        print(f"{symbol:6s} → {['HOLD', 'BUY', 'SELL'][action]}")
    except Exception as e:
        print(f"{symbol:6s} → ERROR: {e}")

# Check if all same
actions = list(results.values())
if len(set(actions)) == 1:
    print(f"\n⚠️  WARNING: All {len(actions)} symbols returned the same action: {['HOLD', 'BUY', 'SELL'][actions[0]]}")
    print("This could indicate:")
    print("  - Model is stuck/broken")
    print("  - Market conditions are uniformly unfavorable")
    print("  - Bug in state construction")
else:
    print(f"\n✅ Model is outputting {len(set(actions))} different actions - working correctly!")
