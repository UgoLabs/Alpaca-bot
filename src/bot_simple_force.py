"""
ULTRA SIMPLE DAY TRADER - Forces trades if ANY signal is positive
"""
import alpaca_trade_api as tradeapi
import time

API_KEY = "PKA7TFQVG5OB3YK6UEJ6ZFEGOH"
API_SECRET = "6ceJ8ZhknodD8iGM2NuMYTpxjr4BMgc5DaoD1xCagtbp"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)

SYMBOLS = ['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT']
PROFIT_TARGET = 0.01  # 1%
STOP_LOSS = 0.005  # 0.5%

print("ULTRA SIMPLE TRADER - Will buy top 3 symbols NOW")
print("=" * 60)

try:
    account = api.get_account()
    equity = float(account.equity)
    print(f"Equity: ${equity:,.2f}")
    
    # Buy top 3
    for symbol in SYMBOLS[:3]:
        try:
            # Get price
            trade = api.get_latest_trade(symbol, feed='iex')
            price = float(trade.price)
            
            # Calculate qty
            allocation = equity / 10  # 10% per trade
            qty = int(allocation / price)
            
            if qty > 0:
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                print(f"✓ BOUGHT {qty} {symbol} @ ${price:.2f}")
        except Exception as e:
            print(f"✗ Failed to buy {symbol}: {e}")
    
    print("\nNow monitoring for exits...")
    
    while True:
        try:
            positions = api.list_positions()
            
            for pos in positions:
                entry = float(pos.avg_entry_price)
                current = float(api.get_latest_trade(pos.symbol, feed='iex').price)
                pnl = (current - entry) / entry
                
                if pnl >= PROFIT_TARGET:
                    api.close_position(pos.symbol)
                    print(f"✓ SOLD {pos.symbol} +{pnl*100:.2f}% PROFIT")
                elif pnl <= -STOP_LOSS:
                    api.close_position(pos.symbol)
                    print(f"✗ SOLD {pos.symbol} {pnl*100:.2f}% STOP")
            
            time.sleep(30)  # Check every 30 sec
            
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

except Exception as e:
    print(f"Fatal error: {e}")
