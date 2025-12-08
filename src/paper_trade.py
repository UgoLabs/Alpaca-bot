"""
Paper Trading Script for Alpaca
Runs the trained DQN model against Alpaca's paper trading API.
Includes position sizing, risk management, and trade logging.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from datetime import datetime, timedelta
from utils import add_technical_indicators, normalize_state, get_state_size, detect_market_regime
from swing_model import DuelingDQN

# Load environment variables
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')

# Trading Parameters
WINDOW_SIZE = 20
MAX_POSITION_PCT = float(os.getenv('MAX_POSITION_PCT', 0.10))  # Max 10% per position
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))      # Risk 2% per trade
ATR_STOP_MULTIPLIER = 2.5  # Match environment settings
ATR_TRAILING_MULTIPLIER = 3.0
MAX_POSITIONS = 5  # Max number of concurrent positions

# Symbols to trade (read from file or use default)
def load_symbols():
    # Check current dir and parent dir
    paths = ['my_portfolio.txt', '../my_portfolio.txt']
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                return symbols
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    print("Portfolio file not found, using default symbols")
    return ['SPY', 'AAPL', 'MSFT', 'NVDA', 'TSLA']

# =============================================================================
# Paper Trader Class
# =============================================================================
class PaperTrader:
    def __init__(self, model_path='models/shared_dqn_best.pth'):
        # Initialize Alpaca API
        self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        
        # Verify connection
        self.account = self.api.get_account()
        print(f"Connected to Alpaca Paper Trading")
        print(f"Account Status: {self.account.status}")
        print(f"Equity: ${float(self.account.equity):,.2f}")
        print(f"Buying Power: ${float(self.account.buying_power):,.2f}")
        
        # Load model - use dynamic state size from utils
        state_size = get_state_size(WINDOW_SIZE)
        action_size = 3
        # Must match training configuration (Noisy Nets = True)
        self.agent = DuelingDQN(state_size, action_size, use_noisy=True)
        
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load symbols
        self.symbols = load_symbols()
        print(f"Trading symbols: {self.symbols}")
        
        # Trade log
        self.trade_log = []
        self.log_file = f"logs/paper_trades_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs("logs", exist_ok=True)
        
        # Position tracking (for trailing stops)
        self.position_highs = {}
    
    def get_account_info(self):
        """Get current account info."""
        self.account = self.api.get_account()
        return {
            'equity': float(self.account.equity),
            'buying_power': float(self.account.buying_power),
            'cash': float(self.account.cash),
            'portfolio_value': float(self.account.portfolio_value)
        }
    
    def get_positions(self):
        """Get current positions."""
        positions = {}
        try:
            for pos in self.api.list_positions():
                positions[pos.symbol] = {
                    'qty': float(pos.qty),
                    'avg_entry': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                }
        except Exception as e:
            print(f"Error getting positions: {e}")
        return positions
    
    def get_historical_data(self, symbol, days=100):
        """Fetch historical daily bars."""
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            # Use IEX feed for free tier
            bars = self.api.get_bars(
                symbol, 
                tradeapi.TimeFrame.Day, 
                start=start_date,
                feed='iex'  # Explicitly use IEX for free tier
            ).df
            
            if bars.empty:
                return None
            
            # Rename columns
            bars = bars.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            return bars
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def get_realtime_price(self, symbol):
        """Get the latest realtime price (IEX for free tier)."""
        try:
            trade = self.api.get_latest_trade(symbol, feed='iex')
            return float(trade.price)
        except:
            # Fallback to last quote
            try:
                quote = self.api.get_latest_quote(symbol, feed='iex')
                return float(quote.ask_price)
            except:
                return None

    def is_earnings_approaching(self, symbol):
        """
        Check if earnings are within 3 days. 
        Note: Requires external data source. Placeholder logic.
        """
        # TODO: Integrate with specific earnings API (e.g. Finnhub)
        # For now, return False to allow trading
        return False
    
    def calculate_position_size(self, symbol, current_price, atr, account_equity):
        """Calculate position size based on ATR and risk parameters."""
        # Risk amount (2% of equity by default)
        risk_amount = account_equity * RISK_PER_TRADE
        
        # Stop distance based on ATR
        stop_distance = atr * ATR_STOP_MULTIPLIER
        
        # Position size based on risk
        if stop_distance > 0:
            risk_based_shares = int(risk_amount / stop_distance)
        else:
            risk_based_shares = 0
        
        # Max position cap (10% of portfolio)
        max_position_value = account_equity * MAX_POSITION_PCT
        max_shares = int(max_position_value / current_price)
        
        # Take the smaller of risk-based or max cap
        shares = min(risk_based_shares, max_shares)
        
        return max(shares, 0)  # Ensure non-negative
    
    def check_stop_loss(self, symbol, position, current_price, atr):
        """Check if stop loss should be triggered."""
        entry_price = position['avg_entry']
        
        # Initialize position high if not tracked
        if symbol not in self.position_highs:
            self.position_highs[symbol] = current_price
        
        # Update position high
        if current_price > self.position_highs[symbol]:
            self.position_highs[symbol] = current_price
        
        highest_price = self.position_highs[symbol]
        
        # Hard stop loss (2 ATR below entry)
        hard_stop = entry_price - (atr * ATR_STOP_MULTIPLIER)
        if current_price < hard_stop:
            return True, 'hard_stop'
        
        # Trailing stop (3 ATR below highest price)
        trailing_stop = highest_price - (atr * ATR_TRAILING_MULTIPLIER)
        if current_price < trailing_stop:
            return True, 'trailing_stop'
        
        return False, None
    
    def get_model_action(self, symbol, df, position_info):
        """Get action from DQN model."""
        if len(df) < WINDOW_SIZE + 5:
            return 0  # Hold if not enough data
        
        # Add indicators
        df = add_technical_indicators(df)
        
        # Get current step (latest data)
        current_step = len(df) - 1
        current_price = df['Close'].iloc[-1]
        
        # Market state
        market_state = normalize_state(df, current_step, WINDOW_SIZE)
        
        # Regime features are already included in normalize_state
        # Removing manual addition to avoid double counting
        pass
        
        # Portfolio state
        account = self.get_account_info()
        equity = account['equity']
        
        if position_info:
            unrealized_pnl = position_info['unrealized_plpc']
            market_value = position_info['market_value']
            in_position = 1.0
        else:
            unrealized_pnl = 0
            market_value = 0
            in_position = 0.0
        
        portfolio_state = np.array([
            account['cash'] / equity,
            market_value / equity,
            unrealized_pnl,
            in_position,
            0.0 # 5th feature to match model size 231
        ])
        
        state = np.concatenate((market_state, portfolio_state))
        
        # Get action (no exploration in live trading)
        action = self.agent.act(state, epsilon=0.0)
        
        return action
    
    def place_order(self, symbol, side, qty, reason='model_signal'):
        """Place an order and log it."""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            # Log trade
            trade_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'reason': reason,
                'order_id': order.id,
                'status': order.status
            }
            self.trade_log.append(trade_entry)
            self._save_log()
            
            print(f"  ✓ {side.upper()} order placed: {qty} shares of {symbol}")
            return order
            
        except Exception as e:
            print(f"  ✗ Order failed for {symbol}: {e}")
            return None
    
    def _save_log(self):
        """Save trade log to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.trade_log, f, indent=2)
    
    def run_once(self):
        """Run one iteration of the trading loop."""
        print(f"\n{'='*60}")
        print(f"Trading Iteration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Get account info
        account = self.get_account_info()
        print(f"Equity: ${account['equity']:,.2f} | Cash: ${account['cash']:,.2f}")
        
        # Get current positions
        positions = self.get_positions()
        print(f"Open Positions: {len(positions)}/{MAX_POSITIONS}")
        
        for symbol in self.symbols:
            print(f"\n--- {symbol} ---")
            
            # Get historical data
            df = self.get_historical_data(symbol)
            if df is None or len(df) < WINDOW_SIZE + 5:
                print(f"  Skipping: Insufficient data")
                continue
            
            # Add indicators to get ATR
            df = add_technical_indicators(df)
            current_price = df['Close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            print(f"  Price: ${current_price:.2f} | ATR: ${current_atr:.2f}")
            
            # Check if we have a position
            position = positions.get(symbol)
            
            if position:
                print(f"  Position: {position['qty']} shares @ ${position['avg_entry']:.2f}")
                print(f"  P/L: ${position['unrealized_pl']:.2f} ({position['unrealized_plpc']*100:.2f}%)")
                
                # Check stop loss first
                stop_triggered, stop_type = self.check_stop_loss(
                    symbol, position, current_price, current_atr
                )
                
                if stop_triggered:
                    print(f"  ⚠ {stop_type.upper()} triggered!")
                    self.place_order(symbol, 'sell', position['qty'], reason=stop_type)
                    del self.position_highs[symbol]
                    continue
            
            # Get model action
            action = self.get_model_action(symbol, df, position)
            action_name = ['HOLD', 'BUY', 'SELL'][action]
            print(f"  Model Signal: {action_name}")
            
            # Execute action
            if action == 1:  # BUY
                if position:
                    print(f"  Already in position. Holding.")
                elif len(positions) >= MAX_POSITIONS:
                    print(f"  Max positions reached. Skipping.")
                else:
                    # Calculate position size
                    shares = self.calculate_position_size(
                        symbol, current_price, current_atr, account['equity']
                    )
                    if shares > 0:
                        self.place_order(symbol, 'buy', shares, reason='model_buy')
                    else:
                        print(f"  Position size too small. Skipping.")
            
            elif action == 2:  # SELL
                if position:
                    self.place_order(symbol, 'sell', int(position['qty']), reason='model_sell')
                    if symbol in self.position_highs:
                        del self.position_highs[symbol]
                else:
                    print(f"  No position to sell.")
            
            else:  # HOLD
                print(f"  Holding.")
        
        print(f"\n{'='*60}")
        print(f"Iteration complete. Trades logged to {self.log_file}")
    
    def run_loop(self, interval_minutes=60):
        """Run continuously at specified interval."""
        print(f"Starting paper trading loop (interval: {interval_minutes} min)")
        
        while True:
            try:
                # Check if market is open
                clock = self.api.get_clock()
                
                if clock.is_open:
                    self.run_once()
                else:
                    next_open = clock.next_open.strftime('%Y-%m-%d %H:%M')
                    print(f"Market closed. Next open: {next_open}")
                
                # Sleep
                print(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\nStopping paper trader...")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait a minute before retrying


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading with Alpaca')
    parser.add_argument('--model', type=str, default='models/SHARED_dqn_best.pth', help='Model path')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=60, help='Loop interval (minutes)')
    
    args = parser.parse_args()
    
    try:
        trader = PaperTrader(model_path=args.model)
        
        if args.once:
            trader.run_once()
        else:
            trader.run_loop(interval_minutes=args.interval)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
