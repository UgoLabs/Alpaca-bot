import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from models.dueling_dqn import DuelingDQNTrader, TradingEnvironment
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LiveTrader:
    """Live trading system using trained DQN models"""
    
    def __init__(self, symbols=None, model_dir="models"):
        """Initialize live trader with list of symbols to trade"""
        # Load API credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.base_url = os.getenv('ALPACA_API_BASE_URL')
        
        if not all([self.api_key, self.api_secret, self.base_url]):
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
        
        # Get account information
        self.account = self.api.get_account()
        self.initial_balance = float(self.account.equity)
        logger.info(f"Account balance: ${self.initial_balance}")
        
        # Check if account is restricted
        if self.account.trading_blocked:
            logger.error("Account is currently restricted from trading")
            
        # Load symbols
        if symbols:
            self.symbols = symbols
        else:
            try:
                with open('my_portfolio.txt', 'r') as f:
                    self.symbols = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(self.symbols)} symbols from portfolio")
            except FileNotFoundError:
                logger.error("my_portfolio.txt not found")
                self.symbols = []
                
        # Load models
        self.models = {}
        for symbol in self.symbols:
            model_path = os.path.join(model_dir, f"{symbol}_dueling_dqn_best.h5")
            if os.path.exists(model_path):
                try:
                    # Create trader for this symbol
                    trader = DuelingDQNTrader()
                    # Load model
                    trader.agent.load(model_path)
                    self.models[symbol] = trader
                    logger.info(f"Loaded model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading model for {symbol}: {str(e)}")
            else:
                logger.warning(f"No model found for {symbol} at {model_path}")
                
        logger.info(f"Loaded {len(self.models)} models for trading")
        
        # Keep track of current positions
        self.positions = self._get_positions()
        
        # Trading parameters
        self.max_position = float(os.getenv('MAX_POSITION_SIZE', 0.25))
        self.stop_loss = float(os.getenv('STOP_LOSS_PERCENTAGE', 0.05))
        self.trailing_stop = float(os.getenv('TRAILING_STOP_PERCENTAGE', 0.03))
        
        # Observation window size
        self.window_size = 20
        
    def _get_positions(self):
        """Get current positions"""
        positions = {}
        try:
            for position in self.api.list_positions():
                positions[position.symbol] = {
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value)
                }
            logger.info(f"Current positions: {len(positions)}")
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {}
            
    def _get_historical_data(self, symbol, lookback_days=30):
        """Get historical data for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get historical data from Alpaca
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            ).df
            
            if len(bars) < self.window_size:
                logger.warning(f"Not enough historical data for {symbol}, only {len(bars)} bars")
                return None
                
            # Convert to DataFrame format expected by TradingEnvironment
            data = pd.DataFrame({
                'Open': bars['open'],
                'High': bars['high'],
                'Low': bars['low'],
                'Close': bars['close'],
                'Volume': bars['volume']
            })
            
            return data
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None
            
    def _create_environment(self, symbol, data):
        """Create a trading environment for a symbol"""
        # Calculate initial balance based on account equity and max position size
        position_size = self.initial_balance * self.max_position
        
        # Create environment
        env = TradingEnvironment(
            stock_data=data,
            initial_balance=position_size,
            window_size=self.window_size,
            max_position=self.max_position,
            stop_loss=self.stop_loss,
            trailing_stop=self.trailing_stop
        )
        
        return env
        
    def _get_action(self, symbol, data):
        """Get trading action for a symbol"""
        # Check if we have a model for this symbol
        if symbol not in self.models:
            logger.warning(f"No model available for {symbol}")
            return None
            
        # Create environment
        env = self._create_environment(symbol, data)
        
        # Reset environment to latest state
        state = env.reset()
        
        # Get action from model
        action = self.models[symbol].agent.act(state, training=False)
        
        return action
        
    def _place_order(self, symbol, action):
        """Place an order based on the action"""
        try:
            # Get latest price
            latest_quote = self.api.get_latest_quote(symbol)
            current_price = (float(latest_quote.ap) + float(latest_quote.bp)) / 2
            
            # Calculate position size
            position_value = self.initial_balance * self.max_position
            shares_to_trade = position_value / current_price
            
            # Round to nearest whole share
            shares_to_trade = round(shares_to_trade)
            
            # Check if we have an existing position
            existing_position = self.positions.get(symbol, None)
            
            if action == 1:  # Buy
                if existing_position:
                    logger.info(f"Already have position in {symbol}, not buying more")
                    return False
                    
                # Place buy order
                if shares_to_trade > 0:
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=shares_to_trade,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    logger.info(f"Buy order placed for {shares_to_trade} shares of {symbol}")
                    
                    # Set stop loss
                    stop_price = current_price * (1 - self.stop_loss)
                    self.api.submit_order(
                        symbol=symbol,
                        qty=shares_to_trade,
                        side='sell',
                        type='stop',
                        time_in_force='gtc',
                        stop_price=stop_price
                    )
                    logger.info(f"Stop loss set at ${stop_price:.2f} for {symbol}")
                    
                    return True
                else:
                    logger.warning(f"Not enough funds to buy {symbol}")
                    return False
                    
            elif action == 2:  # Sell
                if not existing_position:
                    logger.info(f"No position in {symbol} to sell")
                    return False
                    
                # Sell entire position
                qty = existing_position['qty']
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Sell order placed for {qty} shares of {symbol}")
                
                # Cancel any existing stop orders
                open_orders = self.api.list_orders(
                    status='open',
                    symbols=[symbol]
                )
                for order in open_orders:
                    if order.side == 'sell' and order.type == 'stop':
                        self.api.cancel_order(order.id)
                        logger.info(f"Cancelled stop loss order for {symbol}")
                        
                return True
                
            else:  # Hold
                logger.info(f"Holding position in {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {str(e)}")
            return False
            
    def run_trading_iteration(self):
        """Run one iteration of the trading loop"""
        # Update account balance
        self.account = self.api.get_account()
        self.initial_balance = float(self.account.equity)
        logger.info(f"Updated account balance: ${self.initial_balance}")
        
        # Update positions
        self.positions = self._get_positions()
        
        actions_taken = []
        
        for symbol in self.symbols:
            if symbol in self.models:
                # Get historical data
                data = self._get_historical_data(symbol)
                
                if data is not None and len(data) >= self.window_size:
                    # Get action
                    action = self._get_action(symbol, data)
                    
                    # Log action
                    action_names = {0: "Hold", 1: "Buy", 2: "Sell"}
                    logger.info(f"Action for {symbol}: {action_names.get(action, 'Unknown')}")
                    
                    # Place order if not holding
                    if action != 0:
                        order_placed = self._place_order(symbol, action)
                        if order_placed:
                            actions_taken.append((symbol, action))
                            
        return actions_taken
        
    def run_trading_loop(self, interval_minutes=15):
        """Run the trading loop"""
        logger.info(f"Starting trading loop with interval of {interval_minutes} minutes")
        
        running = True
        while running:
            try:
                # Check if market is open
                clock = self.api.get_clock()
                
                if clock.is_open:
                    logger.info("Market is open, running trading iteration")
                    
                    # Run trading iteration
                    actions = self.run_trading_iteration()
                    
                    if actions:
                        logger.info(f"Actions taken: {actions}")
                    else:
                        logger.info("No actions taken")
                        
                else:
                    next_open = clock.next_open.replace(tzinfo=None)
                    now = datetime.now()
                    time_to_open = (next_open - now).total_seconds() / 60
                    
                    logger.info(f"Market is closed. Next open: {next_open} ({time_to_open:.1f} minutes)")
                    
                # Sleep for the specified interval
                logger.info(f"Sleeping for {interval_minutes} minutes")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Trading loop stopped by user")
                running = False
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                # Sleep for a minute before trying again
                time.sleep(60)

def main():
    """Main function to run live trading"""
    parser = argparse.ArgumentParser(description='Run live trading with trained DQN models')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to trade')
    parser.add_argument('--interval', type=int, default=15, help='Trading interval in minutes')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing trained models')
    
    args = parser.parse_args()
    
    try:
        # Initialize live trader
        trader = LiveTrader(symbols=args.symbols, model_dir=args.model_dir)
        
        if not trader.models:
            logger.error("No models available for trading")
            return
            
        # Run trading loop
        trader.run_trading_loop(interval_minutes=args.interval)
        
    except Exception as e:
        logger.error(f"Error in live trading: {str(e)}")

if __name__ == "__main__":
    main() 