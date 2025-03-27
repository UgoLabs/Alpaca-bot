import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv
from collections import deque

# Import our custom models
from models.dueling_dqn import DuelingDQNAgent, TradingEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class HybridTradingSystem:
    """
    Hybrid trading system that combines DQN and LSTM models.
    DQN serves as the primary decision maker, with LSTM providing supporting signals.
    """
    
    def __init__(self, initial_balance=5222.58, max_positions=5, position_size=0.2,
                 stop_loss=0.05, trailing_stop=0.03, symbols=None):
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        
        # Default symbols if none provided
        self.symbols = symbols or [
            "PLTR", "NET", "NVDA", "TSLA", "INTC", "MSTR", "SMCI", 
            "META", "TSM", "NFLX", "AMZN", "MSFT"
        ]
        
        # Initialize models dictionary
        self.dqn_models = {}
        self.lstm_models = {}
        
        # State configuration for DQN
        self.state_size = int(os.getenv('TARGET_STATE_SIZE', 768))
        self.action_size = 3  # Hold, Buy, Sell
        
        # Portfolio tracking
        self.positions = {}  # symbol -> {shares, cost_basis, highest_price}
        self.balance = initial_balance
        self.portfolio_value_history = [initial_balance]
        self.trade_history = []
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained DQN and LSTM models"""
        logger.info("Loading pre-trained models...")
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Check which models exist
        for symbol in self.symbols:
            # Check for DQN model
            dqn_path = f"models/{symbol}_dueling_dqn_best.h5"
            if os.path.exists(dqn_path):
                # Create a new agent and load weights
                agent = DuelingDQNAgent(self.state_size, self.action_size)
                try:
                    agent.load(dqn_path)
                    self.dqn_models[symbol] = agent
                    logger.info(f"Loaded DQN model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading DQN model for {symbol}: {str(e)}")
            
            # Check for LSTM model
            lstm_path = f"models/{symbol}_lstm_model.keras"
            if os.path.exists(lstm_path):
                try:
                    model = load_model(lstm_path)
                    self.lstm_models[symbol] = model
                    logger.info(f"Loaded LSTM model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading LSTM model for {symbol}: {str(e)}")
        
        logger.info(f"Loaded {len(self.dqn_models)} DQN models and {len(self.lstm_models)} LSTM models")
    
    def _fetch_historical_data(self, symbols, period='5y', interval='1d'):
        """Fetch historical data for multiple symbols"""
        logger.info(f"Fetching {period} historical data for {len(symbols)} symbols...")
        all_data = {}
        
        for symbol in tqdm(symbols, desc="Loading Data"):
            try:
                data = yf.download(symbol, period=period, interval=interval, progress=False)
                
                # If data is multi-level, select only the data for this symbol
                if isinstance(data.columns, pd.MultiIndex):
                    # Flatten the multi-index columns
                    data_single_level = pd.DataFrame()
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if (col, symbol) in data.columns:
                            data_single_level[col] = data[(col, symbol)]
                    data = data_single_level
                
                # Add Date column
                data.reset_index(inplace=True)
                
                if not data.empty and len(data) > 100:  # Need at least ~100 trading days
                    all_data[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return all_data
    
    def prepare_dqn_environment(self, symbol, data):
        """Prepare the DQN environment for trading"""
        env = TradingEnvironment(
            stock_data=data,
            initial_balance=self.initial_balance,
            window_size=20,          # Use 20-day window to match LSTM
            commission_fee=0,        # Commission-free trading
            max_position=self.position_size,
            stop_loss=self.stop_loss,
            trailing_stop=self.trailing_stop
        )
        return env
    
    def prepare_lstm_features(self, data, lookback=20):
        """Prepare features for LSTM prediction with shorter 20-day lookback"""
        # Create features (similar to DataPreprocessor in train_lstm_multi_position.py)
        features_df = data.copy()
        
        # Basic price features
        features_df['Returns'] = data['Close'].pct_change()
        features_df['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volume features
        features_df['Volume_Change'] = data['Volume'].pct_change()
        features_df['Volume_MA10'] = data['Volume'].rolling(10).mean() / data['Volume']
        
        # Moving averages
        features_df['MA5'] = data['Close'].rolling(5).mean() / data['Close']
        features_df['MA10'] = data['Close'].rolling(10).mean() / data['Close']
        features_df['MA20'] = data['Close'].rolling(20).mean() / data['Close']
        features_df['MA50'] = data['Close'].rolling(50).mean() / data['Close']
        
        # Price channels
        features_df['Upper_Channel'] = data['High'].rolling(20).max() / data['Close']
        features_df['Lower_Channel'] = data['Low'].rolling(20).min() / data['Close']
        
        # Volatility
        features_df['Volatility'] = data['Close'].rolling(20).std() / data['Close']
        
        # RSI (14-period)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        features_df['MACD'] = ema12 - ema26
        features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9).mean()
        
        # Drop NaN values
        features_df.dropna(inplace=True)
        
        # Feature scaling with StandardScaler
        feature_columns = features_df.columns.difference(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        features = features_df[feature_columns].values
        
        # Simple normalization instead of fitting a scaler
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        feature_stds[feature_stds == 0] = 1  # Avoid division by zero
        features_scaled = (features - feature_means) / feature_stds
        
        # Create sequences for the most recent data point
        if len(features_scaled) >= lookback:
            sequence = features_scaled[-lookback:].reshape(1, lookback, features_scaled.shape[1])
            return sequence
        else:
            logger.warning(f"Not enough data for LSTM sequence, need {lookback} points but only have {len(features_scaled)}")
            return None
    
    def get_hybrid_action(self, symbol, dqn_state, current_price, historical_data):
        """
        Get trading action using both DQN and LSTM models.
        
        Returns:
        - action: The DQN action (0=Hold, 1=Buy, 2=Sell)
        - confidence: A confidence score for the action
        - lstm_prediction: The LSTM predicted return (if available)
        """
        # Default values
        action = 0  # Hold
        confidence = 0.5
        lstm_prediction = None
        
        # Get DQN prediction
        dqn_model = self.dqn_models.get(symbol)
        if dqn_model:
            # DQN provides the primary action
            action = dqn_model.act(dqn_state, training=False)
            
            # Get confidence from Q-values
            q_values = dqn_model.model.predict(np.array([dqn_state]), verbose=0)[0]
            max_q = np.max(q_values)
            min_q = np.min(q_values)
            q_range = max_q - min_q if max_q > min_q else 1
            dqn_confidence = (q_values[action] - min_q) / q_range
        else:
            dqn_confidence = 0.5  # Default confidence
            
        # Get LSTM prediction if available
        lstm_model = self.lstm_models.get(symbol)
        if lstm_model:
            # Prepare features for LSTM
            lstm_sequence = self.prepare_lstm_features(historical_data)
            if lstm_sequence is not None:
                try:
                    # Get LSTM prediction
                    regression_pred, classification_pred = lstm_model.predict(lstm_sequence, verbose=0)
                    predicted_return = regression_pred[0][0]
                    up_probability = classification_pred[0][0]
                    
                    # Combine DQN and LSTM signals
                    lstm_prediction = {
                        'predicted_return': predicted_return,
                        'up_probability': up_probability
                    }
                    
                    # Modify confidence based on LSTM prediction
                    if action == 1:  # Buy
                        lstm_confidence = up_probability
                    elif action == 2:  # Sell
                        lstm_confidence = 1 - up_probability
                    else:  # Hold
                        lstm_confidence = 0.5
                        
                    # Combine DQN and LSTM confidence (weighted average)
                    confidence = 0.7 * dqn_confidence + 0.3 * lstm_confidence
                    
                    # LSTM can override DQN in extreme cases
                    if action == 1 and predicted_return < -0.05 and up_probability < 0.3:
                        # DQN says buy but LSTM strongly disagrees
                        action = 0  # Change to hold
                        logger.info(f"LSTM override: DQN wanted to buy {symbol} but LSTM predicts negative return")
                    elif action == 0 and predicted_return > 0.05 and up_probability > 0.8:
                        # DQN says hold but LSTM sees strong opportunity
                        action = 1  # Change to buy
                        logger.info(f"LSTM override: DQN wanted to hold but LSTM sees strong opportunity in {symbol}")
                
                except Exception as e:
                    logger.error(f"Error in LSTM prediction for {symbol}: {str(e)}")
        
        return action, confidence, lstm_prediction
    
    def can_open_position(self):
        """Check if we can open another position"""
        return len(self.positions) < self.max_positions and self.balance > 0
    
    def buy(self, symbol, price, date, confidence=1.0):
        """Buy a position in the given symbol"""
        if symbol in self.positions:
            # Already holding this symbol
            return False
            
        if not self.can_open_position():
            # Can't open more positions
            return False
            
        # Calculate position size, adjusted by confidence
        base_position_value = self.initial_balance * self.position_size
        # Scale position size by confidence (0.5-1.5 range)
        position_scaling = 0.5 + confidence
        position_value = base_position_value * position_scaling
        position_value = min(position_value, self.balance)  # Limit by available balance
        
        if position_value < 100:  # Minimum position size ($100)
            return False
            
        # Calculate shares to buy
        shares = position_value / price
        
        # Update portfolio
        self.positions[symbol] = {
            'shares': shares,
            'cost_basis': price,
            'highest_price': price,
            'confidence': confidence
        }
        
        self.balance -= position_value
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'value': position_value,
            'confidence': confidence
        })
        
        logger.info(f"BUY: {symbol} at ${price:.2f} - {shares:.2f} shares (${position_value:.2f}, confidence: {confidence:.2f})")
        return True
    
    def sell(self, symbol, price, date, reason="SELL"):
        """Sell a position"""
        if symbol not in self.positions:
            # Don't have this position
            return False
            
        # Get position info
        position = self.positions[symbol]
        
        # Calculate position value
        position_value = position['shares'] * price
        
        # Update portfolio
        self.balance += position_value
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'action': reason,
            'price': price,
            'shares': position['shares'],
            'value': position_value,
            'confidence': position.get('confidence', 1.0)
        })
        
        profit = price / position['cost_basis'] - 1
        logger.info(f"{reason}: {symbol} at ${price:.2f} - {position['shares']:.2f} shares, P/L: {profit*100:.2f}%")
        
        # Remove position
        del self.positions[symbol]
        
        return True
    
    def update_portfolio(self, date, prices):
        """Update portfolio based on current prices"""
        # Update highest prices and check stops
        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                current_price = prices[symbol]
                
                # Update highest price
                if current_price > position['highest_price']:
                    position['highest_price'] = current_price
                
                # Check stop loss
                if current_price < position['cost_basis'] * (1 - self.stop_loss):
                    self.sell(symbol, current_price, date, "STOP_LOSS")
                
                # Check trailing stop
                elif current_price < position['highest_price'] * (1 - self.trailing_stop):
                    self.sell(symbol, current_price, date, "TRAILING_STOP")
        
        # Calculate current portfolio value
        portfolio_value = self.balance + sum(
            position['shares'] * prices[symbol] 
            for symbol, position in self.positions.items() 
            if symbol in prices
        )
        
        self.portfolio_value_history.append(portfolio_value)
        
        return portfolio_value
    
    def backtest(self, start_date=None, end_date=None, rebalance_interval=1):
        """
        Run a backtest of the hybrid trading system.
        
        Parameters:
        - start_date: Optional start date for backtesting
        - end_date: Optional end date for backtesting
        - rebalance_interval: Number of days between portfolio rebalancing (default: daily)
        
        Returns:
        - Dictionary with backtest results
        """
        logger.info(f"Starting backtest of hybrid DQN-LSTM trading system with {self.max_positions} max positions")
        logger.info(f"Using {rebalance_interval}-day rebalance interval and 20-day lookback window")
        
        # Fetch historical data
        historical_data = self._fetch_historical_data(self.symbols)
        
        if not historical_data:
            logger.error("No data available for backtesting")
            return None
        
        # Get common date range
        common_dates = None
        for symbol, data in historical_data.items():
            dates = set(data['Date'])
            if common_dates is None:
                common_dates = dates
            else:
                common_dates &= dates
        
        common_dates = sorted(list(common_dates))
        
        # Filter by date range if specified
        if start_date:
            start_date = pd.to_datetime(start_date)
            common_dates = [d for d in common_dates if pd.to_datetime(d) >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            common_dates = [d for d in common_dates if pd.to_datetime(d) <= end_date]
        
        logger.info(f"Backtesting on {len(common_dates)} trading days: {common_dates[0]} to {common_dates[-1]}")
        
        # Reset portfolio for backtest
        self.positions = {}
        self.balance = self.initial_balance
        self.portfolio_value_history = [self.initial_balance]
        self.trade_history = []
        
        # For tracking when to rebalance
        days_since_rebalance = 0
        
        # Initialize DQN environments
        environments = {}
        for symbol in self.dqn_models:
            if symbol in historical_data:
                environments[symbol] = self.prepare_dqn_environment(symbol, historical_data[symbol])
        
        # Run backtest day by day
        for date_idx, date in enumerate(tqdm(common_dates, desc="Backtest Progress")):
            # Skip first 20 days to ensure we have enough history for lookback window
            if date_idx < 20:
                continue
            
            days_since_rebalance += 1
            
            # Get current prices for all symbols
            prices = {}
            for symbol, data in historical_data.items():
                date_rows = data[data['Date'] == date]
                if not date_rows.empty:
                    prices[symbol] = date_rows['Close'].values[0]
            
            # Update portfolio with current prices (check stop losses, etc.)
            portfolio_value = self.update_portfolio(date, prices)
            
            # PERIODIC REBALANCING: Sell all positions and reset
            if days_since_rebalance >= rebalance_interval:
                logger.info(f"REBALANCE DAY: {date}")
                
                # Sell all current positions
                for symbol in list(self.positions.keys()):
                    if symbol in prices:
                        self.sell(symbol, prices[symbol], date, "REBALANCE")
                
                days_since_rebalance = 0
            
            # FORCE CLOSE POSITIONS HELD TOO LONG (Max 7 days)
            for symbol, position in list(self.positions.items()):
                # Get entry date
                entry_date = None
                for trade in self.trade_history:
                    if trade['symbol'] == symbol and trade['action'] == 'BUY':
                        entry_date = pd.to_datetime(trade['date'])
                        # Find only the most recent buy for this symbol
                        for later_trade in self.trade_history[::-1]:
                            if later_trade['symbol'] == symbol and later_trade['action'] in ['SELL', 'STOP_LOSS', 'TRAILING_STOP', 'REBALANCE']:
                                # Found a sell after this buy, so this buy isn't the active one
                                entry_date = None
                                break
                            if later_trade['symbol'] == symbol and later_trade['action'] == 'BUY':
                                # This is the most recent buy
                                entry_date = pd.to_datetime(later_trade['date'])
                                break
                
                if entry_date:
                    # If entry date is more than 7 days ago, force close
                    current_date = pd.to_datetime(date)
                    holding_days = (current_date - entry_date).days
                    if holding_days >= 7:  # Max 1 week holding period
                        if symbol in prices:
                            self.sell(symbol, prices[symbol], date, "MAX_HOLD_PERIOD")
                            logger.info(f"Forced close of {symbol} after {holding_days} days holding period")
            
            # Make trading decisions for each symbol
            for symbol in self.symbols:
                # Skip if we already have this position or it's not in our models/data
                if symbol in self.positions or symbol not in self.dqn_models or symbol not in historical_data:
                    continue
                
                # Get data for this symbol
                symbol_data = historical_data[symbol]
                date_idx_in_data = symbol_data[symbol_data['Date'] == date].index
                
                if len(date_idx_in_data) == 0:
                    continue
                
                date_idx_in_data = date_idx_in_data[0]
                
                # Get current price
                if symbol not in prices:
                    continue
                
                current_price = prices[symbol]
                
                # Get DQN state
                env = environments.get(symbol)
                if env:
                    # Reset the environment to the current position in data
                    env.current_step = date_idx_in_data - 1
                    
                    try:
                        dqn_state = env._get_observation()
                        
                        # Get hybrid action (DQN + LSTM)
                        action, confidence, lstm_pred = self.get_hybrid_action(
                            symbol, 
                            dqn_state, 
                            current_price, 
                            symbol_data.iloc[:date_idx_in_data+1]
                        )
                        
                        # Execute action
                        if action == 1 and self.can_open_position():  # Buy
                            self.buy(symbol, current_price, date, confidence)
                        
                        # Log LSTM predictions periodically
                        if date_idx % 20 == 0 and lstm_pred:
                            logger.info(f"LSTM prediction for {symbol}: Return={lstm_pred['predicted_return']:.4f}, " +
                                       f"UpProb={lstm_pred['up_probability']:.4f}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol} on {date}: {str(e)}")
            
            # Print portfolio status every 20 days
            if date_idx % 20 == 0:
                logger.info(f"\nDay {date_idx}: {date}")
                logger.info(f"Portfolio Value: ${portfolio_value:.2f} ({((portfolio_value/self.initial_balance)-1)*100:.2f}%)")
                logger.info(f"Positions: {list(self.positions.keys())}")
                logger.info(f"Cash: ${self.balance:.2f}")
                logger.info(f"Total Trades: {len(self.trade_history)}")
        
        # Calculate performance metrics
        final_value = self.portfolio_value_history[-1]
        pnl = final_value - self.initial_balance
        pnl_pct = pnl / self.initial_balance * 100
        
        # Calculate daily returns
        daily_returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02 / 252  # Daily risk-free rate (assume 2% annual)
        sharpe_ratio = 0
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            excess_returns = daily_returns - risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = self.portfolio_value_history[0]
        for value in self.portfolio_value_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Count trade types
        buys = len([t for t in self.trade_history if t['action'] == 'BUY'])
        sells = len([t for t in self.trade_history if t['action'] == 'SELL'])
        stop_losses = len([t for t in self.trade_history if t['action'] == 'STOP_LOSS'])
        trailing_stops = len([t for t in self.trade_history if t['action'] == 'TRAILING_STOP'])
        
        # Calculate win rate
        wins = 0
        for i in range(len(self.trade_history)):
            trade = self.trade_history[i]
            if trade['action'] == 'BUY':
                # Find the corresponding sell
                for j in range(i+1, len(self.trade_history)):
                    sell_trade = self.trade_history[j]
                    if sell_trade['symbol'] == trade['symbol'] and sell_trade['action'] in ['SELL', 'STOP_LOSS', 'TRAILING_STOP']:
                        if sell_trade['price'] > trade['price']:
                            wins += 1
                        break
        
        # Calculate win rate
        trades = min(buys, buys + sells + stop_losses + trailing_stops - buys)
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        # Display backtest results
        logger.info("\nBACKTEST RESULTS:")
        
        metrics = {
            "Test Period": f"{len(common_dates)} days",
            "Initial Balance": self.initial_balance,
            "Final Portfolio": final_value,
            "P&L": pnl,
            "Return %": pnl_pct,
            "Annualized Return %": pnl_pct / len(common_dates) * 252,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown %": max_drawdown * 100,
            "Win Rate %": win_rate,
            "Total Trades": buys,  # Each buy eventually becomes a sell
            "Buys": buys,
            "Sells": sells,
            "Stop Losses": stop_losses,
            "Trailing Stops": trailing_stops
        }
        
        logger.info("\n" + "="*80)
        logger.info(f"{'METRIC':<25}{'VALUE':<15}")
        logger.info("-"*80)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key:<25}{value:<15.2f}")
            else:
                logger.info(f"{key:<25}{value:<15}")
        
        logger.info("="*80 + "\n")
        
        # Calculate days to target equity
        target_equity = float(os.getenv('TARGET_EQUITY', 25000.0))
        current_equity = self.initial_balance - 4717.79  # Current cash situation
        days_to_target = "N/A"
        
        if len(daily_returns) > 0 and np.mean(daily_returns) > 0:
            daily_return_rate = 1 + np.mean(daily_returns)
            days_to_target = np.log(target_equity / current_equity) / np.log(daily_return_rate)
            days_to_target = int(days_to_target) if not np.isnan(days_to_target) else "N/A"
        
        # Print projection
        logger.info("\nPROJECTED PATH TO $25,000:")
        logger.info(f"Starting equity: ${current_equity:.2f}")
        logger.info(f"Target equity: ${target_equity:.2f}")
        if isinstance(days_to_target, int):
            logger.info(f"Daily return rate: {(np.mean(daily_returns))*100:.4f}%")
            logger.info(f"Estimated days to reach target: {days_to_target}")
            logger.info(f"Estimated months to reach target: {days_to_target/21:.1f}")
        else:
            logger.info(f"Cannot estimate time to target with current return rate")
        
        # Plot results
        self.plot_backtest_results()
        
        return metrics
    
    def plot_backtest_results(self):
        """Plot the backtest results"""
        plt.figure(figsize=(20, 15))
        
        # Plot portfolio value
        plt.subplot(3, 2, 1)
        plt.plot(self.portfolio_value_history)
        plt.title("Portfolio Value")
        plt.xlabel("Trading Day")
        plt.ylabel("Value ($)")
        plt.grid(True)
        
        # Plot daily returns
        daily_returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
        plt.subplot(3, 2, 2)
        plt.hist(daily_returns * 100, bins=50)
        plt.title("Daily Returns Distribution")
        plt.xlabel("Daily Return (%)")
        plt.ylabel("Frequency")
        plt.grid(True)
        
        # Plot drawdown
        max_portfolio_value = np.maximum.accumulate(self.portfolio_value_history)
        drawdowns = (max_portfolio_value - self.portfolio_value_history) / max_portfolio_value
        plt.subplot(3, 2, 3)
        plt.plot(drawdowns * 100)
        plt.title("Drawdown")
        plt.xlabel("Trading Day")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        
        # Plot trade frequency by symbol
        if self.trade_history:
            trade_symbols = [t['symbol'] for t in self.trade_history if t['action'] == 'BUY']
            symbol_counts = {symbol: trade_symbols.count(symbol) for symbol in set(trade_symbols)}
            symbols = list(symbol_counts.keys())
            counts = list(symbol_counts.values())
            
            plt.subplot(3, 2, 4)
            plt.bar(symbols, counts)
            plt.title("Trade Frequency by Symbol")
            plt.xlabel("Symbol")
            plt.ylabel("Number of Trades")
            plt.xticks(rotation=45)
            plt.grid(True)
            
            # Plot confidence distribution
            confidences = [t['confidence'] for t in self.trade_history if 'confidence' in t]
            if confidences:
                plt.subplot(3, 2, 5)
                plt.hist(confidences, bins=20)
                plt.title("Trade Confidence Distribution")
                plt.xlabel("Confidence Score")
                plt.ylabel("Frequency")
                plt.grid(True)
            
            # Plot P&L by trade
            buys = [t for t in self.trade_history if t['action'] == 'BUY']
            sell_actions = ['SELL', 'STOP_LOSS', 'TRAILING_STOP', 'REBALANCE']
            pnl_by_trade = []
            
            for buy in buys:
                # Find corresponding sell
                for sell in self.trade_history:
                    if (sell['symbol'] == buy['symbol'] and sell['action'] in sell_actions and 
                        pd.to_datetime(sell['date']) > pd.to_datetime(buy['date'])):
                        pnl = (sell['price'] / buy['price'] - 1) * 100  # P&L as percentage
                        pnl_by_trade.append((buy['symbol'], pnl))
                        break
            
            if pnl_by_trade:
                plt.subplot(3, 2, 6)
                symbols, pnls = zip(*pnl_by_trade)
                plt.bar(range(len(pnls)), pnls)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.title("P&L by Trade")
                plt.xlabel("Trade #")
                plt.ylabel("P&L (%)")
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("hybrid_trading_results.png")
        logger.info("Saved backtest charts to hybrid_trading_results.png")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Hybrid DQN-LSTM Trading System")
    parser.add_argument("--balance", type=float, default=5222.58, help="Initial balance")
    parser.add_argument("--positions", type=int, default=5, help="Maximum number of positions")
    parser.add_argument("--position-size", type=float, default=0.2, help="Position size as fraction of portfolio")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="Stop loss percentage (0.05 = 5%)")
    parser.add_argument("--trailing-stop", type=float, default=0.03, help="Trailing stop percentage (0.03 = 3%)")
    parser.add_argument("--rebalance", type=int, default=1, help="Rebalance interval in days")
    parser.add_argument("--symbols", type=str, nargs="+", help="Stock symbols to trade")
    parser.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Create hybrid trading system
    system = HybridTradingSystem(
        initial_balance=args.balance,
        max_positions=args.positions,
        position_size=args.position_size,
        stop_loss=args.stop_loss,
        trailing_stop=args.trailing_stop,
        symbols=args.symbols
    )
    
    # Run backtest
    results = system.backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        rebalance_interval=args.rebalance
    )
    
    if results:
        print("\nBacktest complete!")
        print(f"Final portfolio value: ${results['Final Portfolio']:.2f}")
        print(f"Return: {results['Return %']:.2f}%")
        print(f"Sharpe ratio: {results['Sharpe Ratio']:.2f}")
        print(f"Win rate: {results['Win Rate %']:.2f}%")
        print(f"Total trades: {results['Total Trades']}")
        if isinstance(results.get('days_to_target'), int):
            print(f"Estimated days to $25K: {results['days_to_target']}")

if __name__ == "__main__":
    main() 