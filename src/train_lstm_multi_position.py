import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import argparse
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Concatenate, Attention, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Configure for high RAM usage and performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)

import yfinance as yf

# Define constants
INITIAL_BALANCE = 5222.58
TARGET_EQUITY = 25000.00
CURRENT_CASH = -4717.79  # Current cash balance is negative
LOOKBACK_WINDOW = 60     # Longer lookback for LSTM
FORECAST_DAYS = 5        # Predict 5 days ahead
MAX_POSITIONS = 5
POSITION_SIZE = 0.2      # 20% per position
BATCH_SIZE = 1024        # Larger batch size for 64GB RAM
EPOCHS = 100
PATIENCE = 15
LSTM_UNITS = 256         # Larger LSTM units for more capacity

# Pre-selected highly volatile stocks with good movement
FAST_GROWTH_SYMBOLS = [
    "PLTR", "NET", "NVDA", "TSLA", "INTC", "MSTR", "SMCI", 
    "META", "TSM", "NFLX", "AMZN", "MSFT"
]

class DataPreprocessor:
    """Handles data preprocessing for LSTM models"""
    
    def __init__(self, sequence_length=LOOKBACK_WINDOW, forecast_horizon=FORECAST_DAYS):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def create_features(self, df):
        """Create technical indicators and features from price data"""
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Basic price features
        data['Returns'] = df['Close'].pct_change()
        data['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volume features
        data['Volume_Change'] = df['Volume'].pct_change()
        data['Volume_MA10'] = df['Volume'].rolling(10).mean() / df['Volume']
        
        # Moving averages
        data['MA5'] = df['Close'].rolling(5).mean() / df['Close']
        data['MA10'] = df['Close'].rolling(10).mean() / df['Close']
        data['MA20'] = df['Close'].rolling(20).mean() / df['Close']
        data['MA50'] = df['Close'].rolling(50).mean() / df['Close']
        
        # Price channels
        data['Upper_Channel'] = df['High'].rolling(20).max() / df['Close']
        data['Lower_Channel'] = df['Low'].rolling(20).min() / df['Close']
        
        # Volatility
        data['Volatility'] = df['Close'].rolling(20).std() / df['Close']
        
        # RSI (14-period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data
    
    def create_sequences(self, df):
        """Create sequences for LSTM training"""
        # Define features and target
        features = df.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1, errors='ignore')
        target = df['Close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
        
        # Create binary target (1 if price goes up, 0 if down)
        binary_target = (target > 0).astype(int)
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]
        binary_target = binary_target[valid_idx]
        price = df['Close'][valid_idx]
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Create sequences
        X, y, y_binary, p = [], [], [], []
        
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i+self.sequence_length])
            y.append(target.iloc[i+self.sequence_length-1])
            y_binary.append(binary_target.iloc[i+self.sequence_length-1])
            p.append(price.iloc[i+self.sequence_length-1])
        
        return np.array(X), np.array(y), np.array(y_binary), np.array(p)

def create_lstm_model(input_shape, output_units=1):
    """Create a complex LSTM model for time series prediction"""
    inputs = Input(shape=input_shape)
    
    # Convolutional Feature Extraction Branch
    conv_branch = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = MaxPooling1D(pool_size=2)(conv_branch)
    conv_branch = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv_branch)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = MaxPooling1D(pool_size=2)(conv_branch)
    
    # LSTM Branch 1 - Bidirectional
    lstm_branch1 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(inputs)
    lstm_branch1 = Dropout(0.25)(lstm_branch1)
    lstm_branch1 = Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=False))(lstm_branch1)
    lstm_branch1 = Dropout(0.25)(lstm_branch1)
    
    # LSTM Branch 2 - Stacked
    lstm_branch2 = LSTM(LSTM_UNITS, return_sequences=True)(inputs)
    lstm_branch2 = Dropout(0.25)(lstm_branch2)
    lstm_branch2 = LSTM(LSTM_UNITS, return_sequences=True)(lstm_branch2)
    lstm_branch2 = Dropout(0.25)(lstm_branch2)
    lstm_branch2 = LSTM(LSTM_UNITS // 2, return_sequences=False)(lstm_branch2)
    lstm_branch2 = Dropout(0.25)(lstm_branch2)
    
    # Flatten convolutional branch
    conv_branch = Flatten()(conv_branch)
    
    # Merge branches
    merged = Concatenate()([conv_branch, lstm_branch1, lstm_branch2])
    
    # Dense layers for final prediction
    dense = Dense(256, activation='relu')(merged)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    dense = Dense(64, activation='relu')(dense)
    
    # Output layers
    regression_output = Dense(output_units, name='regression')(dense)
    classification_output = Dense(1, activation='sigmoid', name='classification')(dense)
    
    # Create model with multiple outputs
    model = Model(inputs=inputs, outputs=[regression_output, classification_output])
    
    # Compile model with different loss functions for each output
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'regression': 'mse',
            'classification': 'binary_crossentropy'
        },
        metrics={
            'regression': ['mae'],
            'classification': ['accuracy']
        },
        loss_weights={
            'regression': 0.3,
            'classification': 0.7
        }
    )
    
    return model

def fetch_historical_data(symbols, period='5y'):
    """Fetch historical data for multiple symbols"""
    print(f"Fetching {period} historical data for {len(symbols)} symbols...")
    all_data = {}
    
    for symbol in tqdm(symbols, desc="Loading Data"):
        try:
            data = yf.download(symbol, period=period, interval='1d', progress=False)
            
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
            
            if not data.empty and len(data) > 500:  # Need at least ~500 trading days
                all_data[symbol] = data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    
    return all_data

class MultiPositionPortfolio:
    """Portfolio that manages multiple positions"""
    
    def __init__(self, symbols, initial_balance=INITIAL_BALANCE, max_positions=MAX_POSITIONS, 
                 position_size=POSITION_SIZE, stop_loss=0.05, trailing_stop=0.03):
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        
        # Initialize positions dictionary
        self.positions = {}  # symbol -> {shares, cost_basis, highest_price}
        
        # Tracking variables
        self.portfolio_value_history = [initial_balance]
        self.trade_history = []  # (date, symbol, action, price, shares, value)
    
    def can_open_position(self):
        """Check if we can open another position"""
        return len(self.positions) < self.max_positions and self.balance > 0
    
    def get_available_cash_per_position(self):
        """Get available cash per position"""
        if len(self.positions) >= self.max_positions:
            return 0
        
        # Calculate how much cash is available for each remaining position
        remaining_positions = self.max_positions - len(self.positions)
        if remaining_positions == 0:
            return 0
            
        return self.balance / remaining_positions
    
    def get_portfolio_value(self, prices):
        """Calculate total portfolio value based on current prices"""
        position_value = sum(
            position['shares'] * prices[symbol]
            for symbol, position in self.positions.items()
            if symbol in prices
        )
        return self.balance + position_value
    
    def buy(self, symbol, price, date):
        """Buy a position in the given symbol"""
        if symbol in self.positions:
            # Already holding this symbol
            return False
            
        if not self.can_open_position():
            # Can't open more positions
            return False
            
        # Calculate position size
        position_value = self.initial_balance * self.position_size
        position_value = min(position_value, self.balance)  # Limit by available balance
        
        if position_value < 100:  # Minimum position size ($100)
            return False
            
        # Calculate shares to buy
        shares = position_value / price
        
        # Update portfolio
        self.positions[symbol] = {
            'shares': shares,
            'cost_basis': price,
            'highest_price': price
        }
        
        self.balance -= position_value
        
        # Record trade
        self.trade_history.append((date, symbol, 'BUY', price, shares, position_value))
        
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
        self.trade_history.append((date, symbol, reason, price, position['shares'], position_value))
        
        # Remove position
        del self.positions[symbol]
        
        return True
    
    def update(self, date, prices):
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio from a series of returns"""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from portfolio values"""
    peak = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown * 100  # Convert to percentage

def train_lstm_models(symbols, training_data):
    """Train LSTM models for each symbol"""
    print(f"Training LSTM models for {len(symbols)} symbols...")
    
    models = {}
    preprocessors = {}
    
    for symbol in symbols:
        if symbol not in training_data:
            continue
            
        print(f"\n{'='*40}")
        print(f"TRAINING LSTM MODEL FOR {symbol}")
        print(f"{'='*40}")
        
        # Preprocess data
        preprocessor = DataPreprocessor(sequence_length=LOOKBACK_WINDOW, forecast_horizon=FORECAST_DAYS)
        data = preprocessor.create_features(training_data[symbol])
        
        X, y_reg, y_cls, prices = preprocessor.create_sequences(data)
        
        # Split data
        X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val, prices_train, prices_val = train_test_split(
            X, y_reg, y_cls, prices, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        # Create and train model
        model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Early stopping and checkpoint
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # Custom callback to evaluate P/L every 10 epochs
        class PLEvaluationCallback(tf.keras.callbacks.Callback):
            def __init__(self, validation_data, prices, initial_balance=INITIAL_BALANCE):
                self.X_val = validation_data[0]
                self.prices = prices
                self.initial_balance = initial_balance
                self.best_pl = -np.inf
                self.best_epoch = 0
                
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    # Make predictions on validation data
                    reg_preds, cls_preds = self.model.predict(self.X_val, verbose=0)
                    
                    # Simple trading strategy for P/L calculation
                    balance = self.initial_balance
                    position = 0  # 0 = no position, 1 = long position
                    entry_price = 0
                    trades = 0
                    wins = 0
                    
                    for i in range(len(reg_preds)):
                        pred_return = reg_preds[i][0]
                        pred_up_prob = cls_preds[i][0]
                        current_price = self.prices[i]
                        
                        # Calculate signal
                        signal = 0  # 0 = hold, 1 = buy, -1 = sell
                        
                        if position == 0 and pred_return > 0.01 and pred_up_prob > 0.6:
                            # Buy signal
                            signal = 1
                        elif position == 1 and (pred_return < -0.01 or pred_up_prob < 0.4):
                            # Sell signal
                            signal = -1
                            
                        # Execute trade
                        if signal == 1 and position == 0:
                            # Buy
                            position = 1
                            entry_price = current_price
                            trades += 1
                        elif signal == -1 and position == 1:
                            # Sell
                            position = 0
                            exit_price = current_price
                            
                            # Calculate P/L for this trade
                            trade_pl = (exit_price / entry_price - 1) * 100
                            balance *= (1 + trade_pl / 100)
                            
                            if trade_pl > 0:
                                wins += 1
                                
                            entry_price = 0
                            trades += 1
                    
                    # Calculate final P/L
                    pl = balance - self.initial_balance
                    pl_pct = (pl / self.initial_balance) * 100
                    win_rate = (wins / trades * 100) if trades > 0 else 0
                    
                    # Save best P/L
                    if pl_pct > self.best_pl:
                        self.best_pl = pl_pct
                        self.best_epoch = epoch + 1
                    
                    print(f"\nEpoch {epoch+1} validation P/L: ${pl:.2f} ({pl_pct:.2f}%)")
                    print(f"Trades: {trades}, Win rate: {win_rate:.1f}%, Best P/L so far: {self.best_pl:.2f}% at epoch {self.best_epoch}")
        
        # Create P/L evaluation callback
        pl_callback = PLEvaluationCallback(
            validation_data=(X_val, y_reg_val, y_cls_val),
            prices=prices_val
        )
        
        # Train model
        history = model.fit(
            X_train, 
            {'regression': y_reg_train, 'classification': y_cls_train},
            validation_data=(X_val, {'regression': y_reg_val, 'classification': y_cls_val}),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, reduce_lr, pl_callback],
            verbose=1
        )
        
        # Save model and preprocessor
        model.save(f"models/{symbol}_lstm_model.keras")
        models[symbol] = model
        preprocessors[symbol] = preprocessor
        
        print(f"\nModel for {symbol} trained and saved")
        print(f"Best validation P/L: {pl_callback.best_pl:.2f}% at epoch {pl_callback.best_epoch}")
        
    return models, preprocessors

def backtest_lstm_portfolio(symbols, testing_data, models, preprocessors, initial_balance=INITIAL_BALANCE):
    """Backtest the LSTM models on a multi-position portfolio"""
    print("\n" + "="*80)
    print("BACKTESTING LSTM MULTI-POSITION PORTFOLIO")
    print("="*80)
    
    # Get common date range for testing
    common_dates = None
    for symbol in testing_data:
        if common_dates is None:
            common_dates = set(testing_data[symbol]['Date'])
        else:
            common_dates &= set(testing_data[symbol]['Date'])
    
    common_dates = sorted(list(common_dates))
    print(f"Common testing period: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} days)")
    
    # Initialize portfolio
    portfolio = MultiPositionPortfolio(
        symbols=symbols,
        initial_balance=initial_balance,
        max_positions=MAX_POSITIONS,
        position_size=POSITION_SIZE,
        stop_loss=0.05,
        trailing_stop=0.03
    )
    
    # Run backtest
    print("Running backtest...")
    
    portfolio_values = [initial_balance]
    predictions = {}
    
    # Minimum sequence length needed
    min_seq_length = LOOKBACK_WINDOW + FORECAST_DAYS
    
    # Track days since last trade
    days_since_trade = 0
    forced_trade_interval = 10  # Force a trade every 10 days if none has occurred
    
    # Weekly rebalance counter
    days_since_rebalance = 0
    rebalance_interval = 5  # Rebalance every 5 trading days
    
    for i, date in enumerate(tqdm(common_dates, desc="Backtest Progress")):
        # Skip first days until we have enough data
        if i < min_seq_length:
            continue
        
        days_since_trade += 1
        days_since_rebalance += 1
            
        # Get current prices
        prices = {}
        for symbol in testing_data:
            symbol_data = testing_data[symbol]
            date_idx = symbol_data[symbol_data['Date'] == date].index
            if len(date_idx) > 0:
                prices[symbol] = symbol_data.loc[date_idx[0], 'Close']
        
        # Skip days with no price data
        if not prices:
            print(f"Warning: No price data found for {date}, skipping day")
            continue
        
        # Update portfolio with current prices
        portfolio.update(date, prices)
        
        # Make predictions for all symbols
        for symbol in models:
            if symbol not in testing_data or symbol not in preprocessors:
                continue
                
            # Get preprocessor
            preprocessor = preprocessors[symbol]
            
            # Get data up to current date
            symbol_data = testing_data[symbol]
            date_idx = symbol_data[symbol_data['Date'] == date].index
            
            if len(date_idx) == 0:
                continue
                
            current_idx = date_idx[0]
            
            # Make sure we have enough data
            if current_idx < LOOKBACK_WINDOW:
                continue
                
            # Get historical data for sequence
            hist_data = symbol_data.iloc[current_idx-LOOKBACK_WINDOW:current_idx+1]
            
            # Preprocess data
            try:
                processed_data = preprocessor.create_features(hist_data)
                
                # Check if we have enough processed data
                if len(processed_data) < LOOKBACK_WINDOW:
                    continue
                    
                # Get latest sequence
                features = processed_data.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1, errors='ignore')
                features_scaled = preprocessor.feature_scaler.transform(features)
                
                # Create sequence
                sequence = features_scaled[-LOOKBACK_WINDOW:].reshape(1, LOOKBACK_WINDOW, features_scaled.shape[1])
                
                # Make prediction
                reg_pred, cls_pred = models[symbol].predict(sequence, verbose=0)
                
                # Store prediction
                predictions[symbol] = {
                    'return_pred': reg_pred[0][0],
                    'up_prob': cls_pred[0][0],
                    'current_price': prices.get(symbol, 0)
                }
            except Exception as e:
                print(f"Error in prediction pipeline for {symbol}: {str(e)}")
                continue
        
        # Print some predictions for monitoring
        if i % 20 == 0 and len(predictions) > 0:
            print(f"\nSample predictions for {date}:")
            for symbol, pred in list(predictions.items())[:3]:
                print(f"{symbol}: Return={pred['return_pred']:.4f}, UpProb={pred['up_prob']:.4f}, Price=${pred['current_price']:.2f}")
        
        # FORCED PERIODIC REBALANCING: Sell everything on rebalance days
        if days_since_rebalance >= rebalance_interval and len(portfolio.positions) > 0:
            print(f"\n>>> REBALANCE DAY ({date}): Liquidating positions for reallocation")
            # Sell all current positions
            for symbol in list(portfolio.positions.keys()):
                if symbol in prices:
                    portfolio.sell(symbol, prices[symbol], date, "REBALANCE")
                    print(f"REBALANCE SELL: {symbol} at ${prices[symbol]:.2f}")
            
            days_since_rebalance = 0
            days_since_trade = 0
        
        # VERY AGGRESSIVE BUY CRITERIA: If we have room for positions, buy the best predictions
        if portfolio.can_open_position() and len(predictions) > 0:
            # Rank all symbols by prediction score (combination of return and probability)
            candidates = []
            for symbol, pred in predictions.items():
                if symbol not in portfolio.positions and symbol in prices:
                    # Aggressive scoring that favors any positive prediction
                    score = pred['return_pred'] * pred['up_prob'] * 10  # Amplify signal
                    if pred['up_prob'] > 0.5:  # Any better than random chance
                        candidates.append((symbol, score, pred))
            
            # Sort by score, buy top candidates
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 2 candidates or all if less than 2
                for symbol, score, pred in candidates[:2]:
                    if portfolio.can_open_position():
                        success = portfolio.buy(symbol, prices[symbol], date)
                        if success:
                            print(f"BUY: {symbol} at ${prices[symbol]:.2f} - Score: {score:.4f}")
                            days_since_trade = 0
        
        # FORCED BUYS: If no trades for a while and we have cash, force a buy
        if days_since_trade >= forced_trade_interval and portfolio.can_open_position() and len(predictions) > 0:
            print(f"\n>>> FORCE TRADING DAY ({date}): No trades for {days_since_trade} days")
            
            # Rank all symbols by up probability only
            all_candidates = [(symbol, pred['up_prob'], pred) 
                             for symbol, pred in predictions.items()
                             if symbol not in portfolio.positions and symbol in prices]
            
            if all_candidates:
                # Sort by up probability
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Just buy the top symbol by up probability
                for symbol, up_prob, pred in all_candidates[:1]:
                    if portfolio.can_open_position():
                        success = portfolio.buy(symbol, prices[symbol], date)
                        if success:
                            print(f"FORCED BUY: {symbol} at ${prices[symbol]:.2f} - UpProb: {up_prob:.4f}")
                            days_since_trade = 0
        
        # Store portfolio value
        portfolio_value = portfolio.get_portfolio_value(prices)
        portfolio_values.append(portfolio_value)
        
        # Print portfolio status every 20 days
        if i % 20 == 0:
            print(f"\nDay {i-min_seq_length}: {date}")
            print(f"Portfolio Value: ${portfolio_value:.2f} ({((portfolio_value/initial_balance)-1)*100:.2f}%)")
            print(f"Positions: {list(portfolio.positions.keys())}")
            print(f"Cash: ${portfolio.balance:.2f}")
            print(f"Total Trades: {len(portfolio.trade_history)}")
    
    # Print final portfolio state
    print(f"\nFinal portfolio state:")
    print(f"Positions: {list(portfolio.positions.keys())}")
    print(f"Cash balance: ${portfolio.balance:.2f}")
    print(f"Trade history: {len(portfolio.trade_history)} trades")
    
    # Calculate performance metrics
    final_portfolio = portfolio_values[-1]
    test_pnl = final_portfolio - initial_balance
    test_pnl_pct = (test_pnl / initial_balance) * 100
    
    # Calculate daily returns for Sharpe ratio
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd = calculate_max_drawdown(portfolio_values)
    
    # Count trades
    num_trades = len(portfolio.trade_history)
    buys = len([t for t in portfolio.trade_history if t[2] == 'BUY'])
    sells = len([t for t in portfolio.trade_history if t[2] == 'SELL'])
    stop_losses = len([t for t in portfolio.trade_history if t[2] == 'STOP_LOSS'])
    trailing_stops = len([t for t in portfolio.trade_history if t[2] == 'TRAILING_STOP'])
    
    # Display backtest results
    print("\nBACKTEST RESULTS:")
    
    metrics = {
        "Test Period": f"{len(common_dates) - min_seq_length} days",
        "Initial Balance": initial_balance,
        "Final Portfolio": final_portfolio,
        "P&L": test_pnl,
        "Return %": test_pnl_pct,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": max_dd,
        "Total Trades": num_trades,
        "Buys": buys,
        "Sells": sells,
        "Stop Losses": stop_losses,
        "Trailing Stops": trailing_stops
    }
    
    print("\n" + "="*80)
    print(f"{'METRIC':<25}{'VALUE':<15}")
    print("-"*80)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<25}{value:<15.2f}")
        else:
            print(f"{key:<25}{value:<15}")
    
    print("="*80 + "\n")
    
    # Calculate days to target
    if len(daily_returns) > 0:
        daily_return_rate = (1 + test_pnl_pct/100) ** (1 / len(daily_returns))
        
        # Based on current account situation
        current_equity = INITIAL_BALANCE + (CURRENT_CASH + INITIAL_BALANCE)
        
        if daily_return_rate > 1:
            days_to_target = np.log(TARGET_EQUITY / current_equity) / np.log(daily_return_rate)
            days_to_target = int(days_to_target) if not np.isnan(days_to_target) else "N/A"
        else:
            days_to_target = "N/A"
        
        # Print projection
        print("\nPROJECTED PATH TO $25,000:")
        print(f"Starting equity: ${current_equity:.2f}")
        print(f"Target equity: ${TARGET_EQUITY:.2f}")
        print(f"Daily return rate: {(daily_return_rate-1)*100:.4f}%")
        print(f"Estimated days to reach target: {days_to_target}")
    else:
        days_to_target = "N/A"
        
    # Plot portfolio performance
    plt.figure(figsize=(20, 12))
    
    # Plot portfolio value
    plt.subplot(2, 2, 1)
    plt.plot(portfolio_values)
    plt.title("Portfolio Value")
    plt.xlabel("Trading Day")
    plt.ylabel("Value ($)")
    
    # Plot daily returns
    plt.subplot(2, 2, 2)
    plt.hist(daily_returns * 100, bins=50)
    plt.title("Daily Returns Distribution")
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Frequency")
    
    # Plot trade counts by symbol
    plt.subplot(2, 2, 3)
    
    # Count trades by symbol
    symbol_trades = {}
    for _, symbol, action, _, _, _ in portfolio.trade_history:
        if symbol not in symbol_trades:
            symbol_trades[symbol] = {'BUY': 0, 'SELL': 0, 'STOP_LOSS': 0, 'TRAILING_STOP': 0}
        symbol_trades[symbol][action] += 1
    
    # Plot trades
    if symbol_trades:
        symbols = list(symbol_trades.keys())
        buy_counts = [symbol_trades[s]['BUY'] for s in symbols]
        sell_counts = [symbol_trades[s]['SELL'] for s in symbols]
        stop_loss_counts = [symbol_trades[s]['STOP_LOSS'] for s in symbols]
        trailing_stop_counts = [symbol_trades[s]['TRAILING_STOP'] for s in symbols]
        
        x = np.arange(len(symbols))
        width = 0.2
        
        plt.bar(x - 1.5*width, buy_counts, width, label='Buy')
        plt.bar(x - 0.5*width, sell_counts, width, label='Sell')
        plt.bar(x + 0.5*width, stop_loss_counts, width, label='Stop Loss')
        plt.bar(x + 1.5*width, trailing_stop_counts, width, label='Trailing Stop')
        
        plt.xlabel("Symbol")
        plt.ylabel("Number of Trades")
        plt.title("Trades by Symbol")
        plt.xticks(x, symbols)
        plt.legend()
    
    # Plot portfolio composition over time
    plt.subplot(2, 2, 4)
    plt.plot(portfolio.portfolio_value_history)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Trading Day")
    plt.ylabel("Value ($)")
    
    plt.tight_layout()
    plt.savefig("lstm_multi_position_results.png")
    print("Saved results chart to lstm_multi_position_results.png")
    
    return {
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
        'final_portfolio': final_portfolio,
        'return_pct': test_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'trades': num_trades,
        'days_to_target': days_to_target,
        'trade_history': portfolio.trade_history
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and backtest a multi-position LSTM trading portfolio')
    parser.add_argument('--symbols', type=str, nargs='+', default=FAST_GROWTH_SYMBOLS,
                       help='Stock symbols to include in the portfolio')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs per symbol')
    parser.add_argument('--balance', type=float, default=INITIAL_BALANCE, help='Initial balance')
    parser.add_argument('--positions', type=int, default=MAX_POSITIONS, help='Maximum number of positions')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--lstm-units', type=int, default=LSTM_UNITS, help='Number of LSTM units')
    args = parser.parse_args()
    
    # Update global variables based on arguments
    MAX_POSITIONS = args.positions
    POSITION_SIZE = 1.0 / MAX_POSITIONS
    BATCH_SIZE = args.batch_size
    LSTM_UNITS = args.lstm_units
    EPOCHS = args.epochs
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Print settings
    print("\n" + "="*80)
    print(f"LSTM MULTI-POSITION TRADING SYSTEM - PATH TO ${TARGET_EQUITY}")
    print("="*80)
    print(f"Initial balance: ${args.balance:.2f}")
    print(f"Maximum positions: {MAX_POSITIONS}")
    print(f"Position size: {POSITION_SIZE*100:.1f}% of portfolio")
    print(f"Stop loss: 5%, Trailing stop: 3%")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"LSTM configuration: {LSTM_UNITS} units, {BATCH_SIZE} batch size")
    print(f"Using 64GB RAM optimization for large scale model training")
    
    # Fetch data for all symbols
    all_data = fetch_historical_data(args.symbols)
    
    if not all_data:
        print("Error: No valid data found for any symbols")
        sys.exit(1)
    
    print(f"Successfully loaded data for {len(all_data)} symbols")
    
    # Split data into training and testing sets
    training_data = {}
    testing_data = {}
    
    for symbol, data in all_data.items():
        split_idx = int(len(data) * 0.6)
        training_data[symbol] = data.iloc[:split_idx].copy()
        testing_data[symbol] = data.iloc[split_idx:].copy()
        
        print(f"{symbol}: {len(training_data[symbol])} days for training, {len(testing_data[symbol])} days for testing")
    
    # Train LSTM models
    models, preprocessors = train_lstm_models(
        symbols=args.symbols,
        training_data=training_data
    )
    
    # Backtest portfolio
    results = backtest_lstm_portfolio(
        symbols=args.symbols,
        testing_data=testing_data,
        models=models, 
        preprocessors=preprocessors,
        initial_balance=args.balance
    )
    
    print("\nBacktest complete!")
    print(f"Final portfolio value: ${results['final_portfolio']:.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Total trades: {results['trades']}")
    print(f"Estimated days to $25K: {results['days_to_target']}") 