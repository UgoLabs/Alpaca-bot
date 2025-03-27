import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import random
import time
from datetime import datetime, timedelta
import gym
from gym import spaces
import yfinance as yf
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_dqn.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure TensorFlow for GPU
if os.getenv('USE_GPU', 'true').lower() == 'true':
    gpu_memory_fraction = float(os.getenv('GPU_MEMORY_FRACTION', '0.9'))
    mixed_precision = os.getenv('USE_MIXED_PRECISION', 'true').lower() == 'true'
    
    # List all available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info(f"Available GPUs: {gpus}")
    
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Set memory growth for GPU: {gpu}")
            
            # Set visible devices to use first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logger.info(f"Set visible device to: {gpus[0]}")
            
            # Configure memory limit for RTX 4070 (12GB VRAM)
            memory_limit = int(gpu_memory_fraction * 12 * 1024)  # Convert to MB
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )
            logger.info(f"Set memory limit to {memory_limit}MB ({gpu_memory_fraction*100}% of 12GB)")
            
            # Enable TF32 precision for better performance
            tf.config.experimental.enable_tensor_float_32_execution(True)
            logger.info("Enabled TF32 precision")
            
            # Enable mixed precision training
            if mixed_precision:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Enabled mixed precision (FP16) training")
            
            # Verify GPU is being used
            logger.info(f"TensorFlow is using GPU: {tf.config.get_visible_devices('GPU')}")
            logger.info(f"Current GPU memory usage: {tf.config.experimental.get_memory_info('GPU:0')}")
                
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPU found, using CPU")

class TradingEnvironment(gym.Env):
    """Custom trading environment that follows gym interface"""
    
    def __init__(self, stock_data, initial_balance=None, max_steps=252, window_size=20, 
                 commission_fee=0.001, max_position=None, stop_loss=None, trailing_stop=None):
        super(TradingEnvironment, self).__init__()
        
        # Ensure stock_data has all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in stock_data.columns:
                if col == 'Volume':
                    # Add dummy volume if missing
                    stock_data['Volume'] = 1000000
                else:
                    raise ValueError(f"Stock data missing required column: {col}")
        
        # Make sure data is sorted by date
        if 'Date' in stock_data.columns:
            stock_data = stock_data.sort_values('Date')
        
        self.stock_data = stock_data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_steps = max_steps
        self.window_size = window_size
        self.commission_fee = commission_fee
        self.max_position = max_position or float(os.getenv('MAX_POSITION_SIZE', 0.25))
        self.stop_loss = stop_loss or float(os.getenv('STOP_LOSS_PERCENTAGE', 0.05))
        self.trailing_stop = trailing_stop or float(os.getenv('TRAILING_STOP_PERCENTAGE', 0.03))
        
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # State space: price data + indicators + portfolio state
        self.state_size = int(os.getenv('TARGET_STATE_SIZE', 768))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )
        
        self.reset()
        
    def _get_observation(self):
        """Get the current state observation"""
        idx = self.current_step
        
        # Ensure we have enough history
        if idx < self.window_size:
            raise ValueError(f"Not enough history to create observation at step {idx}")
            
        # Get price and volume history
        prices = self.stock_data['Close'].values[idx-self.window_size:idx].astype(np.float32)
        volumes = self.stock_data['Volume'].values[idx-self.window_size:idx].astype(np.float32)
        
        # Normalize data
        if len(prices) > 0 and prices[-1] != 0:
            prices_norm = prices / prices[-1] - 1.0  # Normalize to percentage change
        else:
            prices_norm = np.zeros_like(prices)
            
        max_volume = np.max(volumes) if np.max(volumes) > 0 else 1.0
        volumes_norm = volumes / max_volume
        
        # Technical indicators (placeholder)
        # This is where you could add more features like RSI, MACD, etc.
        
        # Portfolio state
        portfolio_state = np.array([
            self.balance / self.initial_balance - 1.0,  # Normalized balance
            self.shares_held * self.current_price / self.initial_balance if self.initial_balance > 0 else 0.0,  # Position size
            1.0 if self.shares_held > 0 else 0.0,  # Long indicator
            1.0 if self.shares_held < 0 else 0.0,  # Short indicator
            self.cost_basis / self.current_price - 1.0 if (self.cost_basis > 0 and self.current_price > 0) else 0.0,  # Entry price relative to current
        ], dtype=np.float32)
        
        # Combine all features
        features = np.concatenate([prices_norm, volumes_norm, portfolio_state])
        
        # Ensure observation matches the required state size
        if len(features) > self.state_size:
            # Truncate if too long
            return features[:self.state_size].astype(np.float32)
        else:
            # Pad with zeros if too short
            padded = np.zeros(self.state_size, dtype=np.float32)
            padded[:len(features)] = features
            return padded
        
    def _calculate_reward(self, action):
        """Calculate reward for the current action"""
        step_reward = 0
        
        # Current portfolio value
        current_portfolio_value = self.balance + self.shares_held * self.current_price
        
        # Reward based on portfolio value change
        portfolio_change = current_portfolio_value - self.previous_portfolio_value
        step_reward += portfolio_change / self.initial_balance * 100  # Scale to percentage
        
        # Add reward multipliers based on action type
        if action == 1:  # Buy
            step_reward *= float(os.getenv('BUY_REWARD_MULTIPLIER', 1.5))
        elif action == 2:  # Sell
            step_reward *= float(os.getenv('SELL_PENALTY_FACTOR', 0.8))
        
        # Penalize for not utilizing capital
        if self.shares_held == 0:
            step_reward -= 0.01  # Small penalty for sitting idle
            
        # Update previous portfolio value
        self.previous_portfolio_value = current_portfolio_value
        
        # Check if we hit stop loss (additional penalty)
        if self.hit_stop_loss:
            step_reward -= 1.0
            self.hit_stop_loss = False
            
        return step_reward
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Get current price
        close_val = self.stock_data['Close'].iloc[self.current_step]
        self.current_price = float(close_val.iloc[0]) if hasattr(close_val, 'iloc') else float(close_val)
        
        # Get previous price
        prev_close_val = self.stock_data['Close'].iloc[self.current_step - 1]
        previous_price = float(prev_close_val.iloc[0]) if hasattr(prev_close_val, 'iloc') else float(prev_close_val)
        
        # Set hit_stop_loss flag to False initially
        self.hit_stop_loss = False
        
        # Execute trading action
        if action == 0:  # Hold
            pass
        
        elif action == 1:  # Buy
            if self.balance > 0:
                # Calculate maximum shares to buy
                max_affordable = self.balance / (self.current_price * (1 + self.commission_fee))
                max_position_shares = self.initial_balance * self.max_position / self.current_price
                shares_to_buy = min(max_affordable, max_position_shares)
                
                if shares_to_buy > 0:
                    # Update portfolio
                    cost = shares_to_buy * self.current_price * (1 + self.commission_fee)
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    self.cost_basis = self.current_price
                    self.highest_price_since_buy = self.current_price
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Sell all shares
                self.balance += self.shares_held * self.current_price * (1 - self.commission_fee)
                self.shares_held = 0
                self.cost_basis = 0
                self.highest_price_since_buy = 0
        
        # Check stop loss if we have a position
        if self.shares_held > 0:
            # Update highest price since buy for trailing stop
            if self.current_price > self.highest_price_since_buy:
                self.highest_price_since_buy = self.current_price
                
            # Check regular stop loss
            if self.current_price < self.cost_basis * (1 - self.stop_loss):
                # Sell all shares
                self.balance += self.shares_held * self.current_price * (1 - self.commission_fee)
                self.shares_held = 0
                self.cost_basis = 0
                self.hit_stop_loss = True
                logger.info(f"Stop loss triggered at step {self.current_step}")
                
            # Check trailing stop
            elif self.current_price < self.highest_price_since_buy * (1 - self.trailing_stop):
                # Sell all shares
                self.balance += self.shares_held * self.current_price * (1 - self.commission_fee)
                self.shares_held = 0
                self.cost_basis = 0
                self.hit_stop_loss = True
                logger.info(f"Trailing stop triggered at step {self.current_step}")
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self.current_step >= len(self.stock_data) - 1
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': self.balance + self.shares_held * self.current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'price_change': self.current_price / previous_price - 1
        }
        
        return next_obs, reward, done, info
    
    def reset(self):
        """Reset the environment"""
        # Ensure we start with enough history for observation window
        self.current_step = self.window_size
        
        # If we don't have enough data, raise error
        if len(self.stock_data) <= self.window_size:
            raise ValueError(f"Not enough data points in stock_data. Need at least {self.window_size+1} data points.")
            
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.highest_price_since_buy = 0
        self.hit_stop_loss = False
        
        # Handle the case if Close is a pandas Series
        close_val = self.stock_data['Close'].iloc[self.current_step]
        self.current_price = float(close_val.iloc[0]) if hasattr(close_val, 'iloc') else float(close_val)
        self.previous_portfolio_value = self.balance
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the environment state"""
        profit = self.balance - self.initial_balance + self.shares_held * self.current_price
        logger.info(f"Step: {self.current_step}, Price: ${self.current_price:.2f}, "
                   f"Balance: ${self.balance:.2f}, Shares: {self.shares_held:.4f}, "
                   f"P/L: ${profit:.2f} ({profit/self.initial_balance*100:.2f}%)")

class DuelingDQNAgent:
    """Dueling DQN Agent for trading"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Agent parameters (from .env)
        self.gamma = float(os.getenv('GAMMA', 0.99))
        self.epsilon = float(os.getenv('INITIAL_EXPLORATION', 1.0))
        self.epsilon_min = float(os.getenv('FINAL_EXPLORATION', 0.1))
        self.epsilon_decay = 0.995  # Calculated based on epochs
        self.learning_rate = float(os.getenv('LEARNING_RATE', 5e-5))
        self.batch_size = int(os.getenv('BATCH_SIZE', 512))
        self.hidden_size = int(os.getenv('HIDDEN_SIZE', 512))
        
        # Memory for experience replay
        self.memory = deque(maxlen=100000)
        
        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Track training metrics
        self.loss_history = []
        
    def _build_model(self):
        """Build the dueling DQN model"""
        # Input layer
        input_layer = Input(shape=(self.state_size,))
        
        # Shared network
        shared = Dense(self.hidden_size, activation='relu')(input_layer)
        shared = Dense(self.hidden_size, activation='relu')(shared)
        
        # Value stream
        value_stream = Dense(self.hidden_size//2, activation='relu')(shared)
        value = Dense(1)(value_stream)
        
        # Advantage stream
        advantage_stream = Dense(self.hidden_size//2, activation='relu')(shared)
        advantage = Dense(self.action_size)(advantage_stream)
        
        # Combine value and advantage streams
        outputs = Lambda(lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
                        output_shape=(self.action_size,))([value, advantage])
        
        # Build and compile model
        model = Model(inputs=input_layer, outputs=outputs)
        
        # Use tf.keras.losses.Huber() instead of 'huber_loss' string
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """Update target model weights with current model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=None):
        """Train the model with experiences from memory"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0  # Not enough samples
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract states and predict Q values
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        done_flags = np.array([experience[4] for experience in minibatch])
        
        # Predict Q values for current states
        targets = self.model.predict(states, verbose=0)
        
        # Predict Q values for next states with target model
        target_next = self.target_model.predict(next_states, verbose=0)
        
        # Update target values
        for i in range(batch_size):
            if done_flags[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
        
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
        self.update_target_model()
        
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)

class DuelingDQNTrader:
    """Trading system using Dueling DQN"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Alpaca API
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.base_url = os.getenv('ALPACA_API_BASE_URL')
        
        if not all([self.api_key, self.api_secret, self.base_url]):
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
        
        # Get account information
        self.account = self.api.get_account()
        self.initial_balance = float(self.account.equity)
        logger.info(f"Account balance: ${self.initial_balance}")
        
        # Load portfolio stocks
        self.portfolio_symbols = self._load_portfolio_symbols()
        logger.info(f"Loaded {len(self.portfolio_symbols)} symbols from portfolio")
        
        # Training parameters
        self.epochs = int(os.getenv('EPOCHS', 100))
        self.state_size = int(os.getenv('TARGET_STATE_SIZE', 768))
        self.action_size = 3  # Hold, Buy, Sell
        
        # Create agent
        self.agent = DuelingDQNAgent(self.state_size, self.action_size)
        
    def _load_portfolio_symbols(self):
        """Load stock symbols from my_portfolio.txt"""
        try:
            with open('my_portfolio.txt', 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            return symbols
        except FileNotFoundError:
            logger.error("my_portfolio.txt not found")
            return []
    
    def _fetch_historical_data(self, symbol, period='5y'):
        """Fetch historical data from yfinance"""
        logger.info(f"Fetching {period} historical data for {symbol}")
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
                
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def train(self, symbol, data=None, episodes=None):
        """Train the agent on historical data for a single symbol"""
        if data is None:
            data = self._fetch_historical_data(symbol)
            
        if data.empty or len(data) < 30:  # Need at least 30 data points
            logger.error(f"No data or insufficient data available for {symbol}")
            return None
            
        episodes = episodes or self.epochs
        
        # Print data shape for debugging
        logger.info(f"Data shape for {symbol}: {data.shape}, columns: {data.columns.tolist()}")
        
        try:
            # Create environment
            env = TradingEnvironment(
                stock_data=data,
                initial_balance=self.initial_balance,
                window_size=20
            )
            
            # Training loop
            total_rewards = []
            best_reward = -np.inf
            
            for episode in range(episodes):
                state = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    # Choose action
                    action = self.agent.act(state)
                    
                    # Take action
                    next_state, reward, done, info = env.step(action)
                    
                    # Remember experience
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Update state
                    state = next_state
                    total_reward += reward
                    
                    # Replay experiences
                    if len(self.agent.memory) > self.agent.batch_size:
                        self.agent.replay()
                
                # Update target model after each episode
                if episode % 5 == 0:
                    self.agent.update_target_model()
                    
                # Log progress
                total_rewards.append(total_reward)
                
                # Save best model
                if total_reward > best_reward:
                    best_reward = total_reward
                    self.agent.save(f"models/{symbol}_dueling_dqn_best.h5")
                    
                # Log episode results
                portfolio_value = info['portfolio_value']
                returns = (portfolio_value / self.initial_balance - 1) * 100
                logger.info(f"Episode: {episode+1}/{episodes}, Reward: {total_reward:.2f}, "
                        f"Return: {returns:.2f}%, Epsilon: {self.agent.epsilon:.4f}")
                
                # Early stopping if we achieve good returns
                if returns > 30 and episode > episodes // 2:
                    logger.info(f"Early stopping at episode {episode+1} with return {returns:.2f}%")
                    break
                    
            # Save final model
            self.agent.save(f"models/{symbol}_dueling_dqn_final.h5")
            
            return {
                'symbol': symbol,
                'total_rewards': total_rewards,
                'best_reward': best_reward,
                'final_portfolio_value': portfolio_value,
                'return': returns
            }
        except Exception as e:
            logger.error(f"Error training on {symbol}: {str(e)}")
            return None
    
    def train_portfolio(self):
        """Train the agent on all symbols in the portfolio"""
        results = []
        
        for symbol in self.portfolio_symbols:
            try:
                # Fetch data
                data = self._fetch_historical_data(symbol)
                
                if data.empty:
                    logger.warning(f"Skipping {symbol} due to empty data")
                    continue
                    
                # Train agent
                logger.info(f"Training on {symbol}")
                result = self.train(symbol, data)
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error training on {symbol}: {str(e)}")
                
        # Analyze results
        if results:
            # Find best performing model
            best_symbol = max(results, key=lambda x: x['return'])['symbol']
            logger.info(f"Best performing model: {best_symbol}")
            
            # Load best model
            self.agent.load(f"models/{best_symbol}_dueling_dqn_best.h5")
            
        return results

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Initialize trader
    trader = DuelingDQNTrader()
    
    # Train on portfolio
    results = trader.train_portfolio()
    
    # Log training summary
    if results:
        avg_return = np.mean([r['return'] for r in results])
        max_return = np.max([r['return'] for r in results])
        
        logger.info(f"Training complete. Average return: {avg_return:.2f}%, Max return: {max_return:.2f}%")
    else:
        logger.warning("No successful training results")
