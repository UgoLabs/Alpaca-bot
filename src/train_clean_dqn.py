import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.dueling_dqn import DuelingDQNTrader, TradingEnvironment, DuelingDQNAgent
import logging
from datetime import datetime, timedelta
import yfinance as yf

# Configure logging with WARNING level to suppress INFO messages
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING to suppress INFO messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dueling_dqn_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep INFO level just for the main script

# Set models.dueling_dqn logger to WARNING level
dueling_dqn_logger = logging.getLogger("models.dueling_dqn")
dueling_dqn_logger.setLevel(logging.WARNING)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio from a series of returns"""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

def print_metrics_table(metrics):
    """Print a formatted table of metrics"""
    print("\n" + "="*80)
    print(f"{'METRIC':<20}{'VALUE':<15}")
    print("-"*80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<20}{value:<15.4f}")
        else:
            print(f"{key:<20}{value:<15}")
    print("="*80 + "\n")

def fetch_historical_data(symbol, period='5y'):
    """Direct function to fetch historical data without requiring the trader object"""
    print(f"Fetching {period} historical data for {symbol}")
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
        print(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def train_and_backtest(symbol, initial_balance=5222.58, epochs=100, lookback_window=20):
    """Train on first 3 years and backtest on last 2 years of data"""
    # Fetch 5 years of data
    print(f"\nFetching historical data for {symbol}...")
    data = fetch_historical_data(symbol)
    
    if data.empty or len(data) < 500:  # Need at least ~500 trading days for 2 years
        print(f"Insufficient data for {symbol}. Need at least 500 trading days.")
        return None
    
    # Split data: ~3 years for training, ~2 years for testing
    split_idx = int(len(data) * 0.6)  # 60% for training
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"Data split for {symbol}: {len(train_data)} days for training, {len(test_data)} days for testing")
    
    # Create agent directly
    state_size = int(os.getenv('TARGET_STATE_SIZE', 768))
    action_size = 3  # Hold, Buy, Sell
    
    agent = DuelingDQNAgent(state_size, action_size)
    
    # TRAINING PHASE
    print(f"Beginning training phase for {symbol} on {len(train_data)} days of data")
    
    # Create environment for training
    train_env = TradingEnvironment(
        stock_data=train_data,
        initial_balance=initial_balance,
        window_size=lookback_window
    )
    
    # Training metrics
    train_rewards = []
    portfolio_values = []
    best_reward = -np.inf
    daily_returns = []
    
    # Progress bar for training
    print(f"\nTraining {symbol} - {epochs} epochs on {len(train_data)} days")
    
    for episode in tqdm(range(epochs), desc="Training Progress"):
        state = train_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action
            action = agent.act(state, training=True)
            
            # Take action
            next_state, reward, done, info = train_env.step(action)
            
            # Store in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Train on batch
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
            
            # Store portfolio value at end of episode
            if done:
                portfolio_values.append(info['portfolio_value'])
                daily_return = info['portfolio_value'] / initial_balance - 1
                daily_returns.append(daily_return)
        
        # Update target model periodically
        if episode % 5 == 0:
            agent.update_target_model()
        
        # Store episode reward
        train_rewards.append(episode_reward)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f"models/{symbol}_dueling_dqn_best.weights.h5")
        
        # Display metrics every 5 epochs
        if episode % 5 == 0 and episode > 0:
            # Calculate metrics
            current_portfolio = portfolio_values[-1]
            pnl = current_portfolio - initial_balance
            pnl_pct = (pnl / initial_balance) * 100
            
            # Calculate Sharpe ratio if we have enough returns
            sharpe = calculate_sharpe_ratio(np.array(daily_returns)) if len(daily_returns) > 1 else 0
            
            metrics = {
                "Epoch": episode,
                "Reward": episode_reward,
                "Portfolio": current_portfolio,
                "P&L": pnl,
                "P&L %": pnl_pct,
                "Sharpe Ratio": sharpe
            }
            
            print_metrics_table(metrics)
    
    # BACKTESTING PHASE
    print(f"Beginning backtesting phase for {symbol} on {len(test_data)} days of data")
    
    # Create environment for testing
    test_env = TradingEnvironment(
        stock_data=test_data,
        initial_balance=initial_balance,
        window_size=lookback_window
    )
    
    # Load best model for testing
    agent.load(f"models/{symbol}_dueling_dqn_best.weights.h5")
    
    # Test metrics
    state = test_env.reset()
    done = False
    test_reward = 0
    actions_taken = []
    test_portfolio_values = [initial_balance]
    
    # Progress bar for testing
    print(f"\nBacktesting {symbol} on {len(test_data)} days")
    
    steps = 0
    pbar = tqdm(total=len(test_data) - lookback_window, desc="Backtest Progress")
    
    while not done:
        # Choose action (no random exploration during testing)
        action = agent.act(state, training=False)
        
        # Take action
        next_state, reward, done, info = test_env.step(action)
        
        # Update state and reward
        state = next_state
        test_reward += reward
        
        # Store action and portfolio value
        actions_taken.append(action)
        test_portfolio_values.append(info['portfolio_value'])
        
        steps += 1
        pbar.update(1)
    
    pbar.close()
    
    # Calculate test metrics
    final_portfolio = test_portfolio_values[-1]
    test_pnl = final_portfolio - initial_balance
    test_pnl_pct = (test_pnl / initial_balance) * 100
    
    # Calculate daily returns for Sharpe ratio
    test_returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
    test_sharpe = calculate_sharpe_ratio(test_returns) if len(test_returns) > 1 else 0
    
    # Count trades
    buys = actions_taken.count(1)
    sells = actions_taken.count(2)
    holds = actions_taken.count(0)
    
    # Buy and hold strategy return
    buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    
    # Display test results
    test_metrics = {
        "Test Period": f"{len(test_data)} days",
        "Final Portfolio": final_portfolio,
        "Test P&L": test_pnl,
        "Test P&L %": test_pnl_pct,
        "Buy & Hold %": buy_hold_return,
        "Outperformance": test_pnl_pct - buy_hold_return,
        "Sharpe Ratio": test_sharpe,
        "Total Reward": test_reward,
        "Buys": buys,
        "Sells": sells,
        "Holds": holds
    }
    
    print("\nBACKTEST RESULTS:")
    print_metrics_table(test_metrics)
    
    # Plot training and testing results
    plt.figure(figsize=(15, 10))
    
    # Plot training rewards
    plt.subplot(2, 2, 1)
    plt.plot(train_rewards)
    plt.title(f"{symbol} - Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # Plot training portfolio value
    plt.subplot(2, 2, 2)
    plt.plot(portfolio_values)
    plt.title(f"{symbol} - Training Portfolio Value")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value ($)")
    
    # Plot backtest portfolio value
    plt.subplot(2, 2, 3)
    plt.plot(test_portfolio_values)
    plt.title(f"{symbol} - Backtest Portfolio Value")
    plt.xlabel("Trading Day")
    plt.ylabel("Portfolio Value ($)")
    
    # Plot trading actions during backtest
    plt.subplot(2, 2, 4)
    plt.plot(test_data['Close'].values)
    plt.title(f"{symbol} - Price and Actions")
    plt.xlabel("Trading Day")
    plt.ylabel("Price ($)")
    
    # Mark buy/sell points
    for i, action in enumerate(actions_taken):
        if action == 1:  # Buy
            plt.scatter(i, test_data['Close'].values[i + lookback_window], color='green', marker='^')
        elif action == 2:  # Sell
            plt.scatter(i, test_data['Close'].values[i + lookback_window], color='red', marker='v')
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_training_results.png")
    
    return {
        'symbol': symbol,
        'train_rewards': train_rewards,
        'train_portfolio_values': portfolio_values,
        'test_reward': test_reward,
        'test_portfolio_value': final_portfolio,
        'test_return': test_pnl_pct,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': test_sharpe,
        'trades': (buys, sells, holds)
    }

def load_portfolio_symbols():
    """Load stock symbols from my_portfolio.txt"""
    try:
        with open('my_portfolio.txt', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        return symbols
    except FileNotFoundError:
        print("my_portfolio.txt not found")
        return []

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Initial account balance
    INITIAL_BALANCE = 5222.58
    
    try:
        print("Starting DuelingDQN training and testing with clean output...")
        
        # Load symbols from portfolio file
        all_symbols = load_portfolio_symbols()
        print(f"Loaded {len(all_symbols)} symbols from portfolio")
        
        # Get list of top volatile symbols that might yield quick profits
        print("Finding volatile symbols for quick trading...")
        symbol_volatility = []
        
        # Get volatility for each symbol
        for symbol in tqdm(all_symbols[:50], desc="Analyzing Volatility"):
            try:
                data = fetch_historical_data(symbol, period='1y')
                if not data.empty and len(data) > 60:  # At least 3 months of data
                    # Calculate daily returns
                    returns = data['Close'].pct_change().dropna()
                    # Calculate volatility (annualized standard deviation)
                    volatility = returns.std() * np.sqrt(252)
                    symbol_volatility.append((symbol, volatility))
            except Exception as e:
                continue
        
        # Sort by volatility (highest first)
        symbol_volatility.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 10 most volatile symbols
        symbols = [s[0] for s in symbol_volatility[:10]]
        
        print(f"\nSelected top 10 volatile symbols for training: {', '.join(symbols)}")
        
        # Results storage
        all_results = []
        
        # Train and backtest each symbol
        for symbol in symbols:
            try:
                print(f"\n{'='*40}")
                print(f"PROCESSING {symbol}")
                print(f"{'='*40}")
                
                # Train and backtest
                result = train_and_backtest(
                    symbol=symbol,
                    initial_balance=INITIAL_BALANCE,
                    epochs=50,
                    lookback_window=20  # 20 days lookback window as requested
                )
                
                if result:
                    all_results.append(result)
                    print(f"Completed {symbol} with test return: {result['test_return']:.2f}%")
                
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
        
        # Summary of results
        if all_results:
            # Sort by test return
            all_results.sort(key=lambda x: x['test_return'], reverse=True)
            
            # Display top performing symbols
            print("\nTOP PERFORMING SYMBOLS (Potential to reach $25,000):")
            print(f"{'Symbol':<10}{'Return %':<15}{'Sharpe':<15}{'vs Buy&Hold':<15}{'Trades (B/S/H)':<20}")
            print("-" * 75)
            
            for result in all_results:
                outperf = result['test_return'] - result['buy_hold_return']
                trades = result['trades']
                print(f"{result['symbol']:<10}{result['test_return']:<15.2f}{result['sharpe_ratio']:<15.2f}{outperf:<15.2f}{f'{trades[0]}/{trades[1]}/{trades[2]}':<20}")
            
            # Calculate how much each model would make with the current balance
            print("\nPROJECTED PROFITS WITH $5,222.58 INVESTMENT:")
            print(f"{'Symbol':<10}{'Return %':<15}{'Profit':<15}{'Days to $25K':<15}")
            print("-" * 55)
            
            current_balance = INITIAL_BALANCE
            target_equity = 25000
            
            for result in all_results:
                projected_profit = current_balance * (result['test_return'] / 100)
                projected_value = current_balance + projected_profit
                
                # Calculate days to reach $25K (assuming compounding and consistent returns)
                daily_return = (1 + result['test_return']/100) ** (1/len(result['train_portfolio_values'])) - 1
                if daily_return > 0 and projected_value < target_equity:
                    days_to_target = np.log(target_equity/projected_value) / np.log(1 + daily_return)
                    days_to_target = int(days_to_target) if not np.isnan(days_to_target) else "N/A"
                else:
                    days_to_target = "N/A" if projected_value < target_equity else "Already there!"
                
                print(f"{result['symbol']:<10}{result['test_return']:<15.2f}${projected_profit:<14.2f}{days_to_target:<15}")
            
            print(f"\nTraining and backtesting complete. Best symbol: {all_results[0]['symbol']} with {all_results[0]['test_return']:.2f}% return")
        else:
            print("No successful results")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}") 