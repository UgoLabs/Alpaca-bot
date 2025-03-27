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

# Silence TensorFlow and all warnings completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
import warnings
warnings.filterwarnings('ignore')

# Suppress ALL logs
logging.basicConfig(level=logging.CRITICAL)

# Silence all other modules
for module in ['tensorflow', 'matplotlib', 'yfinance', 'pandas', 'numpy', 'gym']:
    logging.getLogger(module).setLevel(logging.CRITICAL)

import yfinance as yf
from models.dueling_dqn import DuelingDQNAgent, TradingEnvironment

# Define constants
INITIAL_BALANCE = 5222.58
TARGET_EQUITY = 25000.00
CURRENT_CASH = -4717.79  # Current cash balance is negative
LOOKBACK_WINDOW = 20
MAX_POSITIONS = 5
POSITION_SIZE = 0.2  # 20% per position

# Pre-selected highly volatile stocks with good movement
FAST_GROWTH_SYMBOLS = [
    "PLTR", "NET", "NVDA", "TSLA", "INTC", "MSTR", "SMCI", 
    "META", "TSM", "NFLX", "AMZN", "MSFT"
]

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
    
    def get_portfolio_value(self, prices):
        """Calculate portfolio value based on current prices"""
        return self.balance + sum(
            position['shares'] * prices[symbol] 
            for symbol, position in self.positions.items() 
            if symbol in prices
        )
    
    def get_position_allocations(self, prices):
        """Get current position allocations as a dictionary"""
        portfolio_value = self.get_portfolio_value(prices)
        
        allocations = {'cash': self.balance / portfolio_value}
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position_value = position['shares'] * prices[symbol]
                allocations[symbol] = position_value / portfolio_value
        
        return allocations

def train_and_backtest_portfolio(symbols, training_data, testing_data, 
                                initial_balance=INITIAL_BALANCE, epochs=50):
    """Train agents for each symbol and backtest a multi-position portfolio"""
    # Create agents for each symbol
    agents = {}
    models = {}
    
    print(f"Training agents for {len(symbols)} symbols...")
    
    # Train an agent for each symbol
    for symbol in symbols:
        if symbol not in training_data:
            continue
            
        print(f"\n{'='*40}")
        print(f"TRAINING AGENT FOR {symbol}")
        print(f"{'='*40}")
        
        # Create agent
        state_size = int(os.getenv('TARGET_STATE_SIZE', 768))
        action_size = 3  # Hold, Buy, Sell
        
        agent = DuelingDQNAgent(state_size, action_size)
        
        # Create environment
        train_env = TradingEnvironment(
            stock_data=training_data[symbol],
            initial_balance=initial_balance,
            window_size=LOOKBACK_WINDOW,
            max_position=1.0,  # Allow using full balance in training
            stop_loss=0.05,    # 5% stop loss
            trailing_stop=0.03, # 3% trailing stop
            commission_fee=0.0  # No commission fee for Alpaca
        )
        
        # Training metrics
        train_rewards = []
        portfolio_values = []
        best_reward = -np.inf
        
        # Progress bar for training
        print(f"Training {epochs} epochs on {len(training_data[symbol])} days")
        
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
            
            # Update target model periodically
            if episode % 5 == 0:
                agent.update_target_model()
            
            # Store episode reward
            train_rewards.append(episode_reward)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(f"models/{symbol}_dueling_dqn_best.weights.h5")
                
        # Save the agent
        agents[symbol] = agent
        models[symbol] = {
            'train_rewards': train_rewards,
            'portfolio_values': portfolio_values,
            'best_reward': best_reward
        }
    
    # Now run the backtest with all agents
    print("\n" + "="*80)
    print("BACKTESTING MULTI-POSITION PORTFOLIO")
    print("="*80)
    
    # Get common date range for testing
    common_dates = None
    for symbol in testing_data:
        if common_dates is None:
            common_dates = set(testing_data[symbol].index)
        else:
            common_dates &= set(testing_data[symbol].index)
    
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
    
    # Create environments for each symbol
    test_envs = {}
    states = {}
    
    for symbol in agents:
        # Load best model
        agents[symbol].load(f"models/{symbol}_dueling_dqn_best.weights.h5")
        
        # Create environment
        test_envs[symbol] = TradingEnvironment(
            stock_data=testing_data[symbol],
            initial_balance=initial_balance,
            window_size=LOOKBACK_WINDOW,
            commission_fee=0.0  # No commission fee for Alpaca
        )
        
        # Reset environment
        states[symbol] = test_envs[symbol].reset()
    
    # Run backtest
    print("Running backtest...")
    
    portfolio_values = [initial_balance]
    action_history = {symbol: [] for symbol in agents}
    
    lookback_buffer = LOOKBACK_WINDOW  # Skip first LOOKBACK_WINDOW days
    
    for i, date in enumerate(tqdm(common_dates[lookback_buffer:], desc="Backtest Progress")):
        # Get current prices
        prices = {symbol: testing_data[symbol].loc[date, 'Close'] for symbol in testing_data if date in testing_data[symbol].index}
        
        # Update portfolio with current prices
        portfolio.update(date, prices)
        
        # Get recommendations from each agent
        recommendations = {}
        
        for symbol, agent in agents.items():
            if symbol not in test_envs or date not in testing_data[symbol].index:
                continue
                
            # Get current state
            state = states[symbol]
            
            # Get action recommendation
            action = agent.act(state, training=False)
            action_history[symbol].append(action)
            
            # Store recommendation
            recommendations[symbol] = action
            
            # Take a step in the environment (even if we don't follow recommendation)
            test_env = test_envs[symbol]
            
            # Advance environment one step
            next_state, _, _, _ = test_env.step(0)  # Just advance environment, don't take action
            states[symbol] = next_state
        
        # Determine which positions to sell (priority = 2)
        for symbol in list(portfolio.positions.keys()):
            if symbol in recommendations and recommendations[symbol] == 2:  # Sell
                portfolio.sell(symbol, prices[symbol], date)
        
        # Determine which positions to buy (priority = 1)
        # If we have available slots, try to fill them with buy recommendations
        if portfolio.can_open_position():
            # Sort symbols by highest reward that have buy recommendation
            buy_candidates = [(symbol, models[symbol]['best_reward']) 
                              for symbol in recommendations 
                              if recommendations[symbol] == 1 and symbol not in portfolio.positions]
            
            # If no buy recommendations, consider adding hold recommendations as candidates
            if not buy_candidates:
                buy_candidates = [(symbol, models[symbol]['best_reward']) 
                                 for symbol in recommendations 
                                 if recommendations[symbol] == 0 and symbol not in portfolio.positions]
            
            # Sort by highest reward
            buy_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Try to buy each candidate until we run out of slots or candidates
            for symbol, _ in buy_candidates:
                if portfolio.can_open_position() and symbol in prices:
                    portfolio.buy(symbol, prices[symbol], date)
        
        # Store portfolio value
        portfolio_value = portfolio.get_portfolio_value(prices)
        portfolio_values.append(portfolio_value)
    
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
    
    # Calculate average trade duration
    if len(portfolio.trade_history) > 0:
        # Group trades by symbol to calculate durations
        symbol_trades = {}
        for date, symbol, action, price, shares, value in portfolio.trade_history:
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append((date, action))
        
        # Calculate durations for completed trades
        durations = []
        for symbol, trades in symbol_trades.items():
            trades.sort(key=lambda x: x[0])  # Sort by date
            
            i = 0
            while i < len(trades) - 1:
                if trades[i][1] == 'BUY' and trades[i+1][1] in ['SELL', 'STOP_LOSS', 'TRAILING_STOP']:
                    duration = (trades[i+1][0] - trades[i][0]).days
                    durations.append(duration)
                i += 1
                
        avg_trade_duration = np.mean(durations) if durations else 0
    else:
        avg_trade_duration = 0
    
    # Calculate buy & hold performance for comparison
    buy_hold_returns = []
    for symbol in symbols:
        if symbol in testing_data and len(testing_data[symbol]) > 0:
            symbol_data = testing_data[symbol]
            initial_price = symbol_data.iloc[lookback_buffer]['Close']
            final_price = symbol_data.iloc[-1]['Close']
            symbol_return = (final_price / initial_price - 1) * 100
            buy_hold_returns.append(symbol_return)
    
    avg_buy_hold = np.mean(buy_hold_returns) if buy_hold_returns else 0
    
    # Display backtest results
    print("\nBACKTEST RESULTS:")
    
    metrics = {
        "Test Period": f"{len(common_dates) - lookback_buffer} days",
        "Initial Balance": initial_balance,
        "Final Portfolio": final_portfolio,
        "P&L": test_pnl,
        "Return %": test_pnl_pct,
        "Buy & Hold Avg %": avg_buy_hold,
        "Outperformance": test_pnl_pct - avg_buy_hold,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": max_dd,
        "Total Trades": num_trades,
        "Buys": buys,
        "Sells": sells,
        "Stop Losses": stop_losses,
        "Trailing Stops": trailing_stops,
        "Avg Trade Duration": avg_trade_duration
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
    daily_return_rate = (1 + test_pnl_pct/100) ** (1 / len(daily_returns))
    
    # Based on current account situation
    current_equity = INITIAL_BALANCE + (CURRENT_CASH + INITIAL_BALANCE)
    
    if daily_return_rate > 1:
        days_to_target = np.log(TARGET_EQUITY / current_equity) / np.log(daily_return_rate)
        days_to_target = int(days_to_target) if not np.isnan(days_to_target) else "N/A"
    else:
        days_to_target = "N/A"
    
    # Compound growth projection
    print("\nPROJECTED PATH TO $25,000:")
    print(f"Starting equity: ${current_equity:.2f}")
    print(f"Target equity: ${TARGET_EQUITY:.2f}")
    print(f"Daily return rate: {(daily_return_rate-1)*100:.4f}%")
    print(f"Estimated days to reach target: {days_to_target}")
    
    if days_to_target != "N/A" and days_to_target <= 365:
        # Show monthly projection for the next year
        projected_equity = current_equity
        print("\nMONTHLY GROWTH PROJECTION:")
        print(f"{'Month':<10}{'Equity':<15}")
        print("-" * 25)
        
        for month in range(1, 13):
            projected_equity *= daily_return_rate ** 21  # ~21 trading days per month
            print(f"{month:<10}${projected_equity:<14.2f}")
            if projected_equity >= TARGET_EQUITY:
                print(f"\nTarget reached in month {month}")
                break
    
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
    
    # Plot position allocations over time
    plt.subplot(2, 2, 3)
    
    # Extract position history from trade history
    dates = common_dates[lookback_buffer:]
    positions_over_time = {}
    current_positions = {}
    
    for i, date in enumerate(dates):
        # Update positions based on trades for this date
        for trade_date, symbol, action, price, shares, value in portfolio.trade_history:
            if trade_date == date:
                if action == 'BUY':
                    current_positions[symbol] = value
                elif action in ['SELL', 'STOP_LOSS', 'TRAILING_STOP']:
                    if symbol in current_positions:
                        del current_positions[symbol]
        
        # Store positions for this date
        for symbol, value in current_positions.items():
            if symbol not in positions_over_time:
                positions_over_time[symbol] = [0] * len(dates)
            positions_over_time[symbol][i] = value
    
    # Plot stacked positions
    bottom = np.zeros(len(dates))
    for symbol, values in positions_over_time.items():
        plt.bar(range(len(dates)), values, bottom=bottom, label=symbol, alpha=0.7)
        bottom += np.array(values)
    
    plt.title("Position Allocation Over Time")
    plt.xlabel("Trading Day")
    plt.ylabel("Value ($)")
    plt.legend()
    
    # Plot trade counts by symbol
    plt.subplot(2, 2, 4)
    
    # Count trades by symbol
    symbol_trades = {}
    for _, symbol, action, _, _, _ in portfolio.trade_history:
        if symbol not in symbol_trades:
            symbol_trades[symbol] = {'BUY': 0, 'SELL': 0, 'STOP_LOSS': 0, 'TRAILING_STOP': 0}
        symbol_trades[symbol][action] += 1
    
    # Plot trades
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
    
    plt.tight_layout()
    plt.savefig("multi_position_backtest_results.png")
    print("Saved results chart to multi_position_backtest_results.png")
    
    return {
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
        'final_portfolio': final_portfolio,
        'return_pct': test_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'trades': num_trades,
        'avg_trade_duration': avg_trade_duration,
        'days_to_target': days_to_target,
        'trade_history': portfolio.trade_history
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and backtest a multi-position trading portfolio')
    parser.add_argument('--symbols', type=str, nargs='+', default=FAST_GROWTH_SYMBOLS,
                       help='Stock symbols to include in the portfolio')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs per symbol')
    parser.add_argument('--balance', type=float, default=INITIAL_BALANCE, help='Initial balance')
    parser.add_argument('--positions', type=int, default=MAX_POSITIONS, help='Maximum number of positions')
    args = parser.parse_args()
    
    # Update global variables based on arguments
    MAX_POSITIONS = args.positions
    POSITION_SIZE = 1.0 / MAX_POSITIONS
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Print settings
    print("\n" + "="*80)
    print(f"MULTI-POSITION TRADING SYSTEM - PATH TO ${TARGET_EQUITY}")
    print("="*80)
    print(f"Initial balance: ${args.balance:.2f}")
    print(f"Maximum positions: {MAX_POSITIONS}")
    print(f"Position size: {POSITION_SIZE*100:.1f}% of portfolio")
    print(f"Stop loss: 5%, Trailing stop: 3%")
    print(f"Symbols: {', '.join(args.symbols)}")
    
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
    
    # Train and backtest
    results = train_and_backtest_portfolio(
        symbols=args.symbols,
        training_data=training_data,
        testing_data=testing_data,
        initial_balance=args.balance,
        epochs=args.epochs
    )
    
    print("\nBacktest complete!")
    print(f"Final portfolio value: ${results['final_portfolio']:.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Total trades: {results['trades']}")
    print(f"Estimated days to $25K: {results['days_to_target']}") 