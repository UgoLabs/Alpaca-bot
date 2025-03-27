import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.dueling_dqn import DuelingDQNTrader, TradingEnvironment
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_model(symbol, model_path, test_period='1y'):
    """Evaluate a trained model on test data"""
    # Initialize trader
    trader = DuelingDQNTrader()
    
    # Load the model
    trader.agent.load(model_path)
    
    # Get test data
    test_data = trader._fetch_historical_data(symbol, period=test_period)
    
    if test_data.empty:
        logger.error(f"No test data available for {symbol}")
        return None
    
    # Create test environment
    test_env = TradingEnvironment(
        stock_data=test_data,
        initial_balance=trader.initial_balance,
        window_size=20
    )
    
    # Evaluate model
    state = test_env.reset()
    done = False
    total_reward = 0
    actions_taken = []
    portfolio_values = []
    
    while not done:
        # Choose action - use lower epsilon for evaluation
        action = trader.agent.act(state, training=False)
        
        # Take action
        next_state, reward, done, info = test_env.step(action)
        
        # Update state
        state = next_state
        total_reward += reward
        
        # Store information
        actions_taken.append(action)
        portfolio_values.append(info['portfolio_value'])
        
        # Log progress
        if len(portfolio_values) % 20 == 0:
            logger.info(f"Step: {len(portfolio_values)}, Portfolio Value: ${portfolio_values[-1]:.2f}")
    
    # Calculate final return
    final_return = (portfolio_values[-1] / trader.initial_balance - 1) * 100
    
    # Calculate buy and hold return
    buy_and_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    
    # Log results
    logger.info(f"Evaluation complete for {symbol}")
    logger.info(f"Total reward: {total_reward:.2f}")
    logger.info(f"Final portfolio value: ${portfolio_values[-1]:.2f}")
    logger.info(f"Return: {final_return:.2f}%")
    logger.info(f"Buy and hold return: {buy_and_hold_return:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    
    # Mark buy/sell actions
    for i, action in enumerate(actions_taken):
        if action == 1:  # Buy
            plt.scatter(i, portfolio_values[i], color='green', marker='^')
        elif action == 2:  # Sell
            plt.scatter(i, portfolio_values[i], color='red', marker='v')
    
    plt.title(f"{symbol} - Portfolio Value Over Time")
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{symbol}_evaluation.png")
    
    return {
        'symbol': symbol,
        'total_reward': total_reward,
        'final_portfolio_value': portfolio_values[-1],
        'return': final_return,
        'buy_and_hold_return': buy_and_hold_return
    }


def main():
    """Main function to evaluate a trained model"""
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN model')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to evaluate')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--period', type=str, default='1y', help='Test period (default: 1y)')
    
    args = parser.parse_args()
    
    # If model path not provided, use default path
    model_path = args.model
    if not model_path:
        model_path = f"models/{args.symbol}_dueling_dqn_best.h5"
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Evaluate model
    result = evaluate_model(args.symbol, model_path, args.period)
    
    if result:
        logger.info("Evaluation successful")
    else:
        logger.error("Evaluation failed")


if __name__ == "__main__":
    main() 