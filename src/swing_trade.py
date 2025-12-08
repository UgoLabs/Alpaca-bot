import os
import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from datetime import datetime, timedelta
from utils import add_technical_indicators, normalize_state
from swing_model import DuelingDQN

# Load Environment Variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_API_BASE_URL')

# Configuration
SYMBOLS = ['SPY', 'AAPL', 'MSFT', 'TSLA', 'NVDA'] # Add your portfolio symbols here
MODEL_PATH = "models/SPY_dqn_final.h5" # Use the trained model (Generalize if needed)
WINDOW_SIZE = 20
CASH_PER_TRADE = 1000 # Amount to allocate per trade

def get_alpaca_data(api, symbol, days=100):
    """
    Fetches daily bars from Alpaca.
    """
    # Calculate start date
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    try:
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Day, start=start_date).df
        if bars.empty:
            return None
        
        # Rename columns to match our utils
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

def trade_bot():
    print("Initializing Swing Trading Bot...")
    
    # 1. Connect to Alpaca
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = api.get_account()
    print(f"Account Status: {account.status}")
    print(f"Buying Power: ${account.buying_power}")
    
    # 2. Load Model
    # Note: We need to know the state size. 
    # From environment: (Window * 7) + 4
    state_size = (WINDOW_SIZE * 7) + 4
    action_size = 3
    
    agent = DuelingDQN(state_size, action_size)
    try:
        agent.load(MODEL_PATH)
        print("Model loaded successfully.")
    except:
        print(f"Could not load model from {MODEL_PATH}. Please train first.")
        return

    # 3. Trading Loop (Run once per call, user can schedule this script)
    print(f"Checking signals for: {SYMBOLS}")
    
    for symbol in SYMBOLS:
        print(f"\nAnalyzing {symbol}...")
        
        # Get Data
        df = get_alpaca_data(api, symbol)
        if df is None or len(df) < WINDOW_SIZE + 5:
            print(f"Not enough data for {symbol}. Skipping.")
            continue
            
        # Add Indicators
        df = add_technical_indicators(df)
        
        # Prepare State
        # We want the state for the *latest* complete candle
        current_step = len(df) - 1 
        market_state = normalize_state(df, current_step, WINDOW_SIZE)
        
        # Get Portfolio State for this symbol
        try:
            position = api.get_position(symbol)
            qty = float(position.qty)
            entry_price = float(position.avg_entry_price)
            current_price = float(position.current_price)
            market_value = float(position.market_value)
            unrealized_pnl = (current_price - entry_price) / entry_price
            in_position = 1.0
        except:
            # No position
            qty = 0
            entry_price = 0
            current_price = df['Close'].iloc[-1]
            market_value = 0
            unrealized_pnl = 0
            in_position = 0.0
            
        # Portfolio state vector
        # [Balance_Norm, Position_Size_Norm, PnL, In_Position]
        # Note: We approximate Balance_Norm as 1.0 for simplicity in live inference 
        # or fetch real balance.
        equity = float(account.equity)
        portfolio_state = np.array([
            float(account.cash) / equity,
            market_value / equity,
            unrealized_pnl,
            in_position
        ])
        
        state = np.concatenate((market_state, portfolio_state))
        
        # Get Action
        action = agent.act(state, epsilon=0.0) # No exploration in live trading
        
        print(f"Model Prediction: {['HOLD', 'BUY', 'SELL'][action]}")
        
        # Execute Trade
        if action == 1: # BUY
            if qty == 0:
                print(f"Placing BUY order for {symbol}...")
                # Calculate shares
                shares = int(CASH_PER_TRADE / current_price)
                if shares > 0:
                    try:
                        api.submit_order(
                            symbol=symbol,
                            qty=shares,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        print(f"Buy order submitted for {shares} shares.")
                    except Exception as e:
                        print(f"Order failed: {e}")
            else:
                print("Already in position. Holding.")
                
        elif action == 2: # SELL
            if qty > 0:
                print(f"Placing SELL order for {symbol}...")
                try:
                    api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    print("Sell order submitted.")
                except Exception as e:
                    print(f"Order failed: {e}")
            else:
                print("No position to sell.")
        
        else: # HOLD
            print("Holding.")

if __name__ == "__main__":
    trade_bot()
