"""
Debug script to verify bot is working correctly
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

# Import from bot
sys.path.insert(0, os.path.dirname(__file__))
from bot_standalone import add_technical_indicators, normalize_state, PaperTrader
from swing_model import DuelingDQN

def debug_symbol(symbol='SPY'):
    print(f"\n{'='*60}")
    print(f"🔍 DEBUGGING {symbol}")
    print(f"{'='*60}\n")
    
    # Initialize API
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    
    # Get data
    end = datetime.now()
    start = end - timedelta(days=300)
    print(f"📊 Fetching data from {start.date()} to {end.date()}...")
    
    try:
        bars = api.get_bars(symbol, '1Day', start=start.isoformat(), end=end.isoformat(), limit=300, feed='iex').df
        print(f"✅ Got {len(bars)} bars")
        print(f"   Latest date: {bars.index[-1]}")
        print(f"   Latest close: ${bars['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    # Add indicators
    print(f"\n📈 Computing indicators...")
    df = add_technical_indicators(bars)
    
    # Show key indicators
    latest = df.iloc[-1]
    print(f"\n📊 Current Indicators:")
    print(f"   RSI: {latest['rsi']:.1f}")
    print(f"   ADX: {latest['adx']:.1f}")
    print(f"   Price vs SMA20: {latest['price_vs_sma20']*100:.2f}%")
    print(f"   Price vs SMA50: {latest['price_vs_sma50']*100:.2f}%")
    print(f"   Volume Ratio: {latest['volume_ratio']:.2f}x")
    print(f"   BB %B: {latest['bb_pband']:.2f}")
    
    # Construct state
    print(f"\n🧮 Constructing state vector...")
    current_step = len(df) - 1
    market_state = normalize_state(df, current_step)
    print(f"   Market state size: {len(market_state)}")
    
    # Add dummy portfolio state
    portfolio_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    state = np.concatenate((market_state, portfolio_state))
    print(f"   Total state size: {len(state)}")
    
    # Load model
    print(f"\n🤖 Loading model...")
    state_size = 231
    agent = DuelingDQN(state_size, 3, use_noisy=True)
    agent.load('models/live_model.pth')
    agent.model.eval()
    print(f"   Model loaded successfully")
    
    # Get raw Q-values
    print(f"\n🎯 Getting model predictions...")
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = agent.model(state_tensor)
        print(f"   Raw Q-values: {q_values.numpy()[0]}")
        print(f"   HOLD: {q_values[0][0].item():.4f}")
        print(f"   BUY:  {q_values[0][1].item():.4f}")
        print(f"   SELL: {q_values[0][2].item():.4f}")
    
    # Get action
    action = agent.act(state, epsilon=0.0)
    action_name = ['HOLD', 'BUY', 'SELL'][action]
    print(f"\n   ✅ Final Action: {action_name}")
    
    # Analysis
    print(f"\n📋 Analysis:")
    if action == 0:
        print(f"   Model chose HOLD because:")
        if q_values[0][0] > q_values[0][1]:
            print(f"   - HOLD Q-value ({q_values[0][0].item():.4f}) > BUY ({q_values[0][1].item():.4f})")
        if latest['adx'] < 20:
            print(f"   - Low ADX ({latest['adx']:.1f}) suggests choppy market")
        if latest['rsi'] > 70:
            print(f"   - Overbought RSI ({latest['rsi']:.1f})")
        if latest['volume_ratio'] < 1.5:
            print(f"   - Low volume ({latest['volume_ratio']:.2f}x)")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    symbols = ['SPY', 'AAPL', 'NVDA']
    for sym in symbols:
        try:
            debug_symbol(sym)
        except Exception as e:
            print(f"❌ Error debugging {sym}: {e}\n")
