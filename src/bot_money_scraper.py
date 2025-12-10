"""
AI-Powered Money Scraper Bot
Uses Dueling DQN model to pick entries, exits at +$5 profit or -$2 loss
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from swing_model import DuelingDQN
from utils import add_technical_indicators, normalize_state
import torch
import pickle
from collections import deque

# New Paper Account Credentials
API_KEY = "PKBE4KSRXPQCWYQGRJSS5TAGNT"
API_SECRET = "4syiQj2mVnR8hKkk8gopJJA1wcCF2p2iTpiaBfZnBRW4"
BASE_URL = "https://paper-api.alpaca.markets"

# Money Scraper Settings
PROFIT_TARGET_DOLLARS = 5.0   # Exit at +$5 profit
STOP_LOSS_DOLLARS = 2.0       # Exit at -$2 loss
MAX_POSITIONS = 10            # Max 10 positions - pick best AI signals
SCAN_INTERVAL_SECONDS = 30    # Scan every 30 seconds


class MoneyScraperBot:
    def __init__(self, model_path='models/SHARED_dqn_best.pth'):
        self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        
        # Load AI Model
        state_size = 231
        self.agent = DuelingDQN(state_size, 3, use_noisy=True)
        self.agent.load(model_path)
        self.agent.model.train()  # Training mode for online learning
        self.model_path = model_path
        
        # Load stock symbols from portfolio
        self.symbols = self.load_portfolio()
        
        # Online Learning Components
        self.replay_buffer = deque(maxlen=10000)  # Store recent experiences
        self.position_states = {}  # Track entry states for open positions
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=0.0001)
        self.scan_count = 0
        
        # Load existing replay buffer if exists
        buffer_path = 'models/money_scraper_buffer.pkl'
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, 'rb') as f:
                    self.replay_buffer = pickle.load(f)
                print(f"📚 Loaded {len(self.replay_buffer)} experiences from buffer")
            except:
                pass
        
        print(f"\n{'='*60}")
        print(f"💰 AI MONEY SCRAPER BOT")
        print(f"{'='*60}")
        print(f"📊 Model: {model_path}")
        print(f"📋 Portfolio: {len(self.symbols)} stocks")
        print(f"🎯 Profit: +${PROFIT_TARGET_DOLLARS} | Stop: -${STOP_LOSS_DOLLARS}")
        print(f"🔢 Max Positions: {MAX_POSITIONS}")
        print(f"🧠 Online Learning: ENABLED")
        print(f"{'='*60}\n")
    
    def load_portfolio(self):
        """Load stock symbols from my_portfolio.txt"""
        paths = ['my_portfolio.txt', '../my_portfolio.txt']
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                        print(f"📋 Loaded {len(symbols)} symbols from {path}")
                        return symbols
                except:
                    pass
        return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']  # Fallback
    
    def get_bulk_data(self, symbols):
        """Fetch daily data for multiple symbols using yfinance"""
        all_bars = {}
        chunk_size = 20
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            try:
                data = yf.download(
                    tickers=chunk,
                    period='1y',
                    interval='1d',
                    group_by='ticker',
                    progress=False,
                    auto_adjust=True
                )
                
                if data.empty:
                    continue
                
                if len(chunk) == 1:
                    symbol = chunk[0]
                    df = data.copy()
                    df.columns = [c.capitalize() for c in df.columns]
                    if len(df) >= 200:
                        all_bars[symbol] = df
                else:
                    for symbol in chunk:
                        try:
                            if symbol in data.columns.get_level_values(0):
                                df = data[symbol].copy()
                                df.columns = [c.capitalize() for c in df.columns]
                                df = df.dropna()
                                if len(df) >= 200:
                                    all_bars[symbol] = df
                        except:
                            pass
            except:
                pass
        
        return all_bars
    
    def get_ai_action(self, symbol, df):
        """Get AI model's action for a symbol and confidence score"""
        try:
            df = add_technical_indicators(df)
            current_step = len(df) - 1
            market_state = normalize_state(df, current_step, 20)  # Fixed: added window_size=20
            
            # Get account info
            try:
                account = self.api.get_account()
                equity = float(account.equity)
                cash = float(account.cash)
            except:
                equity = 10000
                cash = 10000
            
            # Check if we have a position
            market_value = 0
            try:
                position = self.api.get_position(symbol)
                market_value = float(position.market_value)
            except:
                pass
            
            portfolio_state = np.array([
                cash / equity,
                market_value / equity,
                0.0,
                1.0 if market_value > 0 else 0.0,
                0.0
            ])
            
            state = np.concatenate((market_state, portfolio_state))
            
            # Get Q-values for confidence scoring
            import torch
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.agent.model(state_tensor)
                action = q_values.argmax(1).item()
                confidence = q_values[0][action].item()  # Q-value for selected action
            
            return action, confidence  # 0=HOLD, 1=BUY, 2=SELL + confidence
        except Exception as e:
            print(f"   ⚠️ AI action error for {symbol}: {str(e)[:50]}")
            return 0, 0.0
    
    def store_experience(self, symbol, state, action, next_state, reward):
        """Store trading experience for online learning"""
        self.replay_buffer.append((state, action, reward, next_state, False))
        
        # Save buffer periodically
        if len(self.replay_buffer) % 100 == 0:
            try:
                with open('models/money_scraper_buffer.pkl', 'wb') as f:
                    pickle.dump(self.replay_buffer, f)
            except:
                pass
    
    def train_on_experiences(self, batch_size=64):
        """Train the model on collected live experiences"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample random batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])
        
        # Compute Q values
        current_q = self.agent.model(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.agent.model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q
        
        # Compute loss and update
        loss = torch.nn.functional.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def check_exits(self):
        """Check all positions for profit/loss targets"""
        try:
            positions = self.api.list_positions()
            
            if not positions:
                return
            
            print(f"\n💼 Managing {len(positions)} positions...")
            
            for p in positions:
                try:
                    symbol = p.symbol
                    qty = float(p.qty)
                    entry_price = float(p.avg_entry_price)
                    current_price = float(p.current_price)
                    
                    # Calculate P/L in dollars
                    total_pnl = (current_price - entry_price) * qty
                    
                    # Check profit target
                    if total_pnl >= PROFIT_TARGET_DOLLARS:
                        self.api.close_position(symbol)
                        print(f"   ✅ {symbol:6s} PROFIT: ${total_pnl:+.2f} ({qty:.0f} shares)")
                        # Store experience
                        if symbol in self.position_states:
                            self.store_experience(symbol, self.position_states[symbol]['state'], 1, self.position_states[symbol]['state'], total_pnl)
                            del self.position_states[symbol]
                    
                    # Check stop loss
                    elif total_pnl <= -STOP_LOSS_DOLLARS:
                        self.api.close_position(symbol)
                        print(f"   🛑 {symbol:6s} STOP: ${total_pnl:+.2f} ({qty:.0f} shares)")
                        # Store experience
                        if symbol in self.position_states:
                            self.store_experience(symbol, self.position_states[symbol]['state'], 1, self.position_states[symbol]['state'], total_pnl)
                            del self.position_states[symbol]
                    
                    else:
                        print(f"   💎 {symbol:6s} ${total_pnl:+.2f} ({current_price:.2f})")
                
                except Exception as e:
                    print(f"   ⚠️ {p.symbol}: {str(e)[:30]}")
        
        except Exception as e:
            print(f"Error checking exits: {e}")
    
    def scan_for_entries(self):
        """Scan for new AI-recommended entries and pick top 10"""
        try:
            # Get current positions
            positions = self.api.list_positions()
            
            # Get current holdings
            held_symbols = [p.symbol for p in positions]
            num_open_slots = MAX_POSITIONS - len(held_symbols)
            
            if num_open_slots <= 0:
                print(f"\n⚠️ Max positions ({MAX_POSITIONS}) reached")
                return
            
            # Filter portfolio to unowned stocks
            available = [s for s in self.symbols if s not in held_symbols]
            
            if not available:
                return
            
            print(f"\n🔍 Scanning {len(available)} stocks for entries... (slots: {num_open_slots}/{MAX_POSITIONS})")
            
            # Fetch data
            all_data = self.get_bulk_data(available)
            
            # Analyze each symbol and collect BUY signals with confidence
            buy_signals = []  # List of (symbol, price, confidence)
            
            for symbol, df in all_data.items():
                if symbol in held_symbols:
                    continue
                
                action, confidence = self.get_ai_action(symbol, df)
                
                if action == 1:  # AI says BUY
                    current_price = float(df['Close'].iloc[-1])
                    buy_signals.append((symbol, current_price, confidence))
            
            # Rank by confidence and take top N
            if buy_signals:
                print(f"\n🎯 AI Recommendations: {len(buy_signals)} BUY signals")
                
                # Sort by confidence (highest first)
                buy_signals.sort(key=lambda x: x[2], reverse=True)
                
                # Take only top N slots available
                top_picks = buy_signals[:num_open_slots]
                
                if len(buy_signals) > num_open_slots:
                    print(f"   📊 Ranked top {len(top_picks)} from {len(buy_signals)} candidates")
                
                try:
                    account = self.api.get_account()
                    cash = float(account.cash)
                except:
                    cash = 1000
                
                # Split cash equally among top picks
                allocation_per_stock = cash / len(top_picks) if len(top_picks) > 0 else 0
                
                for symbol, price, confidence in top_picks:
                    qty = int(allocation_per_stock / price)
                    
                    if qty > 0:
                        try:
                            self.api.submit_order(
                                symbol=symbol,
                                qty=qty,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )
                            print(f"   🟢 BUY {symbol}: {qty} @ ${price:.2f} (conf: {confidence:.3f})")
                        except Exception as e:
                            print(f"   ❌ {symbol}: {str(e)[:30]}")
        
        except Exception as e:
            print(f"Error scanning: {e}")
    
    def run(self):
        """Main loop"""
        print("🚀 Starting Money Scraper Bot...\n")
        
        import pytz
        eastern = pytz.timezone('US/Eastern')
        
        while True:
            try:
                now = datetime.now().strftime('%H:%M:%S')
                print(f"\n{'='*60}")
                print(f"[{now}] 📊 Scan Cycle")
                print(f"{'='*60}")
                
                # Check market schedule and EOD liquidation
                now_et = datetime.now(eastern)
                today_str = now_et.strftime('%Y-%m-%d')
                
                # Get market schedule
                market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                try:
                    schedules = self.api.get_calendar(start=today_str, end=today_str)
                    if schedules:
                        s = schedules[0]
                        if hasattr(s.close, 'astimezone'):
                            market_close = s.close.astimezone(eastern)
                        else:
                            market_close = eastern.localize(datetime.combine(now_et.date(), s.close))
                except:
                    pass
                
                # EOD Liquidation (15 mins before close)
                time_to_close = (market_close - now_et).total_seconds() / 60
                if time_to_close <= 15 and time_to_close > 0:
                    positions = self.api.list_positions()
                    if positions:
                        print(f"\n🔔 EOD LIQUIDATION: Closing {len(positions)} positions...")
                        for p in positions:
                            try:
                                self.api.close_position(p.symbol)
                                print(f"   📤 Closed {p.symbol}")
                            except:
                                pass
                        print("✅ All positions liquidated for EOD")
                        time.sleep(900)  # Wait 15 mins
                        continue
                
                # 1. Check exits first
                self.check_exits()
                
                # 2. Scan for new entries
                self.scan_for_entries()
                
                # Sleep
                print(f"\n⏳ Next scan in {SCAN_INTERVAL_SECONDS}s...")
                time.sleep(SCAN_INTERVAL_SECONDS)
            
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                break
            except Exception as e:
                print(f"❌ Loop error: {e}")
                time.sleep(10)


if __name__ == "__main__":
    bot = MoneyScraperBot()
    bot.run()
