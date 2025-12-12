"""
AI-Powered Money Scraper Bot
Uses Dueling DQN model to pick entries, exits at +$5 profit or -$2 loss
"""

import os
import time
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from swing_model import DuelingDQN
from utils import add_technical_indicators, normalize_state
import torch
import pickle
from collections import deque

# Money Scraper API Credentials (Fresh keys for WebSocket)
API_KEY = "PK5EWIG3M7IDV2KL7WW4VUBSRJ"
API_SECRET = "DRuaa1fLvg1n3vxCBtnqcw55FEbdkf7NJZr5yhdZ5Vva"
BASE_URL = "https://paper-api.alpaca.markets"

# Money Scraper Settings
# Replaced dollar-based P/L with Percentage-based
PROFIT_TARGET_PCT = 0.04      # Exit at +4% profit
STOP_LOSS_PCT = 0.02          # Exit at -2% loss
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
MAX_POSITIONS = 8             # Max 8 positions
SCAN_INTERVAL_SECONDS = 5     # Scan every 5s


class MoneyScraperBot:
    def __init__(self, model_path='models/SHARED_dqn_best.pth'):
        self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        
        # WebSocket Stream for real-time data (NO API LIMITS!)
        self.stream = tradeapi.Stream(
            API_KEY, 
            API_SECRET, 
            BASE_URL,
            data_feed='sip'  # Use paid SIP data
        )
        self.bar_cache = {}  # In-memory cache
        self.stream_ready = False
        
        # Load AI Model
        state_size = 231
        self.agent = DuelingDQN(state_size, 3, use_noisy=True)
        self.agent.load(model_path)
        self.agent.model.train()  # Training mode for online learning
        self.model_path = model_path
        
        # Load stock symbols from portfolio
        self.symbols = self.load_portfolio()
        
        # Initialize cache
        for symbol in self.symbols:
            self.bar_cache[symbol] = deque(maxlen=500)
        
        # Start WebSocket stream
        import threading
        import asyncio
        
        async def bar_handler(bar):
            if bar.symbol in self.bar_cache:
                self.bar_cache[bar.symbol].append({
                    'Open': bar.open, 'High': bar.high,
                    'Low': bar.low, 'Close': bar.close,
                    'Volume': bar.volume
                })
                if not self.stream_ready and len(self.bar_cache[bar.symbol]) >= 10:
                    ready = sum(1 for s in self.symbols if len(self.bar_cache.get(s,[])) >= 10)
                    if ready >= len(self.symbols) * 0.1:
                        self.stream_ready = True
                        print(f"✅ Stream ready: {ready}/{len(self.symbols)} symbols")
        
        self.stream.subscribe_bars(bar_handler, *self.symbols)
        threading.Thread(target=lambda: asyncio.run(self.stream._run_forever()), daemon=True).start()
        print(f"🌐 WebSocket streaming {len(self.symbols)} symbols...")
        
        # Online Learning Components
        self.replay_buffer = deque(maxlen=10000)
        self.position_states = {}
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
        print(f"🎯 Profit: {PROFIT_TARGET_PCT*100}% | Stop: -{STOP_LOSS_PCT*100}%")
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
        """Get data from WebSocket cache (instant!) or REST API as fallback"""
        all_bars = {}
        
        # PRIORITY 1: Use WebSocket cache (instant, no API calls)
        for symbol in symbols:
            if symbol in self.bar_cache and len(self.bar_cache[symbol]) >= 10:
                bars_list = list(self.bar_cache[symbol])
                df = pd.DataFrame(bars_list)
                all_bars[symbol] = df
        
        # If stream is ready, return cached data
        if self.stream_ready:
            return all_bars
        
        # FALLBACK: Use REST API (only during startup while stream initializes)
        print(f"   ⏳ Stream initializing, using REST API...")
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=2)
        
        for symbol in symbols:
            try:
                # Get 1-minute bars with SIP feed
                bars = self.api.get_bars(
                    symbol,
                    '1Min',
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=10000,
                    feed='sip'
                ).df
                
                if bars.empty or len(bars) < 200:
                    continue
                
                # Convert to expected format (OHLCV)
                df = pd.DataFrame({
                    'Open': bars['open'],
                    'High': bars['high'],
                    'Low': bars['low'],
                    'Close': bars['close'],
                    'Volume': bars['volume']
                })
                
                all_bars[symbol] = df
                
            except Exception as e:
                # Skip symbols that fail (delisted, etc)
                pass
        
        return all_bars
    
    def get_ai_action(self, symbol, df, account=None, position=None):
        """Get AI model's action for a symbol and confidence score"""
        try:
            df = add_technical_indicators(df)
            current_step = len(df) - 1
            market_state = normalize_state(df, current_step, 20)
            
            # Use cached account info or defaults
            if account:
                equity = float(account.equity)
                cash = float(account.cash)
            else:
                equity = 10000.0
                cash = 10000.0
            
            # Use cached position info
            market_value = 0.0
            if position:
                market_value = float(position.market_value)
            
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
                confidence = q_values[0][action].item()
            
            return action, confidence
        except Exception as e:
            # print(f"   ⚠️ AI action error for {symbol}: {str(e)[:50]}")
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
                    
                    # Calculate P/L in percentage and dollars
                    pnl_pct = float(p.unrealized_plpc)
                    total_pnl = float(p.unrealized_pl)
                    
                    # Check profit target (e.g. +4%)
                    if pnl_pct >= PROFIT_TARGET_PCT:
                        # Cancel any open orders first to avoid conflicts
                        try:
                            self.api.cancel_all_orders()
                        except:
                            pass
                        self.api.close_position(symbol)
                        print(f"   ✅ {symbol:6s} PROFIT: {pnl_pct*100:+.2f}% (${total_pnl:+.2f})")
                        # Store experience
                        if symbol in self.position_states:
                            self.store_experience(symbol, self.position_states[symbol]['state'], 1, self.position_states[symbol]['state'], total_pnl)
                            del self.position_states[symbol]
                    
                    # Check stop loss (e.g. -2%)
                    elif pnl_pct <= -STOP_LOSS_PCT:
                        # Cancel any open orders first to avoid conflicts
                        try:
                            self.api.cancel_all_orders()
                        except:
                            pass
                        self.api.close_position(symbol)
                        print(f"   🛑 {symbol:6s} STOP: {pnl_pct*100:+.2f}% (${total_pnl:+.2f})")
                        # Store experience
                        if symbol in self.position_states:
                            self.store_experience(symbol, self.position_states[symbol]['state'], 1, self.position_states[symbol]['state'], total_pnl)
                            del self.position_states[symbol]
                    
                    else:
                        print(f"   💎 {symbol:6s} {pnl_pct*100:+.2f}% (${total_pnl:+.2f})")
                
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
            
            # Fetch account info ONCE
            try:
                account = self.api.get_account()
            except:
                account = None
            
            # Create position map for fast lookup
            position_map = {p.symbol: p for p in positions}
            
            for symbol, df in all_data.items():
                if symbol in held_symbols:
                    continue
                
                # Get current price
                current_price = df.iloc[-1]['Close']
                
                # Get AI action with cached account/position data
                action, confidence = self.get_ai_action(symbol, df, account=account, position=position_map.get(symbol))
                
                # Check for BUY signal (1) with high confidence
                if action == 1 and confidence > 0.5:
                    buy_signals.append((symbol, current_price, confidence))
            
            # Sort signals by confidence (highest first)
            buy_signals.sort(key=lambda x: x[2], reverse=True)
            
            if buy_signals:
                print(f"\n🎯 AI Recommendations: {len(buy_signals)} BUY signals")
                
                # Take top N signals to fill slots
                top_picks = buy_signals[:num_open_slots]
                
                if len(buy_signals) > num_open_slots:
                    print(f"   📊 Ranked top {len(top_picks)} from {len(buy_signals)} candidates")
                
                for pick in top_picks:
                    symbol, price, confidence = pick
                    
                    # Calculate quantity based on position size limit ($200 per trade?)
                    # Dynamic sizing: Use 10% of equity or fixed amount?
                    # Let's use simpler logic: 1 share for expensive, more for cheap
                    # Better: Allocate remaining buying power / open slots
                    if account:
                        buying_power = float(account.daytrading_buying_power)
                    else:
                        buying_power = 20000.0
                    
                    # Allocate portion of BP
                    allocation = min(2000.0, buying_power / max(1, num_open_slots)) # Cap at $2k per trade
                    qty = int(allocation / price)
                    
                    if qty > 0:
                        try:
                            # Calculate stop and profit prices
                            stop_price = round(price - (STOP_LOSS_DOLLARS / qty), 2)
                            take_profit_price = round(price + (PROFIT_TARGET_DOLLARS / qty), 2)
                            
                            # Submit BRACKET order (BUY + STOP + PROFIT in one atomic order)
                            self.api.submit_order(
                                symbol=symbol,
                                qty=qty,
                                side='buy',
                                type='market',
                                time_in_force='day',
                                order_class='bracket',
                                stop_loss={'stop_price': stop_price},
                                take_profit={'limit_price': take_profit_price}
                            )
                            print(f"   🟢 BUY {symbol}: {qty} @ ${price:.2f} (conf: {confidence:.3f})")
                            print(f"   🛡️ STOP @ ${stop_price:.2f} | 🎯 PROFIT @ ${take_profit_price:.2f}")
                                
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
                        # Cancel all open orders first to avoid conflicts
                        try:
                            self.api.cancel_all_orders()
                            print("   🚫 Cancelled all open orders")
                        except:
                            pass
                        for p in positions:
                            try:
                                self.api.close_position(p.symbol)
                                print(f"   📤 Closed {p.symbol}")
                            except Exception as e:
                                print(f"   ⚠️ Failed to close {p.symbol}: {str(e)[:30]}")
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
