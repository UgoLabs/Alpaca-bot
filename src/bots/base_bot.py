"""
Base Trading Bot Class
Shared functionality for all trading bots
"""
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import alpaca_trade_api as tradeapi
import numpy as np
import torch
import pytz

from config.settings import ALPACA_BASE_URL, RISK_PER_TRADE
from src.models.agent import DuelingDQN
from src.core.state import get_state_size


class BaseBot(ABC):
    """
    Abstract base class for all trading bots.
    Provides shared API connection, model loading, and utility methods.
    """
    
    def __init__(self, api_key, api_secret, model_path, watchlist_file):
        """
        Initialize the bot.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            model_path: Path to trained model weights
            watchlist_file: Path to stock watchlist file
        """
        # API Connection
        self.api = tradeapi.REST(api_key, api_secret, ALPACA_BASE_URL)
        
        # Load watchlist
        self.symbols = self._load_watchlist(watchlist_file)
        
        # Load AI model
        self.agent = self._load_model(model_path)
        
        # Position tracking
        self.position_states = {}
        self.position_tracking = {}
        
        # Experience replay for online learning
        self.replay_buffer = deque(maxlen=10000)
        
        # Timezone
        self.eastern = pytz.timezone('US/Eastern')
        
        print(f"{'='*60}")
        print(f"🤖 {self.__class__.__name__} Initialized")
        print(f"📊 Model: {model_path}")
        print(f"📋 Watchlist: {len(self.symbols)} symbols")
        print(f"{'='*60}\n")
    
    def _load_watchlist(self, filepath):
        """Load symbols from watchlist file."""
        # Try multiple possible locations
        paths_to_try = [
            Path(filepath),
            Path(__file__).parent.parent.parent / filepath,
            Path(__file__).parent.parent.parent / "config" / "watchlists" / filepath,
        ]
        
        for path in paths_to_try:
            if path.exists():
                with open(path) as f:
                    symbols = [line.strip().upper() for line in f if line.strip()]
                print(f"📋 Loaded {len(symbols)} symbols from {path}")
                return symbols
        
        raise FileNotFoundError(f"Watchlist not found: {filepath}")
    
    def _load_model(self, model_path):
        """Load or create the DQN agent."""
        state_size = get_state_size()
        action_size = 3  # HOLD, BUY, SELL
        
        agent = DuelingDQN(state_size, action_size)
        
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print(f"⚠️ Model not found at {model_path}. Using untrained model.")
        
        return agent
    
    def get_account_info(self):
        """Get current account information."""
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'daytrading_buying_power': float(account.daytrading_buying_power)
        }
    
    def get_positions_map(self):
        """Get a map of current positions by symbol."""
        positions = self.api.list_positions()
        return {p.symbol: p for p in positions}
    
    def is_market_open(self):
        """Check if market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception:
            return False
    
    def get_market_schedule(self):
        """Get today's market open/close times."""
        now_et = datetime.now(self.eastern)
        today_str = now_et.strftime('%Y-%m-%d')
        
        try:
            schedules = self.api.get_calendar(start=today_str, end=today_str)
            if not schedules:
                return None, None  # Holiday
            
            s = schedules[0]
            if hasattr(s.open, 'astimezone'):
                market_open = s.open.astimezone(self.eastern)
                market_close = s.close.astimezone(self.eastern)
            else:
                market_open = self.eastern.localize(datetime.combine(now_et.date(), s.open))
                market_close = self.eastern.localize(datetime.combine(now_et.date(), s.close))
            
            return market_open, market_close
        except Exception as e:
            print(f"⚠️ Calendar error: {e}")
            # Default times
            return (
                now_et.replace(hour=9, minute=30, second=0),
                now_et.replace(hour=16, minute=0, second=0)
            )
    
    def store_experience(self, symbol, state, action, next_state, reward):
        """Store experience for online learning."""
        self.replay_buffer.append((state, action, reward, next_state, False))
    
    def train_on_experiences(self, batch_size=64, gamma=0.99):
        """Train on replay buffer experiences."""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        batch = list(self.replay_buffer)[-batch_size:]
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        state_tensor = torch.FloatTensor(states).to(self.agent.device)
        with torch.no_grad():
            current_q = self.agent.model(state_tensor).cpu().numpy()
        
        # Target Q-values (Double DQN)
        next_state_tensor = torch.FloatTensor(next_states).to(self.agent.device)
        with torch.no_grad():
            next_q = self.agent.target_model(next_state_tensor).cpu().numpy()
        
        # Update targets
        targets = current_q.copy()
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + gamma * np.max(next_q[i])
        
        # Train
        loss = self.agent.train_step(states, targets)
        self.agent.soft_update_target_model()
        
        return loss
    
    @abstractmethod
    def check_exits(self):
        """Check positions for exit signals. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def scan_for_entries(self):
        """Scan for new entry opportunities. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def run_once(self):
        """Run one complete trading cycle. Must be implemented by subclass."""
        pass
    
    def run_loop(self, interval_seconds):
        """Main trading loop with strict market hours management."""
        print(f"🚀 Starting {self.__class__.__name__} (interval: {interval_seconds}s)")
        
        # Schedule configuration
        WARMUP_MINUTES = 10
        LIQUIDATION_MINUTES = 10
        
        # Flag to prevent re-liquidation or re-training multiple times a day
        has_liquidated_today = False
        
        while True:
            try:
                now_et = datetime.now(self.eastern)
                market_open, market_close = self.get_market_schedule()
                
                # Case 1: Holiday / Error
                if market_open is None:
                    print(f"💤 Market Closed (Holiday). Sleeping 1 hour.")
                    time.sleep(3600)
                    has_liquidated_today = False
                    continue
                
                # Define key timestamps
                warmup_time = market_open - timedelta(minutes=WARMUP_MINUTES)
                liquidation_time = market_close - timedelta(minutes=LIQUIDATION_MINUTES)
                
                # Case 2: Before Warmup (Deep Sleep)
                if now_et < warmup_time:
                    sleep_seconds = (warmup_time - now_et).total_seconds()
                    print(f"💤 Market Closed. Deep sleep until {warmup_time.strftime('%H:%M')} ET ({sleep_seconds/3600:.1f} hours).")
                    has_liquidated_today = False # Reset for the new day
                    time.sleep(min(sleep_seconds, 21600)) 
                    continue
                
                # Case 3: Warmup Phase (10 mins before open)
                if warmup_time <= now_et < market_open:
                    print(f"🔥 Pre-Market Warmup ({WARMUP_MINUTES} mins before open)...")
                    self.on_warmup()
                    
                    # Wait for market open (poll every 10s)
                    while datetime.now(self.eastern) < market_open:
                        time.sleep(10)
                    print("🔔 Market is OPEN!")
                    continue
                
                # Case 4: Market Open (Trading Session)
                if market_open <= now_et < liquidation_time:
                    # Reset liquidation flag if we somehow wrapped around (failsafe)
                    if has_liquidated_today and now_et < liquidation_time:
                        has_liquidated_today = False
                        
                    self.run_once()
                    
                    print(f"⏳ Next scan in {interval_seconds}s...")
                    time.sleep(interval_seconds)
                    continue
                
                # Case 5: EOD Liquidation Window (10 mins before close)
                if liquidation_time <= now_et < market_close:
                    if not has_liquidated_today:
                        print(f"🛑 EOD WINDOW ({LIQUIDATION_MINUTES} mins to close) - Stopping Trading & Liquidating.")
                        
                        # 1. Liquidate if configured
                        if hasattr(self, 'config') and getattr(self.config, 'LIQUIDATE_EOD', False):
                            self._liquidate_all_positions()
                        
                        # 2. Train on Daily Experiences
                        if hasattr(self, 'config') and getattr(self.config, 'ONLINE_LEARNING', False):
                            print("🧠 Running EOD Online Training...")
                            loss = self.train_on_experiences(batch_size=64) # Train on a larger batch at EOD
                            print(f"🧠 EOD Training Complete. Loss: {loss:.4f}")
                            self._save_experiences()
                        
                        has_liquidated_today = True
                    
                    # Sleep until market close to prevent re-entering
                    sleep_seconds = (market_close - datetime.now(self.eastern)).total_seconds()
                    if sleep_seconds > 0:
                        print(f"💤 Sleeping until Market Close ({sleep_seconds:.0f}s)...")
                        time.sleep(sleep_seconds + 5) # Sleep past close
                    continue

                # Case 6: After Close (Market just closed)
                if now_et >= market_close:
                    print("🏁 Market Closed.")
                    self.on_shutdown()
                    time.sleep(60) # Wait a minute before restarting loop to hit Deep Sleep
                    continue

            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                self.on_shutdown()
                break
            except Exception as e:
                print(f"❌ Error in run_loop: {e}")
                time.sleep(60)
                
                time.sleep(60)

    def on_warmup(self):
        """Called 5 mins before market open. Override in subclasses."""
        pass
    
    def on_shutdown(self):
        """Called after market close. Handles EOD liquidation if configured."""
        # Check if this bot should liquidate at EOD
        if hasattr(self, 'config') and getattr(self.config, 'LIQUIDATE_EOD', False):
            self._liquidate_all_positions()
        
        # Save any learned experiences
        if len(self.replay_buffer) > 0:
            self._save_experiences()
    
    def _liquidate_all_positions(self):
        """Close all open positions (EOD cleanup)."""
        try:
            positions = self.api.list_positions()
            if not positions:
                print("🔔 EOD: No positions to liquidate")
                return
            
            print(f"🔔 EOD LIQUIDATION: Closing {len(positions)} positions...")
            
            for p in positions:
                try:
                    self.api.close_position(p.symbol)
                    pnl = float(p.unrealized_pl)
                    emoji = "✅" if pnl >= 0 else "🛑"
                    print(f"   {emoji} {p.symbol}: ${pnl:+.2f}")
                    
                    # Store experience for learning
                    if p.symbol in self.position_states:
                        self._record_trade_experience(p.symbol, pnl, "EOD")
                        
                except Exception as e:
                    print(f"   ❌ Failed to close {p.symbol}: {e}")
            
            print("🔔 EOD: Liquidation complete")
            
        except Exception as e:
            print(f"❌ EOD Liquidation error: {e}")
    
    def _record_trade_experience(self, symbol, pnl, exit_type):
        """Record a completed trade as experience for online learning."""
        if symbol not in self.position_states:
            return
        
        entry_state = self.position_states[symbol].get('state')
        if entry_state is None:
            return
        
        # Calculate reward based on PnL
        # Normalize reward to reasonable range
        reward = np.clip(pnl / 100, -2, 2)  # $100 = 1.0 reward
        
        # Boost reward for profitable exits
        if exit_type == "PROFIT":
            reward *= 1.5
        elif exit_type == "STOP":
            reward *= 0.8  # Slightly less penalty for disciplined stops
        
        # Store experience (state, action=1 for buy, reward, next_state, done=True)
        # Note: We don't have perfect next_state, use entry_state as approximation
        self.replay_buffer.append((entry_state, 1, reward, entry_state, True))
        
        # Clean up
        del self.position_states[symbol]
    
    def _save_experiences(self):
        """Save replay buffer for persistence across restarts."""
        try:
            import pickle
            from config.settings import REPLAY_BUFFER_PATH
            with open(REPLAY_BUFFER_PATH, 'wb') as f:
                pickle.dump(list(self.replay_buffer), f)
            print(f"💾 Saved {len(self.replay_buffer)} experiences")
        except Exception as e:
            print(f"⚠️ Could not save experiences: {e}")
    
    def _load_experiences(self):
        """Load replay buffer from disk."""
        try:
            import pickle
            from config.settings import REPLAY_BUFFER_PATH
            import os
            if os.path.exists(REPLAY_BUFFER_PATH):
                with open(REPLAY_BUFFER_PATH, 'rb') as f:
                    experiences = pickle.load(f)
                    for exp in experiences:
                        self.replay_buffer.append(exp)
                print(f"📂 Loaded {len(experiences)} saved experiences")
        except Exception as e:
            print(f"⚠️ Could not load experiences: {e}")
