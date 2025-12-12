"""
Base Trading Bot Class
Shared functionality for all trading bots
"""
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
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
        
        # 5 minute pre-market warmup
        WARMUP_MINUTES = 5
        
        while True:
            try:
                now_et = datetime.now(self.eastern)
                market_open, market_close = self.get_market_schedule()
                
                # Case 1: Holiday / Error
                if market_open is None:
                    print(f"💤 Market Closed (Holiday). Sleeping 1 hour.")
                    time.sleep(3600)
                    continue
                
                # Timestamps
                warmup_time = market_open - timedelta(minutes=WARMUP_MINUTES)
                
                # Case 2: Before Warmup Time (Deep Sleep)
                if now_et < warmup_time:
                    sleep_seconds = (warmup_time - now_et).total_seconds()
                    # Sleep in chunks to allow interrupts, but max sleep
                    print(f"💤 Market Closed. Deep sleep until {warmup_time.strftime('%H:%M')} ET ({sleep_seconds/3600:.1f} hours).")
                    
                    # Sleep until warmup (cap at 6 hours to periodic check)
                    time.sleep(min(sleep_seconds, 21600)) 
                    continue
                
                # Case 3: Warmup Phase (5 mins before open)
                if warmup_time <= now_et < market_open:
                    print(f"🔥 Pre-Market Warmup ({WARMUP_MINUTES} mins before open)...")
                    self.on_warmup()
                    
                    # Wait for market open (poll every 10s)
                    while datetime.now(self.eastern) < market_open:
                        time.sleep(10)
                    print("🔔 Market is OPEN!")
                
                # Case 4: Market Open
                if market_open <= now_et <= market_close:
                    self.run_once()
                    
                    # Compute sleep time (account for execution time)
                    # We want to maintain exact interval cadence if possible
                    print(f"⏳ Next scan in {interval_seconds}s...")
                    time.sleep(interval_seconds)
                    continue
                
                # Case 5: After Market Close
                if now_et > market_close:
                    print("🔔 Market Closed. Shutting down systems.")
                    self.on_shutdown()
                    
                    # Calculate time until tomorrow's warmup
                    # Basic logic: Sleep 1 hour then loop will recalc tomorrow's schedule
                    print("💤 sleeping until tomorrow...")
                    time.sleep(3600)
                    continue
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                self.on_shutdown()
                break
            except Exception as e:
                print(f"❌ Error in run_loop: {e}")
                time.sleep(60)

    def on_warmup(self):
        """Called 5 mins before market open. Override in subclasses."""
        pass
        
    def on_shutdown(self):
        """Called after market close. Override in subclasses."""
        pass
