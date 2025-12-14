#!/usr/bin/env python
"""
Scalper Model Trainer
Trains on 5-Minute bars for Money Scraper and Day Trader bots.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_trade_api as tradeapi
from config.settings import (
    TrainingConfig, SCALPER_MODEL_PATH, MODEL_DIR,
    SwingTraderCreds, ALPACA_BASE_URL
)
from src.models.agent import DuelingDQN
from src.models.buffers import PrioritizedReplayBuffer
from src.environments.swing_env import SwingTradingEnv
from src.core.indicators import add_technical_indicators


class ScalperTrainer:
    """
    Trainer for scalping models using intraday (5Min) data.
    """
    
    def __init__(self, timeframe='5Min', lookback_days=30, episodes_per_symbol=10, workers=4, fresh=False):
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.episodes_per_symbol = episodes_per_symbol
        self.workers = workers
        self.config = TrainingConfig
        
        # Initialize Agent
        state_size = self.config.get_state_size()
        action_size = 3
        
        self.agent = DuelingDQN(state_size, action_size, learning_rate=self.config.LEARNING_RATE)
        
        # Load existing scalper model if exists and not fresh
        if not fresh and os.path.exists(SCALPER_MODEL_PATH):
            print(f"📥 Loading existing scalper model: {SCALPER_MODEL_PATH}")
            self.agent.load(str(SCALPER_MODEL_PATH))
        
        self.buffer = PrioritizedReplayBuffer(max_size=100000)
        
        # API
        self.api = tradeapi.REST(SwingTraderCreds.API_KEY, SwingTraderCreds.API_SECRET, ALPACA_BASE_URL)
        
        print(f"🚀 Scalper Trainer Initialized")
        print(f"   Timeframe: {self.timeframe}")
        print(f"   Lookback: {self.lookback_days} days")
        print(f"   Workers: {self.workers}")
    
    def fetch_intraday_data(self, symbols):
        """Fetch intraday bars for all symbols."""
        data_map = {}
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)
        
        print(f"📥 Fetching {self.timeframe} data for {len(symbols)} symbols...")
        
        pbar = tqdm(symbols)
        for symbol in pbar:
            try:
                bars = self.api.get_bars(
                    symbol, self.timeframe,
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    limit=10000,  # Max per request
                    feed='sip'
                ).df
                
                if len(bars) > 100:  # Need enough bars for indicators
                    bars = bars.reset_index()
                    bars = bars.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    })
                    bars = add_technical_indicators(bars)
                    data_map[symbol] = bars
                    pbar.set_description(f"{symbol}: {len(bars)} bars")
            except Exception as e:
                pbar.set_description(f"Error {symbol}")
                continue
        
        print(f"✅ Loaded {len(data_map)} symbols with valid data")
        return data_map
    
    def train_episode(self, symbol, df, epsilon):
        """Run one episode and return experiences."""
        env = SwingTradingEnv(df, window_size=self.config.WINDOW_SIZE)
        state = env.reset()
        done = False
        experiences = []
        total_reward = 0
        
        while not done:
            action = self.agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)
            experiences.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
        
        return experiences, env.get_metrics()
    
    def train(self, symbols):
        """Main training loop."""
        data_map = self.fetch_intraday_data(symbols)
        valid_symbols = list(data_map.keys())
        
        if not valid_symbols:
            print("❌ No valid data found. Exiting.")
            return
        
        total_episodes = len(valid_symbols) * self.episodes_per_symbol
        pbar = tqdm(total=total_episodes, desc="Training Scalper")
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            
            # Submit all jobs
            for _ in range(self.episodes_per_symbol):
                for symbol in valid_symbols:
                    epsilon = max(0.01, 0.5 - (0.5 * len(futures) / total_episodes))
                    futures.append(
                        executor.submit(self.train_episode, symbol, data_map[symbol], epsilon)
                    )
            
            # Process results
            completed = 0
            training_losses = []
            
            for future in as_completed(futures):
                experiences, metrics = future.result()
                
                # Add to buffer
                for exp in experiences:
                    self.buffer.add(exp)
                
                # Train if buffer ready
                if self.buffer.size() > self.config.BATCH_SIZE:
                    batch, indices, weights = self.buffer.sample(self.config.BATCH_SIZE)
                    
                    states = np.array([e[0] for e in batch])
                    actions = np.array([e[1] for e in batch])
                    rewards = np.array([e[2] for e in batch])
                    next_states = np.array([e[3] for e in batch])
                    dones = np.array([e[4] for e in batch])
                    
                    # Target Q calculation (Double DQN)
                    states_t = torch.FloatTensor(states).to(self.agent.device)
                    next_states_t = torch.FloatTensor(next_states).to(self.agent.device)
                    
                    with torch.no_grad():
                        next_actions = self.agent.model(next_states_t).argmax(1)
                        next_q = self.agent.target_model(next_states_t)
                        next_q_values = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1).cpu().numpy()
                    
                    targets = rewards + (1 - dones) * self.config.GAMMA * next_q_values
                    
                    # Construct target vector
                    current_q_values = self.agent.model(states_t).cpu().detach().numpy()
                    target_vector = current_q_values.copy()
                    for i in range(self.config.BATCH_SIZE):
                        target_vector[i][actions[i]] = targets[i]
                    
                    loss = self.agent.train_step(states, target_vector, weights)
                    training_losses.append(loss)
                    
                    # Update priorities
                    td_errors = np.abs(target_vector[np.arange(len(batch)), actions] - 
                                      current_q_values[np.arange(len(batch)), actions])
                    self.buffer.update_priorities(indices, td_errors)
                    
                    # Soft update target
                    self.agent.soft_update_target_model()
                
                pbar.update(1)
                pbar.set_postfix({'Loss': np.mean(training_losses[-10:]) if training_losses else 0})
                
                # Periodic save
                completed += 1
                if completed % 50 == 0:
                    self.agent.save(str(SCALPER_MODEL_PATH))
        
        # Final save
        self.agent.save(str(SCALPER_MODEL_PATH))
        print(f"\n✅ Training Complete! Model saved to: {SCALPER_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train Scalper Model (5Min bars)")
    parser.add_argument('--timeframe', type=str, default='5Min', help='Bar timeframe (1Min, 5Min, 15Min)')
    parser.add_argument('--days', type=int, default=30, help='Days of history to fetch')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per symbol')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore existing model)')
    args = parser.parse_args()
    
    # Load watchlist
    watchlist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_portfolio.txt")
    with open(watchlist_path) as f:
        symbols = [line.strip().upper() for line in f if line.strip()]
    
    print(f"📋 Loaded {len(symbols)} symbols")
    
    trainer = ScalperTrainer(
        timeframe=args.timeframe,
        lookback_days=args.days,
        episodes_per_symbol=args.episodes,
        workers=args.workers,
        fresh=args.fresh
    )
    
    trainer.train(symbols)


if __name__ == "__main__":
    main()
