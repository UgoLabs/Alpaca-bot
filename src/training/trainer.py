"""
Parallel Trainer for Dueling DQN
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

from src.models.agent import DuelingDQN
from src.models.buffers import PrioritizedReplayBuffer
from src.environments.swing_env import SwingTradingEnv
from config.settings import TrainingConfig, SHARED_MODEL_PATH, SwingTraderCreds, ALPACA_BASE_URL
from src.core.indicators import add_technical_indicators

class ParallelTrainer:
    def __init__(self, episodes_per_symbol=10, parallel_jobs=4, fresh=False):
        self.config = TrainingConfig
        self.episodes_per_symbol = episodes_per_symbol
        self.parallel_jobs = parallel_jobs
        
        # Initialize Agent
        state_size = self.config.get_state_size()
        action_size = 3
        
        self.agent = DuelingDQN(state_size, action_size, learning_rate=self.config.LEARNING_RATE)
        
        if not fresh and os.path.exists(SHARED_MODEL_PATH):
            self.agent.load(str(SHARED_MODEL_PATH))
            
        self.buffer = PrioritizedReplayBuffer(max_size=100000)
        
        # API for data fetching
        self.api = tradeapi.REST(SwingTraderCreds.API_KEY, SwingTraderCreds.API_SECRET, ALPACA_BASE_URL)
        
    def fetch_data(self, symbols, years=2):
        """Fetch and cache data for all symbols."""
        data_map = {}
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        
        print(f"📥 Fetching data for {len(symbols)} symbols...")
        
        pbar = tqdm(symbols)
        for symbol in pbar:
            try:
                bars = self.api.get_bars(
                    symbol, '1Day', 
                    start=start.strftime('%Y-%m-%d'), 
                    end=end.strftime('%Y-%m-%d'),
                    limit=10000,
                    feed='sip'
                ).df
                
                if len(bars) > 200:
                    bars = bars.reset_index()
                    bars = bars.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    })
                    # Add indicators
                    bars = add_technical_indicators(bars)
                    data_map[symbol] = bars
            except Exception as e:
                pbar.set_description(f"Error {symbol}")
                continue
                
        print(f"✅ Loaded {len(data_map)} valid datasets")
        return data_map

    def train_episode(self, symbol, df, epsilon):
        """Run a single episode and return experiences."""
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
        data_map = self.fetch_data(symbols)
        valid_symbols = list(data_map.keys())
        
        total_episodes = len(valid_symbols) * self.episodes_per_symbol
        pbar = tqdm(total=total_episodes, desc="Training")
        
        # Parallel Execution
        with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            futures = []
            
            # Submit initial jobs
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
                    
                    # Target Q calculation
                    states_t = torch.FloatTensor(states).to(self.agent.device)
                    next_states_t = torch.FloatTensor(next_states).to(self.agent.device)
                    
                    # Double DQN logic
                    with torch.no_grad():
                        next_actions = self.agent.model(next_states_t).argmax(1)
                        next_q = self.agent.target_model(next_states_t)
                        next_q_values = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1).cpu().numpy()
                        
                    targets = rewards + (1 - dones) * self.config.GAMMA * next_q_values
                    
                    # Update target array for loss calc
                    # We need current Q values to compute loss only for chosen actions
                    # But agent.train_step takes 'targets' as full vector or handles it?
                    # My agent.train_step takes 'targets' as tensor. 
                    # Usually we want target Q value for the specific action.
                    # My agent.train_step implementation calculates Loss(Q[action], target).
                    
                    # Let's check agent.train_step in agent.py
                    # It computes predictions = model(states)
                    # loss = loss_fn(predictions, targets)
                    # SO 'targets' must be same shape as predictions (Batch, Action_Size)
                    # I need to construct the full target vector!
                    
                    current_q_values = self.agent.model(states_t).cpu().detach().numpy()
                    target_vector = current_q_values.copy()
                    for i in range(self.config.BATCH_SIZE):
                        target_vector[i][actions[i]] = targets[i]
                    
                    loss = self.agent.train_step(states, target_vector, weights)
                    training_losses.append(loss)
                    
                    # Update priorities
                    td_errors = np.abs(target_vector[np.arange(len(batch)), actions] - current_q_values[np.arange(len(batch)), actions])
                    self.buffer.update_priorities(indices, td_errors)
                    
                    # Soft update
                    self.agent.soft_update_target_model()
                
                pbar.update(1)
                pbar.set_postfix({'Loss': np.mean(training_losses[-10:]) if training_losses else 0})
                
                # Periodic Save
                completed += 1
                if completed % 100 == 0:
                    self.agent.save(str(SHARED_MODEL_PATH))
            
        self.agent.save(str(SHARED_MODEL_PATH))
        print("Training Complete!")
