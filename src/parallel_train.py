"""
Parallel Training Script for Shared Model DQN
Uses multiple environments (different symbols) simultaneously with one shared model.
Significantly faster than sequential training.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import os
import json
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import threading

from swing_environment import SwingTradingEnv
from swing_model import DuelingDQN, NStepReplayBuffer, ReplayBuffer
from utils import get_state_size, add_technical_indicators
from backtest import Backtester

# =============================================================================
# Hyperparameters
# =============================================================================
EPISODES_PER_SYMBOL = 100     # Episodes each symbol goes through
BATCH_SIZE = 256              # Larger batch for high parallelism
GAMMA = 0.99
N_STEP = 3
LEARNING_RATE = 0.0003
WINDOW_SIZE = 20
USE_NOISY = True
MEMORY_SIZE = 500000          # Much larger buffer for high parallelism

# Parallel settings - RTX 4070 can handle 32-64 easily
NUM_PARALLEL_ENVS = 64        # Number of environments running simultaneously
MAX_PARALLEL_ENVS = 128       # Maximum allowed

# Risk Management
RISK_PER_TRADE = 0.02
MAX_POSITION_PCT = 0.25

# Target network
TARGET_UPDATE_FREQ = 5        # More frequent updates with more data
SOFT_UPDATE_TAU = 0.005

# Training frequency - train more often with more parallel data
TRAIN_EVERY_N_STEPS = 4       # Train every N environment steps


# =============================================================================
# Data Fetching with Caching
# =============================================================================
def fetch_data_with_cache(symbol, start_date, end_date):
    """Fetch data from yfinance with caching."""
    os.makedirs("data_cache", exist_ok=True)
    cache_file = f"data_cache/{symbol}_{start_date}_{end_date}.csv"
    
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error loading cache for {symbol}: {e}")
    
    try:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, 
                           progress=False, auto_adjust=True, multi_level_index=False)
        except TypeError:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
        if len(df) > 0:
            df.to_csv(cache_file)
        return df
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


def load_portfolio(portfolio_file='../my_portfolio.txt'):
    """Load symbols from portfolio file."""
    paths_to_try = [
        portfolio_file,
        'my_portfolio.txt',
        '../my_portfolio.txt',
        os.path.join(os.path.dirname(__file__), '..', 'my_portfolio.txt')
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                # Remove duplicates while preserving order
                seen = set()
                symbols = []
                for line in f:
                    sym = line.strip()
                    if sym and not sym.startswith('#') and sym not in seen:
                        seen.add(sym)
                        symbols.append(sym)
            print(f"Loaded {len(symbols)} unique symbols from {path}")
            return symbols
    
    print("Portfolio file not found, using default symbols")
    return ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']


# =============================================================================
# Parallel Environment Manager
# =============================================================================
class ParallelEnvManager:
    """
    Manages multiple trading environments running different symbols.
    Efficiently collects experiences for batch training.
    """
    
    def __init__(self, symbol_data_dict, window_size=20, 
                 risk_per_trade=0.02, max_position_pct=0.25,
                 indicators_precomputed=False):
        """
        Args:
            symbol_data_dict: Dict of {symbol: dataframe}
            indicators_precomputed: If True, skip indicator calculation in environments
        """
        self.symbols = list(symbol_data_dict.keys())
        self.data = symbol_data_dict
        self.window_size = window_size
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.indicators_precomputed = indicators_precomputed
        
        # Create environments
        self.envs = {}
        self.states = {}
        self.dones = {}
        
        for symbol in self.symbols:
            self._create_env(symbol)
    
    def _create_env(self, symbol):
        """Create a new environment for a symbol."""
        df = self.data[symbol]
        env = SwingTradingEnv(
            df,
            window_size=self.window_size,
            risk_per_trade=self.risk_per_trade,
            max_position_pct=self.max_position_pct,
            skip_indicators=self.indicators_precomputed
        )
        self.envs[symbol] = env
        self.states[symbol] = env.reset()
        self.dones[symbol] = False
        return env
    
    @property
    def state_size(self):
        """Get state size from any environment."""
        return list(self.envs.values())[0].state_size
    
    @property
    def action_size(self):
        """Get action size from any environment."""
        return list(self.envs.values())[0].action_space.n
    
    def get_active_symbols(self):
        """Get symbols that haven't finished their episode."""
        return [s for s in self.symbols if not self.dones[s]]
    
    def get_states_batch(self, symbols=None):
        """
        Get current states for a batch of symbols.
        Returns: (symbols_list, states_array)
        """
        if symbols is None:
            symbols = self.get_active_symbols()
        
        if not symbols:
            return [], np.array([])
        
        states = np.array([self.states[s] for s in symbols])
        return symbols, states
    
    def step_batch(self, symbols, actions):
        """
        Step multiple environments with given actions.
        
        Args:
            symbols: List of symbols to step
            actions: Array of actions for each symbol
            
        Returns:
            experiences: List of (state, action, reward, next_state, done, symbol)
        """
        experiences = []
        completed_episodes = []
        
        for symbol, action in zip(symbols, actions):
            if self.dones[symbol]:
                continue
                
            state = self.states[symbol]
            env = self.envs[symbol]
            
            next_state, reward, done, info = env.step(action)
            
            experiences.append({
                'symbol': symbol,
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'info': info
            })
            
            self.states[symbol] = next_state
            self.dones[symbol] = done
            
            if done:
                completed_episodes.append({
                    'symbol': symbol,
                    'net_worth': env.net_worth,
                    'metrics': env.get_metrics()
                })
        
        return experiences, completed_episodes
    
    def reset_symbol(self, symbol):
        """Reset a specific symbol's environment."""
        self.states[symbol] = self.envs[symbol].reset()
        self.dones[symbol] = False
    
    def reset_all(self):
        """Reset all environments."""
        for symbol in self.symbols:
            self.reset_symbol(symbol)


# =============================================================================
# Parallel Trainer
# =============================================================================
class ParallelTrainer:
    """
    Trains a single shared DQN model using multiple parallel environments.
    """
    
    def __init__(self, symbol_data, num_parallel=NUM_PARALLEL_ENVS, 
                 learning_rate=LEARNING_RATE, use_noisy=USE_NOISY):
        """
        Args:
            symbol_data: Dict of {symbol: dataframe}
            num_parallel: Number of environments to run simultaneously
        """
        self.symbol_data = symbol_data
        self.symbols = list(symbol_data.keys())
        self.num_parallel = min(num_parallel, len(self.symbols))
        
        # State/action sizes
        self.state_size = get_state_size(WINDOW_SIZE)
        self.action_size = 3
        
        # Initialize shared model
        self.agent = DuelingDQN(
            self.state_size, 
            self.action_size,
            learning_rate=learning_rate,
            use_noisy=use_noisy
        )
        
        # Shared replay buffer
        self.memory = NStepReplayBuffer(
            max_size=MEMORY_SIZE, 
            n_step=N_STEP, 
            gamma=GAMMA
        )
        
        # Training stats
        self.episode_count = 0
        self.training_log = []
        self.best_avg_sharpe = -np.inf
        
        print(f"\nParallel Trainer initialized:")
        print(f"  Symbols: {len(self.symbols)}")
        print(f"  Parallel environments: {self.num_parallel}")
        print(f"  State size: {self.state_size}")
        print(f"  Device: {self.agent.device}")
    
    def _get_symbol_batch(self, episode):
        """Get the next batch of symbols to train on."""
        # Rotate through symbols
        start_idx = (episode * self.num_parallel) % len(self.symbols)
        indices = [(start_idx + i) % len(self.symbols) for i in range(self.num_parallel)]
        return [self.symbols[i] for i in indices]
    
    def train(self, episodes_per_symbol=EPISODES_PER_SYMBOL, resume=True):
        """
        Main training loop with parallel environments.
        """
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Try to load existing model
        model_path = "models/SHARED_dqn_best.pth"
        buffer_path = "models/SHARED_replay_buffer.pkl"
        
        if resume and os.path.exists(model_path):
            print(f"Resuming from existing model: {model_path}")
            self.agent.load(model_path)
        
        # Total episodes needed
        total_episodes = episodes_per_symbol * len(self.symbols)
        # But we process num_parallel symbols per iteration
        total_iterations = total_episodes // self.num_parallel
        
        print(f"\n{'='*60}")
        print(f"PARALLEL SHARED MODEL TRAINING")
        print(f"{'='*60}")
        print(f"Total symbol-episodes: {total_episodes}")
        print(f"Parallel batch size: {self.num_parallel}")
        print(f"Total iterations: {total_iterations}")
        print(f"{'='*60}\n")
        
        epsilon = 1.0 if not USE_NOISY else 0.0
        
        for iteration in range(total_iterations):
            iter_start = datetime.now()
            
            # Get batch of symbols for this iteration
            batch_symbols = self._get_symbol_batch(iteration)
            batch_data = {s: self.symbol_data[s] for s in batch_symbols}
            
            # Create parallel environment manager for this batch
            env_manager = ParallelEnvManager(
                batch_data,
                window_size=WINDOW_SIZE,
                risk_per_trade=RISK_PER_TRADE,
                max_position_pct=MAX_POSITION_PCT,
                indicators_precomputed=True  # Skip indicator calc - already done
            )
            
            # Run episode in all parallel environments
            batch_rewards = {s: 0 for s in batch_symbols}
            batch_steps = 0
            total_loss = 0
            train_steps = 0
            
            while True:
                # Get active environments
                active_symbols = env_manager.get_active_symbols()
                if not active_symbols:
                    break
                
                # Get states batch
                symbols, states = env_manager.get_states_batch(active_symbols)
                
                if len(symbols) == 0:
                    break
                
                # Batch action selection (GPU accelerated)
                actions = self._batch_act(states, epsilon)
                
                # Step all environments
                experiences, completed = env_manager.step_batch(symbols, actions)
                
                # Store experiences (using constant priority to avoid slow individual TD calc)
                for exp in experiences:
                    experience_tuple = (
                        exp['state'], exp['action'], exp['reward'],
                        exp['next_state'], exp['done']
                    )
                    # Use reward magnitude as simple priority (fast, no GPU call)
                    self.memory.add(experience_tuple, abs(exp['reward']) + 1.0)
                    batch_rewards[exp['symbol']] += exp['reward']
                
                batch_steps += len(experiences)
                
                # Train on batch (every few steps to balance speed)
                if self.memory.size() > BATCH_SIZE and batch_steps % TRAIN_EVERY_N_STEPS == 0:
                    loss = self._train_step()
                    total_loss += loss
                    train_steps += 1
                    
                    # Soft update target network
                    self.agent.soft_update_target_model(SOFT_UPDATE_TAU)
            
            # Episode complete for all parallel envs
            iter_time = (datetime.now() - iter_start).total_seconds()
            self.episode_count += self.num_parallel
            
            # Collect metrics from completed episodes
            metrics_list = []
            for symbol in batch_symbols:
                metrics = env_manager.envs[symbol].get_metrics()
                metrics['symbol'] = symbol
                metrics['reward'] = batch_rewards[symbol]
                metrics_list.append(metrics)
            
            # Log
            avg_sharpe = np.mean([m.get('sharpe', 0) for m in metrics_list])
            avg_return = np.mean([m.get('total_return', 0) for m in metrics_list]) * 100
            avg_trades = np.mean([m.get('total_trades', 0) for m in metrics_list])
            
            log_entry = {
                'iteration': iteration + 1,
                'episodes': self.episode_count,
                'symbols': batch_symbols,
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'avg_trades': avg_trades,
                'loss': total_loss / max(train_steps, 1),
                'buffer_size': self.memory.size(),
                'time': iter_time
            }
            self.training_log.append(log_entry)
            
            # Decay epsilon if not using noisy nets
            if not USE_NOISY and epsilon > 0.01:
                epsilon *= 0.995
            
            # Hard update target network periodically
            if (iteration + 1) % TARGET_UPDATE_FREQ == 0:
                self.agent.update_target_model()
            
            # Print progress
            if (iteration + 1) % 5 == 0:
                syms_str = ','.join(batch_symbols[:3]) + ('...' if len(batch_symbols) > 3 else '')
                print(f"Iter {iteration+1:4d}/{total_iterations} | "
                      f"Eps: {self.episode_count:5d} | "
                      f"[{syms_str}] | "
                      f"Return: {avg_return:+6.1f}% | "
                      f"Sharpe: {avg_sharpe:5.2f} | "
                      f"Trades: {avg_trades:.0f} | "
                      f"Time: {iter_time:.1f}s")
                
                # Check for new best
                recent_sharpes = [l['avg_sharpe'] for l in self.training_log[-50:]]
                rolling_sharpe = np.mean(recent_sharpes) if recent_sharpes else 0
                
                if rolling_sharpe > self.best_avg_sharpe and iteration > 20:
                    self.best_avg_sharpe = rolling_sharpe
                    self.agent.save("models/SHARED_dqn_best_training.pth")
                    print(f"  *** New Best! Rolling Sharpe: {self.best_avg_sharpe:.2f} ***")
            
            # Save periodically
            if (iteration + 1) % 50 == 0:
                self.agent.save(f"models/SHARED_dqn_iter_{iteration+1}.pth")
                with open("logs/SHARED_parallel_training_log.json", 'w') as f:
                    json.dump(self.training_log, f, indent=2, default=str)
        
        # Final save
        self.agent.save("models/SHARED_dqn_final.pth")
        with open("logs/SHARED_parallel_training_log.json", 'w') as f:
            json.dump(self.training_log, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("Parallel Training Complete!")
        print(f"Best Rolling Sharpe: {self.best_avg_sharpe:.2f}")
        print(f"Total Episodes: {self.episode_count}")
        print(f"Model saved to: models/SHARED_dqn_best.pth")
        print(f"{'='*60}")
        
        return self.agent
    
    def _batch_act(self, states, epsilon=0.0):
        """
        Get actions for a batch of states (GPU accelerated).
        """
        states_tensor = torch.FloatTensor(states).to(self.agent.device)
        
        self.agent.model.eval()
        with torch.no_grad():
            q_values = self.agent.model(states_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()
        self.agent.model.train()
        
        # Epsilon-greedy (only if not using noisy nets)
        if not USE_NOISY and epsilon > 0:
            random_mask = np.random.rand(len(actions)) < epsilon
            random_actions = np.random.randint(0, self.action_size, size=len(actions))
            actions = np.where(random_mask, random_actions, actions)
        
        return actions
    
    def _calculate_td_error(self, state, action, reward, next_state, done):
        """Calculate TD error for prioritized replay."""
        self.agent.model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.agent.device)
            
            q_current = self.agent.model(state_t)[0][action].item()
            
            if done:
                td_target = reward
            else:
                q_next = self.agent.target_model(next_state_t).max(1)[0].item()
                td_target = reward + (GAMMA ** N_STEP) * q_next
            
            td_error = td_target - q_current
        self.agent.model.train()
        return td_error
    
    def _train_step(self):
        """Perform one training step on batch from replay buffer."""
        minibatch = self.memory.sample(BATCH_SIZE)
        weights = np.ones(BATCH_SIZE)
        
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.agent.device)
        next_states_t = torch.FloatTensor(next_states).to(self.agent.device)
        
        # Double DQN
        self.agent.model.eval()
        with torch.no_grad():
            next_actions = self.agent.model(next_states_t).argmax(1).unsqueeze(1)
            target_q_next = self.agent.target_model(next_states_t).gather(1, next_actions).squeeze(1)
            current_q = self.agent.model(states_t)
        self.agent.model.train()
        
        # Construct targets
        targets = current_q.clone().detach()
        gamma_n = GAMMA ** N_STEP
        
        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + gamma_n * target_q_next[i].item()
        
        # Train
        loss = self.agent.train(states, targets.cpu().numpy(), weights)
        return loss


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Parallel Training for Shared DQN')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--episodes', type=int, default=EPISODES_PER_SYMBOL, 
                        help='Episodes per symbol')
    parser.add_argument('--parallel', type=int, default=NUM_PARALLEL_ENVS,
                        help='Number of parallel environments')
    parser.add_argument('--fresh', action='store_true', help='Start fresh')
    parser.add_argument('--train-years', type=int, default=None, help='Years of training')
    parser.add_argument('--backtest-years', type=int, default=None, help='Years of backtest')
    
    args = parser.parse_args()
    
    # Handle date logic
    backtest_start = None
    backtest_end = None
    do_backtest = False
    
    if args.train_years and args.backtest_years:
        print(f"Calculating dates for {args.train_years}y train + {args.backtest_years}y backtest...")
        end_dt = datetime.now()
        split_dt = end_dt - timedelta(days=args.backtest_years * 365)
        start_dt = split_dt - timedelta(days=args.train_years * 365)
        
        args.start = start_dt.strftime('%Y-%m-%d')
        args.end = split_dt.strftime('%Y-%m-%d')
        backtest_start = split_dt.strftime('%Y-%m-%d')
        backtest_end = end_dt.strftime('%Y-%m-%d')
        do_backtest = True
        
        print(f"Training Period:   {args.start} to {args.end}")
        print(f"Backtest Period:   {backtest_start} to {backtest_end}")
    
    # Load symbols
    symbols = load_portfolio()
    
    # Fetch all data
    print(f"\n{'='*60}")
    print(f"FETCHING DATA")
    print(f"{'='*60}")
    
    symbol_data = {}
    for symbol in tqdm(symbols, desc="Downloading"):
        try:
            df = fetch_data_with_cache(symbol, args.start, args.end)
            if len(df) >= 500:
                symbol_data[symbol] = df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    
    print(f"\nLoaded {len(symbol_data)} symbols with sufficient data")
    
    # PRE-COMPUTE INDICATORS (Major speedup!)
    print(f"\n{'='*60}")
    print("PRE-COMPUTING TECHNICAL INDICATORS")
    print(f"{'='*60}")
    for symbol in tqdm(symbol_data.keys(), desc="Computing indicators"):
        symbol_data[symbol] = add_technical_indicators(symbol_data[symbol])
    print("Indicators pre-computed for all symbols!")
    
    if len(symbol_data) == 0:
        print("ERROR: No valid symbols to train!")
        return
    
    # Train
    trainer = ParallelTrainer(
        symbol_data,
        num_parallel=args.parallel,
        learning_rate=LEARNING_RATE,
        use_noisy=USE_NOISY
    )
    
    trainer.train(
        episodes_per_symbol=args.episodes,
        resume=not args.fresh
    )
    
    # Backtest
    if do_backtest:
        print(f"\n{'='*60}")
        print("RUNNING BACKTEST")
        print(f"{'='*60}")
        
        model_path = "models/SHARED_dqn_best.pth"
        try:
            bt = Backtester(model_path)
            
            # Test on a sample of symbols
            test_symbols = list(symbol_data.keys())[:10]
            all_results = []
            
            for symbol in test_symbols:
                try:
                    results = bt.run_backtest(symbol, backtest_start, backtest_end)
                    if results:
                        bt.print_results(results)
                        all_results.append(results)
                except Exception as e:
                    print(f"Backtest failed for {symbol}: {e}")
            
            # Summary
            if all_results:
                avg_return = np.mean([r['total_return_pct'] for r in all_results])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
                avg_alpha = np.mean([r['alpha'] for r in all_results])
                
                print(f"\n{'='*60}")
                print("BACKTEST SUMMARY")
                print(f"{'='*60}")
                print(f"Symbols tested: {len(all_results)}")
                print(f"Avg Return: {avg_return:+.2f}%")
                print(f"Avg Sharpe: {avg_sharpe:.2f}")
                print(f"Avg Alpha vs B&H: {avg_alpha:+.2f}%")
                print(f"{'='*60}")
                
        except Exception as e:
            print(f"Backtest failed: {e}")


if __name__ == "__main__":
    main()
