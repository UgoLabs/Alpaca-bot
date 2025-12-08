import numpy as np
import pandas as pd
import yfinance as yf
import torch
import os
import json
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from swing_environment import SwingTradingEnv
from swing_model import DuelingDQN, NStepReplayBuffer, PrioritizedReplayBuffer
from utils import get_state_size
from backtest import Backtester

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
    
    # Download if not cached
    try:
        # Try with new yfinance parameters first
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        except TypeError:
            # Fallback for older yfinance
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
        if len(df) > 0:
            df.to_csv(cache_file)
        return df
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()

# =============================================================================
# Hyperparameters
# =============================================================================
EPISODES = 200                # More episodes for better learning
BATCH_SIZE = 64
GAMMA = 0.99
N_STEP = 3                    # N-step returns
EPSILON_START = 1.0           # Only used if use_noisy=False
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 5        # Episodes between hard updates
SOFT_UPDATE_TAU = 0.005       # Soft update rate
MEMORY_SIZE = 100000
LEARNING_RATE = 0.0003        # Slightly lower for stability
WINDOW_SIZE = 20
USE_NOISY = True              # Use Noisy Networks for exploration
USE_PER = True                # Use Prioritized Experience Replay
USE_N_STEP = True             # Use N-step returns
SAVE_REPLAY_BUFFER = True     # Save replay buffer to disk

# Risk Management (passed to environment)
RISK_PER_TRADE = 0.02         # Risk 2% per trade
MAX_POSITION_PCT = 0.25       # Max 25% in single position

# =============================================================================
# Portfolio Loading
# =============================================================================
def load_portfolio(portfolio_file='../my_portfolio.txt'):
    """Load symbols from portfolio file."""
    # Try multiple paths
    paths_to_try = [
        portfolio_file,
        'my_portfolio.txt',
        '../my_portfolio.txt',
        os.path.join(os.path.dirname(__file__), '..', 'my_portfolio.txt')
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"Loaded {len(symbols)} symbols from {path}")
            return symbols
    
    print("Portfolio file not found, using default symbols")
    return ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# =============================================================================
# Training Function
# =============================================================================
def train_agent(symbol='SPY', start_date='2010-01-01', end_date='2024-01-01', 
                episodes=EPISODES, resume=True):
    """
    Train the Dueling DQN agent on historical data.
    """
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 1. Fetch Data
    print(f"\n{'='*60}")
    print(f"SWING TRADING DQN TRAINER (PyTorch)")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Episodes: {episodes}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")
    
    print(f"Fetching data for {symbol}...")
    try:
        df = fetch_data_with_cache(symbol, start_date, end_date)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    if len(df) == 0:
        print("ERROR: No data found.")
        return None
    
    if len(df) < 500:
        print(f"WARNING: Only {len(df)} days of data. Recommend at least 500 days.")

    # 2. Initialize Environment
    env = SwingTradingEnv(
        df, 
        window_size=WINDOW_SIZE,
        risk_per_trade=RISK_PER_TRADE,
        max_position_pct=MAX_POSITION_PCT
    )
    
    state_size = env.state_size
    action_size = env.action_space.n
    
    print(f"State Size: {state_size}")
    print(f"Action Size: {action_size}")
    print(f"Training Data: {len(df)} days")
    
    # 3. Initialize Agent
    agent = DuelingDQN(state_size, action_size, learning_rate=LEARNING_RATE, use_noisy=USE_NOISY)
    
    # Initialize Replay Buffer
    if USE_N_STEP:
        memory = NStepReplayBuffer(max_size=MEMORY_SIZE, n_step=N_STEP, gamma=GAMMA)
    elif USE_PER:
        memory = PrioritizedReplayBuffer(max_size=MEMORY_SIZE)
    else:
        from swing_model import ReplayBuffer
        memory = ReplayBuffer(max_size=MEMORY_SIZE)
    
    # Try to load existing replay buffer
    buffer_path = f"models/{symbol}_replay_buffer.pkl"
    if resume and SAVE_REPLAY_BUFFER and os.path.exists(buffer_path):
        memory.load(buffer_path)
    
    # Try to load existing model
    model_path = f"models/{symbol}_dqn_best.pth"
    if resume and os.path.exists(model_path):
        print(f"Resuming from existing model: {model_path}")
        agent.load(model_path)
    
    epsilon = EPSILON_START
    best_net_worth = env.initial_balance
    best_sharpe = -np.inf
    
    # Training metrics
    training_log = []
    
    # 4. Training Loop
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        done = False
        
        episode_start = datetime.now()
        
        while not done:
            # Action
            action = agent.act(state, epsilon)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            # Store Experience
            experience = (state, action, reward, next_state, done)
            
            # Calculate TD-error for priority (if needed)
            if USE_PER or USE_N_STEP:
                agent.model.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                    
                    q_current = agent.model(state_tensor)[0][action].item()
                    
                    if done:
                        td_target = reward
                    else:
                        q_next = agent.target_model(next_state_tensor).max(1)[0].item()
                        td_target = reward + (GAMMA ** N_STEP if USE_N_STEP else GAMMA) * q_next
                    
                    td_error = td_target - q_current
                agent.model.train()
                memory.add(experience, td_error)
            else:
                memory.add(experience)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train
            if memory.size() > BATCH_SIZE:
                # Only PrioritizedReplayBuffer returns (batch, indices, weights)
                if USE_PER and not USE_N_STEP:
                    minibatch, indices, weights = memory.sample(BATCH_SIZE)
                else:
                    minibatch = memory.sample(BATCH_SIZE)
                    weights = np.ones(BATCH_SIZE)
                    indices = None
                
                states = np.array([i[0] for i in minibatch])
                actions = np.array([i[1] for i in minibatch])
                rewards = np.array([i[2] for i in minibatch])
                next_states = np.array([i[3] for i in minibatch])
                dones = np.array([i[4] for i in minibatch])
                
                # Convert to tensors
                states_t = torch.FloatTensor(states).to(agent.device)
                next_states_t = torch.FloatTensor(next_states).to(agent.device)
                
                # Double DQN Logic
                agent.model.eval()
                with torch.no_grad():
                    # Select action using online model
                    next_actions = agent.model(next_states_t).argmax(1).unsqueeze(1)
                    # Evaluate action using target model
                    target_q_next = agent.target_model(next_states_t).gather(1, next_actions).squeeze(1)
                    
                    # Current Q values (for calculating targets)
                    current_q = agent.model(states_t)
                agent.model.train()
                
                # Construct targets
                targets = current_q.clone().detach()
                gamma_n = GAMMA ** N_STEP if USE_N_STEP else GAMMA
                
                td_errors = []
                for i in range(BATCH_SIZE):
                    old_val = targets[i][actions[i]].item()
                    if dones[i]:
                        target_val = rewards[i]
                    else:
                        target_val = rewards[i] + gamma_n * target_q_next[i].item()
                    
                    targets[i][actions[i]] = target_val
                    td_errors.append(target_val - old_val)
                
                # Train step
                loss = agent.train(states, targets.cpu().numpy(), weights)
                total_loss += loss
                
                # Update priorities
                if indices is not None:
                    memory.update_priorities(indices, td_errors)
                
                # Soft update target network
                agent.soft_update_target_model(SOFT_UPDATE_TAU)
        
        # End of Episode
        episode_time = (datetime.now() - episode_start).total_seconds()
        avg_loss = total_loss / max(steps, 1)
        
        # Get environment metrics
        metrics = env.get_metrics()
        
        # Hard update target network periodically
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
            
        # Decay Epsilon
        if not USE_NOISY and epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY
        
        # Log metrics
        log_entry = {
            'episode': e + 1,
            'total_reward': total_reward,
            'net_worth': env.net_worth,
            'total_return': metrics.get('total_return', 0) * 100,
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0) * 100,
            'sharpe': metrics.get('sharpe', 0),
            'avg_loss': avg_loss,
            'epsilon': epsilon if not USE_NOISY else 0,
            'steps': steps,
            'time': episode_time,
            'memory_size': memory.size()
        }
        training_log.append(log_entry)
        
        # Print progress
        win_rate = metrics.get('win_rate', 0) * 100
        total_trades = metrics.get('total_trades', 0)
        sharpe = metrics.get('sharpe', 0)
        
        print(f"Ep {e+1:3d}/{episodes} | Net Worth: ${env.net_worth:,.2f} | "
              f"Return: {metrics.get('total_return', 0)*100:+.1f}% | "
              f"Trades: {total_trades} | Win: {win_rate:.0f}% | "
              f"Sharpe: {sharpe:.2f} | Time: {episode_time:.1f}s")
        
        # Save best model
        if sharpe > best_sharpe and total_trades >= 5:
            best_sharpe = sharpe
            best_net_worth = env.net_worth
            agent.save(f"models/{symbol}_dqn_best.pth")
            print(f"  *** New Best! Sharpe: {best_sharpe:.2f}, Net Worth: ${best_net_worth:,.2f} ***")
        
        # Save periodically
        if (e + 1) % 20 == 0:
            agent.save(f"models/{symbol}_dqn_episode_{e+1}.pth")
            
            if SAVE_REPLAY_BUFFER:
                memory.save(buffer_path)
            
            with open(f"logs/{symbol}_training_log.json", 'w') as f:
                json.dump(training_log, f, indent=2)

    # Save Final Model and Buffer
    agent.save(f"models/{symbol}_dqn_final.pth")
    if SAVE_REPLAY_BUFFER:
        memory.save(buffer_path)
    
    with open(f"logs/{symbol}_training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Completed!")
    print(f"Best Sharpe Ratio: {best_sharpe:.2f}")
    print(f"Best Net Worth: ${best_net_worth:,.2f}")
    print(f"Models saved to: models/")
    print(f"Logs saved to: logs/{symbol}_training_log.json")
    print("="*60)
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='Train Swing Trading DQN')
    parser.add_argument('--symbol', type=str, default=None, help='Single stock symbol (overrides portfolio)')
    parser.add_argument('--portfolio', action='store_true', help='Train on all symbols in my_portfolio.txt')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Training episodes per symbol')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore existing model)')
    parser.add_argument('--shared-model', action='store_true', help='Use single shared model for all symbols')
    parser.add_argument('--train-years', type=int, default=None, help='Years of training data (e.g. 10)')
    parser.add_argument('--backtest-years', type=int, default=None, help='Years of backtest data (e.g. 4)')
    
    args = parser.parse_args()
    
    # Handle Date Logic
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
    
    # Training Phase
    if args.symbol:
        # Train single symbol
        train_agent(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            episodes=args.episodes,
            resume=not args.fresh
        )
        
        # Backtest Phase
        if do_backtest:
            print(f"\nRunning Backtest for {args.symbol}...")
            model_path = f"models/{args.symbol}_dqn_best.pth"
            try:
                bt = Backtester(model_path)
                results = bt.run_backtest(args.symbol, backtest_start, backtest_end)
                if results:
                    bt.print_results(results)
            except Exception as e:
                print(f"Backtest failed: {e}")

    elif args.portfolio or args.shared_model:
        # Train on all portfolio symbols
        train_portfolio(
            start_date=args.start,
            end_date=args.end,
            episodes_per_symbol=args.episodes,
            resume=not args.fresh,
            shared_model=args.shared_model
        )
        
        # Backtest Phase
        if do_backtest:
            print("\nRunning Portfolio Backtest...")
            symbols = load_portfolio()
            
            if args.shared_model:
                model_path = "models/SHARED_dqn_best.pth"
                try:
                    bt = Backtester(model_path)
                    for symbol in symbols:
                        results = bt.run_backtest(symbol, backtest_start, backtest_end)
                        if results:
                            bt.print_results(results)
                except Exception as e:
                    print(f"Shared model backtest failed: {e}")
            else:
                for symbol in symbols:
                    model_path = f"models/{symbol}_dqn_best.pth"
                    try:
                        bt = Backtester(model_path)
                        results = bt.run_backtest(symbol, backtest_start, backtest_end)
                        if results:
                            bt.print_results(results)
                    except Exception as e:
                        print(f"Backtest failed for {symbol}: {e}")
    else:
        # Default: train SPY
        train_agent(
            symbol='SPY',
            start_date=args.start,
            end_date=args.end,
            episodes=args.episodes,
            resume=not args.fresh
        )


def train_portfolio(start_date='2010-01-01', end_date='2024-01-01', 
                    episodes_per_symbol=50, resume=True, shared_model=False):
    """
    Train on all symbols in the portfolio.
    """
    symbols = load_portfolio()
    
    print(f"\n{'='*60}")
    print(f"MULTI-SYMBOL TRAINING")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Episodes per symbol: {episodes_per_symbol}")
    print(f"Shared model: {shared_model}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Results tracking
    all_results = {}
    
    if shared_model:
        # Train one model on all symbols (round-robin)
        train_shared_model(symbols, start_date, end_date, episodes_per_symbol, resume)
    else:
        # Train individual models per symbol
        for i, symbol in enumerate(symbols):
            print(f"\n{'#'*60}")
            print(f"Training {i+1}/{len(symbols)}: {symbol}")
            print(f"{'#'*60}")
            
            try:
                agent = train_agent(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    episodes=episodes_per_symbol,
                    resume=resume
                )
                all_results[symbol] = 'success'
            except Exception as e:
                print(f"ERROR training {symbol}: {e}")
                all_results[symbol] = f'failed: {str(e)}'
    
    # Save summary
    summary_path = "logs/portfolio_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'symbols': symbols,
            'episodes_per_symbol': episodes_per_symbol,
            'shared_model': shared_model,
            'results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PORTFOLIO TRAINING COMPLETE")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


def train_shared_model(symbols, start_date, end_date, episodes_per_symbol, resume):
    """
    Train a single shared model on all symbols using round-robin training.
    """
    print("\n" + "="*60)
    print("SHARED MODEL TRAINING (Round-Robin)")
    print("="*60)
    
    # Fetch all data first
    print("\nFetching data for all symbols...")
    symbol_data = {}
    for symbol in tqdm(symbols, desc="Downloading"):
        try:
            df = fetch_data_with_cache(symbol, start_date, end_date)
            if len(df) >= 500:
                symbol_data[symbol] = df
            else:
                # print(f"  Skipping {symbol}: only {len(df)} days of data")
                pass
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
    
    valid_symbols = list(symbol_data.keys())
    print(f"\nTraining on {len(valid_symbols)} symbols with sufficient data")
    
    if len(valid_symbols) == 0:
        print("ERROR: No valid symbols to train on!")
        return
    
    # Initialize shared agent
    state_size = get_state_size(WINDOW_SIZE)
    action_size = 3
    agent = DuelingDQN(state_size, action_size, learning_rate=LEARNING_RATE, use_noisy=USE_NOISY)
    
    # Shared replay buffer
    if USE_N_STEP:
        memory = NStepReplayBuffer(max_size=MEMORY_SIZE * 2, n_step=N_STEP, gamma=GAMMA)
    elif USE_PER:
        memory = PrioritizedReplayBuffer(max_size=MEMORY_SIZE * 2)
    else:
        from swing_model import ReplayBuffer
        memory = ReplayBuffer(max_size=MEMORY_SIZE * 2)
    
    # Try to load existing shared model
    model_path = "models/SHARED_dqn_best.pth"
    buffer_path = "models/SHARED_replay_buffer.pkl"
    
    if resume and os.path.exists(model_path):
        print(f"Resuming from existing model: {model_path}")
        agent.load(model_path)
    
    if resume and SAVE_REPLAY_BUFFER and os.path.exists(buffer_path):
        memory.load(buffer_path)
    
    # Training loop - round robin through symbols
    total_episodes = episodes_per_symbol * len(valid_symbols)
    epsilon = EPSILON_START
    best_avg_sharpe = -np.inf
    training_log = []
    
    print(f"\nTotal episodes: {total_episodes}")
    print("Starting round-robin training...\n")
    
    for episode in range(total_episodes):
        # Select symbol for this episode (round-robin)
        symbol = valid_symbols[episode % len(valid_symbols)]
        df = symbol_data[symbol]
        
        # Create environment
        env = SwingTradingEnv(
            df, 
            window_size=WINDOW_SIZE,
            risk_per_trade=RISK_PER_TRADE,
            max_position_pct=MAX_POSITION_PCT
        )
        
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        done = False
        
        while not done:
            # Action
            action = agent.act(state, epsilon)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            experience = (state, action, reward, next_state, done)
            
            # Calculate TD-error for priority
            if USE_PER or USE_N_STEP:
                agent.model.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                    
                    q_current = agent.model(state_tensor)[0][action].item()
                    
                    if done:
                        td_target = reward
                    else:
                        q_next = agent.target_model(next_state_tensor).max(1)[0].item()
                        td_target = reward + (GAMMA ** N_STEP if USE_N_STEP else GAMMA) * q_next
                    
                    td_error = td_target - q_current
                agent.model.train()
                memory.add(experience, td_error)
            else:
                memory.add(experience)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train
            if memory.size() > BATCH_SIZE:
                # Only PrioritizedReplayBuffer returns (batch, indices, weights)
                if USE_PER and not USE_N_STEP:
                    minibatch, indices, weights = memory.sample(BATCH_SIZE)
                else:
                    minibatch = memory.sample(BATCH_SIZE)
                    weights = np.ones(BATCH_SIZE)
                    indices = None
                
                states = np.array([i[0] for i in minibatch])
                actions = np.array([i[1] for i in minibatch])
                rewards = np.array([i[2] for i in minibatch])
                next_states = np.array([i[3] for i in minibatch])
                dones = np.array([i[4] for i in minibatch])
                
                # Convert to tensors
                states_t = torch.FloatTensor(states).to(agent.device)
                next_states_t = torch.FloatTensor(next_states).to(agent.device)
                
                # Double DQN Logic
                agent.model.eval()
                with torch.no_grad():
                    next_actions = agent.model(next_states_t).argmax(1).unsqueeze(1)
                    target_q_next = agent.target_model(next_states_t).gather(1, next_actions).squeeze(1)
                    current_q = agent.model(states_t)
                agent.model.train()
                
                targets = current_q.clone().detach()
                gamma_n = GAMMA ** N_STEP if USE_N_STEP else GAMMA
                
                td_errors = []
                for i in range(BATCH_SIZE):
                    old_val = targets[i][actions[i]].item()
                    if dones[i]:
                        target_val = rewards[i]
                    else:
                        target_val = rewards[i] + gamma_n * target_q_next[i].item()
                    
                    targets[i][actions[i]] = target_val
                    td_errors.append(target_val - old_val)
                
                # Train step
                loss = agent.train(states, targets.cpu().numpy(), weights)
                total_loss += loss
                
                if indices is not None:
                    memory.update_priorities(indices, td_errors)
                
                agent.soft_update_target_model(SOFT_UPDATE_TAU)
        
        # End of episode
        metrics = env.get_metrics()
        
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
        
        if not USE_NOISY and epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY
        
        # Log
        log_entry = {
            'episode': episode + 1,
            'symbol': symbol,
            'total_reward': total_reward,
            'net_worth': env.net_worth,
            'total_return': metrics.get('total_return', 0) * 100,
            'sharpe': metrics.get('sharpe', 0),
            'trades': metrics.get('total_trades', 0)
        }
        training_log.append(log_entry)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            recent_sharpes = [l['sharpe'] for l in training_log[-50:]]
            avg_sharpe = np.mean(recent_sharpes) if recent_sharpes else 0
            
            print(f"Ep {episode+1:4d}/{total_episodes} | {symbol:5s} | "
                  f"Return: {metrics.get('total_return', 0)*100:+6.1f}% | "
                  f"Sharpe: {metrics.get('sharpe', 0):5.2f} | "
                  f"Avg Sharpe (50ep): {avg_sharpe:5.2f}")
            
            # Save best model
            if avg_sharpe > best_avg_sharpe and episode > 50:
                best_avg_sharpe = avg_sharpe
                agent.save("models/SHARED_dqn_best.pth")
                print(f"  *** New Best! Avg Sharpe: {best_avg_sharpe:.2f} ***")
        
        # Save periodically
        if (episode + 1) % 100 == 0:
            agent.save(f"models/SHARED_dqn_episode_{episode+1}.pth")
            if SAVE_REPLAY_BUFFER:
                memory.save(buffer_path)
            with open("logs/SHARED_training_log.json", 'w') as f:
                json.dump(training_log, f, indent=2)
    
    # Final save
    agent.save("models/SHARED_dqn_final.pth")
    if SAVE_REPLAY_BUFFER:
        memory.save(buffer_path)
    with open("logs/SHARED_training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Shared Model Training Complete!")
    print(f"Best Avg Sharpe: {best_avg_sharpe:.2f}")
    print(f"Model saved to: models/SHARED_dqn_best.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
