"""
Train individual models per symbol, then ensemble them for production.

This approach:
1. Train 100 episodes on EACH symbol (fast, ~2 min per symbol)
2. Each model learns that symbol's patterns deeply
3. Ensemble all models for voting-based decisions

Usage:
    python scripts/train_per_symbol.py --symbols AAPL,MSFT,NVDA --episodes 100
    python scripts/train_per_symbol.py --top-n 50 --episodes 100  # Train on top 50 by volume
"""

import sys
import os
import glob
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# GPU performance knobs
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

from config.settings import TrainingConfig
TrainingConfig.WINDOW_SIZE = 60

from src.environments.vector_env import VectorizedTradingEnv
from src.agents.multimodal_agent import MultimodalTradingAgent
from src.core.indicators import add_technical_indicators


def load_single_symbol_data(symbol: str, cutoff_date: str = "2023-12-31"):
    """Load data for a single symbol."""
    pattern = os.path.join("data/historical_swing", f"{symbol}_1D.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None, None
    
    try:
        df = pd.read_csv(files[0])
        
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        col_map = {'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}
        for k, v in col_map.items():
            if k in df.columns:
                df[v] = df[k]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col.capitalize()] = df[col]

        # Handle date
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
        elif 'timestamp' in df.columns:
            df['Date'] = pd.to_datetime(df['timestamp'])
            df.set_index('Date', inplace=True)

        # Add indicators
        df = add_technical_indicators(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Apply cutoff
        cutoff_ts = pd.to_datetime(cutoff_date)
        df = df[df.index < cutoff_ts]
        
        if len(df) < 200:
            return None, None
        
        # Extract features
        raw_close = df['Close'].values
        
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12'
        ]
        
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < 5:
            return None, None
            
        features = df[valid_cols].values
        
        # Normalize
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        norm_features = (features - mean) / std
        
        # Shape: (1, Time, Features) - single symbol
        data_tensor = torch.FloatTensor(norm_features).unsqueeze(0)
        price_tensor = torch.FloatTensor(raw_close).unsqueeze(0)
        
        return data_tensor, price_tensor
        
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return None, None


def get_available_symbols():
    """Get all available symbols from historical_swing data."""
    pattern = os.path.join("data/historical_swing", "*_1D.csv")
    files = glob.glob(pattern)
    symbols = [os.path.basename(f).replace("_1D.csv", "") for f in files]
    return sorted(symbols)


def train_single_symbol(
    symbol: str,
    episodes: int = 100,
    train_every: int = 10,
    store_every: int = 5,
    initial_epsilon: float = 0.95,
    epsilon_decay: float = 0.98,
    cutoff_date: str = "2023-12-31",
    early_stopping: int = 30,
):
    """Train a model for a single symbol."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    data, prices = load_single_symbol_data(symbol, cutoff_date)
    if data is None:
        print(f"  Skipping {symbol} - insufficient data")
        return None
    
    # Initialize environment (single symbol)
    env = VectorizedTradingEnv(data, prices, device=device)
    
    # Initialize agent
    num_features = data.shape[2]
    agent = MultimodalTradingAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )
    agent.epsilon = initial_epsilon
    
    # Training loop
    seq_len = 64
    dummy_text_ids = torch.zeros((1, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((1, seq_len), dtype=torch.long).to(device)
    
    best_reward = float("-inf")
    no_improve = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = env.total_steps
        step_counter = 0
        
        for _ in range(steps):
            # Get action
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            
            # Step
            next_state, rewards, dones, infos = env.step(actions)
            
            # Store experience
            if step_counter % store_every == 0:
                agent.store_transition(
                    state[0].cpu().numpy(),
                    actions[0].item(),
                    rewards[0].item(),
                    next_state[0].cpu().numpy(),
                    dones[0].item(),
                    dummy_text_ids[0].cpu().numpy(),
                    dummy_text_mask[0].cpu().numpy()
                )
            
            # Train
            if step_counter % train_every == 0:
                agent.train_step()
            
            total_reward += rewards.sum().item()
            state = next_state
            step_counter += 1
        
        # Decay epsilon per episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= epsilon_decay
        
        # Check best
        if total_reward > best_reward:
            best_reward = total_reward
            no_improve = 0
            # Save best for this symbol
            save_path = f"models/symbol_models/{symbol}_best"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)
        else:
            no_improve += 1
        
        # Early stopping
        if early_stopping > 0 and no_improve >= early_stopping:
            break
    
    print(f"  {symbol}: Best={best_reward:.2f} after {episode+1} episodes")
    return best_reward


def main():
    parser = argparse.ArgumentParser(description="Train individual models per symbol")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated list of symbols")
    parser.add_argument("--top-n", type=int, default=0, help="Train on top N symbols (by name, alphabetically)")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per symbol")
    parser.add_argument("--train-every", type=int, default=10, help="Train every N steps")
    parser.add_argument("--store-every", type=int, default=5, help="Store every N steps")
    parser.add_argument("--initial-epsilon", type=float, default=0.95, help="Starting epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.98, help="Epsilon decay per episode")
    parser.add_argument("--cutoff-date", type=str, default="2023-12-31", help="Training cutoff date")
    parser.add_argument("--early-stopping", type=int, default=30, help="Early stop after N no-improve episodes")
    args = parser.parse_args()
    
    # Get symbols to train
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.top_n > 0:
        all_symbols = get_available_symbols()
        symbols = all_symbols[:args.top_n]
    else:
        symbols = get_available_symbols()
    
    print(f"Training {len(symbols)} symbol models...")
    print(f"Episodes per symbol: {args.episodes}")
    print(f"Early stopping: {args.early_stopping}")
    print("-" * 50)
    
    results = {}
    for i, symbol in enumerate(tqdm(symbols, desc="Symbols")):
        print(f"\n[{i+1}/{len(symbols)}] Training {symbol}...")
        reward = train_single_symbol(
            symbol=symbol,
            episodes=args.episodes,
            train_every=args.train_every,
            store_every=args.store_every,
            initial_epsilon=args.initial_epsilon,
            epsilon_decay=args.epsilon_decay,
            cutoff_date=args.cutoff_date,
            early_stopping=args.early_stopping,
        )
        if reward is not None:
            results[symbol] = reward
    
    # Save results summary
    summary_path = "models/symbol_models/training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Trained {len(results)} symbol models")
    print(f"Results saved to {summary_path}")
    
    # Show top performers
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 performers:")
    for sym, rew in sorted_results[:10]:
        print(f"  {sym}: {rew:.2f}")


if __name__ == "__main__":
    main()
