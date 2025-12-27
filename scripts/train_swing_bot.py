import sys
import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

# Force UTF-8 for Windows
if sys.platform == 'win32':
    # type: ignore
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
# Override Window Size for Swing Trading
TrainingConfig.WINDOW_SIZE = 60

from src.environments.vector_env import VectorizedTradingEnv  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.core.indicators import add_technical_indicators  # noqa: E402


def load_swing_data():
    print("📥 Loading Historical Swing Data (1D)...")
    pattern = os.path.join("data/historical_swing", "*_1D.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No data found in data/historical_swing!")
        print("💡 Please run 'python scripts/download_swing_data.py' first.")
        return None, None
        
    data_list = []
    price_list = []
    
    # We want to use the full history (10 years ~ 2500 days)
    # We will pad shorter sequences to the max length found
    max_seq_len = 0
    
    processed_dfs = []
    
    for f in tqdm(files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(f)
            
            # Standardize Columns
            df.columns = [c.lower() for c in df.columns]
            col_map = {'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
            
            # Capitalize existing full names
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col.capitalize()] = df[col]

            # Add Indicators
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 200: # Minimum history
                continue
                
            processed_dfs.append(df)
            if len(df) > max_seq_len:
                max_seq_len = len(df)
                
        except Exception:
            continue
            
    if not processed_dfs:
        return None, None
        
    print(f"📏 Max Sequence Length: {max_seq_len}")
    
    for df in processed_dfs:
        # Extract Raw Prices (Close)
        raw_close = df['Close'].values
        
        # Extract Features
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'rsi', 'macd', 'macd_signal', 'adx', 'sma_20', 'ema_12'
        ]
        
        # Ensure all exist
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < 5:
            continue
            
        features = df[valid_cols].values
        
        # Normalize
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        norm_features = (features - mean) / std
        
        # Pad to max_seq_len (Pre-padding)
        curr_len = len(norm_features)
        if curr_len < max_seq_len:
            pad_len = max_seq_len - curr_len
            # Pad with first value to simulate "waiting"
            # Or pad with zeros. Edge padding is safer for time series.
            norm_features = np.pad(norm_features, ((pad_len, 0), (0, 0)), mode='edge')
            raw_close = np.pad(raw_close, (pad_len, 0), mode='edge')
        
        data_list.append(norm_features)
        price_list.append(raw_close)
            
    # Stack into Tensors
    # Shape: (Num_Envs, Time, Features)
    data_tensor = torch.FloatTensor(np.array(data_list))
    price_tensor = torch.FloatTensor(np.array(price_list))
    
    return data_tensor, price_tensor

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Swing Bot Training on {device}...")
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir="logs/runs/swing_experiment_1")
    
    # 1. Load Data
    data, prices = load_swing_data()
    if data is None:
        return
    
    print(f"📊 Data Shape: {data.shape}")
    
    # 2. Initialize Environment
    env = VectorizedTradingEnv(data, prices, device=device)
    
    # 3. Initialize Ensemble Agent
    num_features = data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )
    
    # Check for existing checkpoints
    # RESUME FROM BEST MODEL (Episode 121)
    print("🔄 Resuming from Best Model (Episode 121)...")
    agent.load("models/swing_best")
    start_episode = 121  # So next episode is 122
    
    # FREEZE EARLY LAYERS - Only train fusion + output layers
    print("🧊 Freezing early layers (TS, Vision, Text heads)...")
    for sub_agent in agent.agents:
        model = sub_agent.policy_net
        
        # Freeze Time-Series Head
        for param in model.ts_head.parameters():
            param.requires_grad = False
        
        # Freeze Vision Head  
        for param in model.vision_head.parameters():
            param.requires_grad = False
        
        # Freeze Text Head
        for param in model.text_head.parameters():
            param.requires_grad = False
        
        # Keep fusion + value/advantage streams trainable
        for param in model.fusion_layer.parameters():
            param.requires_grad = True
        for param in model.value_stream.parameters():
            param.requires_grad = True
        for param in model.advantage_stream.parameters():
            param.requires_grad = True
        
        # Reinitialize optimizer with only trainable params
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        sub_agent.optimizer = torch.optim.Adam(trainable_params, lr=1e-5)  # Lower LR for fine-tuning
    
    # Use LOW epsilon for fine-tuning from a good checkpoint
    epsilon_decay = 0.999995
    calculated_epsilon = 0.20  # Increased to 0.20 as requested to force exploration
    
    print(f"🔄 Setting Epsilon to {calculated_epsilon:.6f} for fine-tuning")
    for sub_agent in agent.agents:
        sub_agent.epsilon = calculated_epsilon
        sub_agent.epsilon_decay = epsilon_decay

    # start_episode = 0
    # checkpoints = glob.glob("models/swing_ep*_balanced.pth")
    # if checkpoints:
    #     ep_nums = []
    #     for cp in checkpoints:
    #         try:
    #             num = int(cp.split("swing_ep")[1].split("_")[0])
    #             ep_nums.append(num)
    #         except:
    #             pass
        
    #     if ep_nums:
    #         latest_ep = max(ep_nums)
    #         print(f"🔄 Found checkpoint for Episode {latest_ep}. Resuming...")
    #         agent.load(f"models/swing_ep{latest_ep}")
    #         start_episode = latest_ep
            
    #         # Recalculate Epsilon
    #         # Total Steps ~ 200 * 2500 = 500,000
    #         # Decay = 0.999995
    #         epsilon_decay = 0.999995
    #         total_steps_passed = start_episode * env.total_steps
    #         calculated_epsilon = max(0.05, 1.0 * (epsilon_decay ** total_steps_passed))
            
    #         print(f"🔄 Adjusting Epsilon to {calculated_epsilon:.6f}")
    #         for sub_agent in agent.agents:
    #             sub_agent.epsilon = calculated_epsilon
    #             sub_agent.epsilon_decay = epsilon_decay
    # else:
    #     # Set initial decay for fresh start
    #     epsilon_decay = 0.999995
        for sub_agent in agent.agents:
            sub_agent.epsilon_decay = epsilon_decay

    # 4. Training Loop
    EPISODES = 200
    
    # Dummy Text Data
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    
    # Track Best Reward
    # Set to 134.0 (Previous High) so we don't overwrite with lower scores
    best_reward = 134.0 
    print(f"🏆 Best Reward Baseline set to {best_reward}. Will only save if exceeded.")

    for episode in range(start_episode, EPISODES):
        state = env.reset()  # (Envs, Window, Features)
        total_reward = 0
        
        steps_per_episode = env.total_steps
        
        pbar = tqdm(total=steps_per_episode, desc=f"Episode {episode+1}/{EPISODES}")
        
        for step_idx in range(steps_per_episode):
            # Ensemble Action
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)
            
            # Step Env
            next_state, rewards, dones, _ = env.step(actions)
            
            # Store Experience & Train
            state_cpu = state.cpu().numpy()
            next_state_cpu = next_state.cpu().numpy()
            actions_cpu = actions.cpu().numpy()
            rewards_cpu = rewards.cpu().numpy()
            dones_cpu = dones.cpu().numpy()
            
            for i in range(env.num_envs):
                s = (state_cpu[i], dummy_text_ids[i].cpu().numpy(), dummy_text_mask[i].cpu().numpy())
                ns = (next_state_cpu[i], dummy_text_ids[i].cpu().numpy(), dummy_text_mask[i].cpu().numpy())
                agent.remember(s, actions_cpu[i], rewards_cpu[i], ns, dones_cpu[i])
            
            losses = agent.train_step()
            
            global_step = episode * steps_per_episode + step_idx
            if losses:
                for name, loss in losses.items():
                    writer.add_scalar(f"Loss/{name}", loss, global_step)
            
            writer.add_scalar("Epsilon/Balanced", agent.balanced.epsilon, global_step)
            
            state = next_state
            total_reward += rewards.sum().item()
                
            pbar.update(1)
            pbar.set_postfix({'Reward': f"{total_reward:.4f}", 'Eps': f"{agent.balanced.epsilon:.2f}"})
            
        pbar.close()
        writer.add_scalar("Reward/Episode_Total", total_reward, episode)
        
        # Save Best Model
        if total_reward > best_reward:
            best_reward = total_reward
            print(f"🏆 New Best Reward: {best_reward:.4f}! Saving model...")
            agent.save("models/swing_best")
        
        # Regular Checkpoint
        if (episode + 1) % 10 == 0:
            save_prefix = f"models/swing_ep{episode+1}"
            agent.save(save_prefix)
            print(f"💾 Saved models to {save_prefix}_*.pth")
            
    writer.close()

if __name__ == "__main__":
    train()
