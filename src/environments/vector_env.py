import torch
import numpy as np
from config.settings import TrainingConfig

class VectorizedTradingEnv:
    """
    GPU-Accelerated Trading Environment.
    Simulates hundreds of tickers simultaneously using PyTorch Tensors.
    Eliminates CPU-GPU data transfer bottleneck.
    """
    def __init__(self, data_tensor, device="cuda"):
        """
        data_tensor: (Num_Envs, Total_Time, Features)
        """
        self.device = device
        self.data = data_tensor.to(device)
        self.num_envs, self.total_steps, self.num_features = self.data.shape
        
        self.window_size = TrainingConfig.WINDOW_SIZE
        self.initial_balance = 10000.0
        
        # State Tracking (Vectorized)
        self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.balance = torch.full((self.num_envs,), self.initial_balance, device=device)
        self.shares = torch.zeros(self.num_envs, device=device)
        self.entry_price = torch.zeros(self.num_envs, device=device)
        self.in_position = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        
        # Performance Tracking
        self.total_reward = torch.zeros(self.num_envs, device=device)
        self.trades = torch.zeros(self.num_envs, device=device)
        
        # Reset all
        self.reset()
        
    def reset(self):
        # Random start times for each env to decorrelate them
        # Ensure we have enough history (window_size) and enough future (at least 1 step)
        max_start = self.total_steps - 100 
        self.current_step = torch.randint(
            self.window_size, max_start, (self.num_envs,), device=self.device
        )
        
        self.balance.fill_(self.initial_balance)
        self.shares.zero_()
        self.entry_price.zero_()
        self.in_position.zero_()
        self.total_reward.zero_()
        
        return self._get_observation()
        
    def _get_observation(self):
        # Extract window for each env efficiently
        # This is tricky in PyTorch without a loop. We use gather or specialized unfolding.
        # Simple Loop is surprisingly fast on GPU if batch is large, but unfold is better.
        
        # Shape: (Num_Envs, Window, Features)
        # We construct indices: [current_step-window ... current_step]
        
        # Fastest way: Loop for window size (small constant 60) vs Loop for Envs (large 258)
        # We loop window size.
        
        frames = []
        for i in range(self.window_size):
            # t = current_step - window + i
            indices = self.current_step - self.window_size + i
            # self.data shape: (Envs, Time, Feat)
            # data[env_idx, indices[env_idx], :]
            
            # We use advanced indexing
            row_indices = torch.arange(self.num_envs, device=self.device)
            frames.append(self.data[row_indices, indices, :])
            
        return torch.stack(frames, dim=1) # (Envs, Window, Features)
        
    def step(self, actions):
        """
        actions: (Num_Envs,) tensor of 0 (Hold), 1 (Buy), 2 (Sell)
        """
        # 1. Get Current Prices
        # We specifically encoded 'Close' as feature 3 (Change this index based on your data!)
        # Assuming Data is Open, High, Low, Close, Volume...
        CLOSE_IDX = 3 
        row_indices = torch.arange(self.num_envs, device=self.device)
        current_prices = self.data[row_indices, self.current_step, CLOSE_IDX]
        
        # 2. Logic Masking
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # BUY Logic (Action 1) & Not in Position
        buy_mask = (actions == 1) & (~self.in_position)
        if buy_mask.any():
            # Buy logic...
            self.entry_price[buy_mask] = current_prices[buy_mask]
            cost = self.entry_price[buy_mask]
            # Use 20% of balance per trade (Vectorized) or fixed? Fixed 20% for now
            # Actually, simplify: Always buy 1 share for RL signal (Reward is pure price delta %)
            # Or use portfolio. Let's use % return as reward.
            self.in_position[buy_mask] = True
            
        # SELL Logic (Action 2) & In Position
        sell_mask = (actions == 2) & (self.in_position)
        if sell_mask.any():
            exit_price = current_prices[sell_mask]
            entry_price = self.entry_price[sell_mask]
            
            # Reward = Log Return or Pct Return
            pct_return = (exit_price - entry_price) / entry_price
            rewards[sell_mask] = pct_return
            
            self.in_position[sell_mask] = False
            self.trades[sell_mask] += 1
            
        # HOLD Logic (Action 0)
        # Optional: Small penalty for holding? No.
        
        # 3. Step Time
        self.current_step += 1
        
        # 4. Done Condition
        dones = self.current_step >= (self.total_steps - 1)
        
        # Auto-Reset finished envs
        if dones.any():
            # Reset specifically those that are done
            # This allows continuous infinite training without stopping
            self.current_step[dones] = torch.randint(
                self.window_size, self.total_steps - 100, (dones.sum(),), device=self.device
            )
            self.in_position[dones] = False
            # We don't reset total_reward here to track episode, but for RL training loop we just need 'done' flag
            
        next_obs = self._get_observation()
        
        return next_obs, rewards, dones, {}

