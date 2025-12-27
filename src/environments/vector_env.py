import torch
from config.settings import TrainingConfig


class VectorizedTradingEnv:
    """
    GPU-Accelerated Trading Environment.
    Simulates hundreds of tickers simultaneously using PyTorch Tensors.
    Eliminates CPU-GPU data transfer bottleneck.
    """
    def __init__(self, data_tensor, price_tensor, device="cuda"):
        """
        data_tensor: (Num_Envs, Total_Time, Features) - Normalized
        price_tensor: (Num_Envs, Total_Time) - Raw Prices
        """
        self.device = device
        self.data = data_tensor.to(device)
        self.prices = price_tensor.to(device) # Store raw prices
        self.num_envs, self.total_steps, self.num_features = self.data.shape
        
        self.window_size = TrainingConfig.WINDOW_SIZE
        self.initial_balance = 10000.0

        # Execution realism (configurable; defaults keep existing behavior)
        # BPS = basis points, e.g. 10 bps = 0.10%
        self.transaction_cost_bps = float(getattr(TrainingConfig, 'TRANSACTION_COST_BPS', 0.0))
        self.slippage_bps = float(getattr(TrainingConfig, 'SLIPPAGE_BPS', 0.0))
        self._total_cost_rate = (self.transaction_cost_bps + self.slippage_bps) / 10000.0
        
        # State Tracking (Vectorized)
        self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.balance = torch.full((self.num_envs,), self.initial_balance, device=device)
        self.shares = torch.zeros(self.num_envs, device=device)
        self.entry_price = torch.zeros(self.num_envs, device=device)
        self.in_position = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        
        # Performance Tracking
        self.total_reward = torch.zeros(self.num_envs, device=device)
        self.trades = torch.zeros(self.num_envs, device=device)

        # Equity tracking (for PnL-based reward)
        self.prev_equity = torch.full((self.num_envs,), self.initial_balance, device=device)
        
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
        self.prev_equity.fill_(self.initial_balance)
        
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

        PnL-BASED REWARD (V3):
        - Reward is log-return of equity between t and t+1 (mark-to-market)
        - Includes transaction costs + slippage via effective entry/exit prices
        - Removes shaped bonuses/penalties that can be reward-hacked
        """
        # 1. Get Current Prices from RAW Prices
        row_indices = torch.arange(self.num_envs, device=self.device)
        current_prices = self.prices[row_indices, self.current_step]

        # Next-step prices for mark-to-market reward (equity at t+1)
        next_step = torch.clamp(self.current_step + 1, max=self.total_steps - 1)
        next_prices = self.prices[row_indices, next_step]
        
        # 2. Logic masking & action sanitization
        rewards = torch.zeros(self.num_envs, device=self.device)

        # Invalid actions (no-ops) should be penalized so the agent can't game the reward
        # by spamming SELL while flat or BUY while already in a position.
        invalid_sell_flat = (actions == 2) & (~self.in_position)
        invalid_buy_in_pos = (actions == 1) & (self.in_position)

        # Action masking (setup change): force invalid actions to HOLD.
        # This prevents policy collapse into meaningless no-ops.
        # We still apply an explicit penalty below using the *original* invalid masks.
        effective_actions = actions
        if invalid_sell_flat.any() or invalid_buy_in_pos.any():
            effective_actions = actions.clone()
            effective_actions[invalid_sell_flat | invalid_buy_in_pos] = 0

        # Equity at time t (before action). When flat, shares=0 so equity==balance.
        equity_t = self.balance + self.shares * current_prices
        
        # BUY Logic (Action 1) & Not in Position
        buy_mask = (effective_actions == 1) & (~self.in_position)
        if buy_mask.any():
            # Apply costs on entry by increasing effective entry price.
            # This reduces apparent PnL on future exits and helps discourage churn.
            if self._total_cost_rate > 0:
                self.entry_price[buy_mask] = current_prices[buy_mask] * (1.0 + self._total_cost_rate)
            else:
                self.entry_price[buy_mask] = current_prices[buy_mask]

            # Full notional position sizing (all-in)
            safe_entry = torch.clamp(self.entry_price[buy_mask], min=0.01)
            self.shares[buy_mask] = self.balance[buy_mask] / safe_entry
            self.balance[buy_mask] = 0.0

            self.in_position[buy_mask] = True
            self.trades[buy_mask] += 1
            
        # SELL Logic (Action 2) & In Position
        sell_mask = (effective_actions == 2) & (self.in_position)
        if sell_mask.any():
            # Apply costs on exit by reducing effective exit price.
            if self._total_cost_rate > 0:
                exit_price = current_prices[sell_mask] * (1.0 - self._total_cost_rate)
            else:
                exit_price = current_prices[sell_mask]

            # Close position into cash
            self.balance[sell_mask] = self.shares[sell_mask] * exit_price
            self.shares[sell_mask] = 0.0
            self.entry_price[sell_mask] = 0.0
            self.in_position[sell_mask] = False
            
        # Reward is equity log-return from t -> t+1 (mark-to-market on next_prices)
        equity_tp1 = self.balance + self.shares * next_prices
        safe_equity_t = torch.clamp(equity_t, min=1e-6)
        reward_lr = torch.log(torch.clamp(equity_tp1 / safe_equity_t, min=1e-6))
        rewards = reward_lr

        # Small penalty for invalid actions (keeps gradients pointing away from nonsense)
        # Adjusted to -0.000001 (0.1 bps) to minimize impact on PnL visibility while still discouraging invalid moves.
        if invalid_sell_flat.any():
            rewards[invalid_sell_flat] -= 0.000001
        if invalid_buy_in_pos.any():
            rewards[invalid_buy_in_pos] -= 0.000001
        
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
            self.balance[dones] = self.initial_balance
            self.shares[dones] = 0.0
            self.entry_price[dones] = 0.0
            self.in_position[dones] = False
            # We don't reset total_reward here to track episode, but for RL training loop we just need 'done' flag
            
        next_obs = self._get_observation()
        
        return next_obs, rewards, dones, {}

