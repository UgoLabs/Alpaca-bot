import torch
from config.settings import TrainingConfig


class VectorizedTradingEnv:
    """
    GPU-Accelerated Trading Environment.
    Simulates hundreds of tickers simultaneously using PyTorch Tensors.
    Eliminates CPU-GPU data transfer bottleneck.
    """
    def __init__(self, data_tensor, price_tensor, device="cuda", position_pct: float = 1.0):
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

        # Cached indices/tensors to avoid per-step reallocations
        self._row_indices = torch.arange(self.num_envs, device=device)
        self._window_offsets = torch.arange(self.window_size, device=device).view(1, -1)

        # Position sizing: fraction of available cash to deploy on BUY.
        # Default 1.0 preserves existing all-in behavior used during training.
        try:
            self.position_pct = float(position_pct)
        except Exception:
            self.position_pct = 1.0
        self.position_pct = max(0.0, min(1.0, self.position_pct))

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
        
        # Exit reward shaping parameters
        self.exit_profit_bonus = float(getattr(TrainingConfig, 'EXIT_PROFIT_BONUS', 0.001))
        self.holding_loss_penalty = float(getattr(TrainingConfig, 'HOLDING_LOSS_PENALTY', 0.0001))
        self.loss_threshold_pct = float(getattr(TrainingConfig, 'LOSS_THRESHOLD_PCT', 0.02))
        
        # Track how long we've been in a position (for holding penalty)
        self.steps_in_position = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        
        # ===== REGIME-AWARE REWARD SHAPING =====
        # Instead of oversampling bear markets, we give BONUS rewards for
        # correct actions during downtrends. This teaches the model without slowing training.
        self.regime_reward_mult = float(getattr(TrainingConfig, 'REGIME_REWARD_MULT', 2.0))  # 2x reward during bear
        self.trend_lookback = int(getattr(TrainingConfig, 'TREND_LOOKBACK', 20))  # Days to detect trend
        
        # Reward scaling for better Q-value spread
        self.reward_scale = float(getattr(TrainingConfig, 'REWARD_SCALE', 100.0))  # Scale up tiny log-returns

        # ===== TRAILING STOP & PROFIT-TAKE (applied during training) =====
        # DISABLED for training - let the model learn exit timing itself
        # These can be enabled in backtest/production for risk management
        self.use_trailing_stop = bool(getattr(TrainingConfig, 'USE_TRAILING_STOP', False))
        self.trailing_stop_atr_mult = float(getattr(TrainingConfig, 'TRAILING_STOP_ATR_MULT', 3.0))
        self.use_profit_take = bool(getattr(TrainingConfig, 'USE_PROFIT_TAKE', False))
        self.profit_take_atr_mult = float(getattr(TrainingConfig, 'PROFIT_TAKE_ATR_MULT', 4.0))
        # Track highest price since entry (for trailing stop)
        self.high_since_entry = torch.zeros(self.num_envs, device=device)
        # New for Gen11: Track Peak Equity for Drawdown Penalty
        self.peak_equity = torch.full((self.num_envs,), self.initial_balance, device=device)

        # Simple ATR approximation: rolling volatility (we'll compute per-step)
        self.atr_window = int(getattr(TrainingConfig, 'ATR_WINDOW', 14))
        
        # Reset all
        self.reset()
        
    def reset(self):
        # Random start times for each env to decorrelate them
        # Ensure we have enough history (window_size) and enough future (at least 1 step)
        # Modified for flexibility with shorter backtest sequences
        max_start = max(self.window_size + 1, self.total_steps - 5)
        
        # If still invalid (sequence too short), force safe start
        if max_start <= self.window_size:
            max_start = self.window_size + 1
            
        self.current_step = torch.randint(
            self.window_size, max_start, (self.num_envs,), device=self.device
        )
        
        self.balance.fill_(self.initial_balance)
        self.shares.zero_()
        self.entry_price.zero_()
        self.in_position.zero_()
        self.total_reward.zero_()
        self.prev_equity.fill_(self.initial_balance)
        self.steps_in_position.zero_()
        self.high_since_entry.zero_()
        # New for Gen11
        self.peak_equity.fill_(self.initial_balance)
        
        return self._get_observation()
        
    def _get_observation(self):
        # Vectorized gather:
        # time_idx: (Envs, Window) where each row is [t-window, ..., t-1]
        time_idx = (self.current_step.view(-1, 1) - self.window_size) + self._window_offsets
        time_idx = torch.clamp(time_idx, min=0, max=self.total_steps - 1)

        # Gather along time dimension (dim=1)
        gather_idx = time_idx.unsqueeze(2).expand(-1, -1, self.num_features)
        return self.data.gather(dim=1, index=gather_idx)
        
    def step(self, actions):
        """
        actions: (Num_Envs,) tensor of 0 (Hold), 1 (Buy), 2 (Sell)

        PnL-BASED REWARD (V3):
        - Reward is log-return of equity between t and t+1 (mark-to-market)
        - Includes transaction costs + slippage via effective entry/exit prices
        - Removes shaped bonuses/penalties that can be reward-hacked
        """
        # 1. Get Current Prices from RAW Prices
        current_prices = self.prices[self._row_indices, self.current_step]

        # Next-step prices for mark-to-market reward (equity at t+1)
        next_step = torch.clamp(self.current_step + 1, max=self.total_steps - 1)
        next_prices = self.prices[self._row_indices, next_step]
        
        # 2. Logic masking & action sanitization
        rewards = torch.zeros(self.num_envs, device=self.device)

        # Invalid actions (no-ops) should be penalized so the agent can't game the reward
        # by spamming SELL while flat or BUY while already in a position.
        invalid_sell_flat = (actions == 2) & (~self.in_position)
        invalid_buy_in_pos = (actions == 1) & (self.in_position)

        # ===== TRAILING STOP & PROFIT-TAKE: Force SELL if triggered =====
        forced_sell_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.in_position.any() and (self.use_trailing_stop or self.use_profit_take):
            # Update high_since_entry for trailing stop
            self.high_since_entry[self.in_position] = torch.maximum(
                self.high_since_entry[self.in_position],
                current_prices[self.in_position]
            )
            # Compute simple ATR approximation: use price volatility over recent window
            # For speed, approximate ATR as % of current price * fixed factor
            approx_atr = current_prices * 0.02  # ~2% of price as ATR proxy (conservative)
            
            if self.use_trailing_stop:
                # Trailing stop: price < high_since_entry - ATR * mult
                stop_price = self.high_since_entry - (approx_atr * self.trailing_stop_atr_mult)
                stop_triggered = self.in_position & (current_prices < stop_price)
                forced_sell_mask = forced_sell_mask | stop_triggered
            
            if self.use_profit_take:
                # Profit take: price > entry + ATR * mult
                profit_target = self.entry_price + (approx_atr * self.profit_take_atr_mult)
                profit_triggered = self.in_position & (current_prices > profit_target)
                forced_sell_mask = forced_sell_mask | profit_triggered

        # Action masking (setup change): force invalid actions to HOLD.
        # This prevents policy collapse into meaningless no-ops.
        # We still apply an explicit penalty below using the *original* invalid masks.
        effective_actions = actions.clone()
        if invalid_sell_flat.any() or invalid_buy_in_pos.any():
            effective_actions[invalid_sell_flat | invalid_buy_in_pos] = 0
        
        # Override action to SELL for forced exits (trailing stop or profit take)
        if forced_sell_mask.any():
            effective_actions[forced_sell_mask] = 2  # Force SELL action

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

            # Position sizing (default all-in). Keep the undeployed cash in balance.
            safe_entry = torch.clamp(self.entry_price[buy_mask], min=0.01)
            deploy_cash = self.balance[buy_mask] * self.position_pct
            self.shares[buy_mask] = deploy_cash / safe_entry
            self.balance[buy_mask] = self.balance[buy_mask] - deploy_cash

            self.in_position[buy_mask] = True
            self.trades[buy_mask] += 1
            
            # Initialize high_since_entry for trailing stop tracking
            self.high_since_entry[buy_mask] = current_prices[buy_mask]
            
        # SELL Logic (Action 2) & In Position
        sell_mask = (effective_actions == 2) & (self.in_position)
        sell_pnl_pct = torch.zeros(self.num_envs, device=self.device)  # Track PnL for reward shaping
        if sell_mask.any():
            # Apply costs on exit by reducing effective exit price.
            if self._total_cost_rate > 0:
                exit_price = current_prices[sell_mask] * (1.0 - self._total_cost_rate)
            else:
                exit_price = current_prices[sell_mask]

            # Calculate profit/loss percentage before closing
            entry_prices_sell = self.entry_price[sell_mask]
            sell_pnl_pct[sell_mask] = (exit_price - entry_prices_sell) / torch.clamp(entry_prices_sell, min=0.01)
            
            # Close position into cash (add proceeds to any remaining balance)
            self.balance[sell_mask] = self.balance[sell_mask] + (self.shares[sell_mask] * exit_price)
            
            self.shares[sell_mask] = 0.0
            self.entry_price[sell_mask] = 0.0
            self.in_position[sell_mask] = False
            self.steps_in_position[sell_mask] = 0
            self.high_since_entry[sell_mask] = 0.0  # Reset trailing stop tracker
            
        # INCREMENT STEPS IN POSITION for those still holding
        self.steps_in_position[self.in_position] += 1
        
        # Reward is equity log-return from t -> t+1 (mark-to-market on next_prices)
        equity_tp1 = self.balance + self.shares * next_prices
        safe_equity_t = torch.clamp(equity_t, min=1e-6)
        reward_lr = torch.log(torch.clamp(equity_tp1 / safe_equity_t, min=1e-6))
        
        # ===== GEN11: DRAWDOWN PENALTY =====
        # Update Peak Equity
        self.peak_equity = torch.maximum(self.peak_equity, equity_tp1)
        
        # Calculate Drawdown %
        current_drawdown = (self.peak_equity - equity_tp1) / self.peak_equity
        
        # Quadratic Penalty: Penalize deep drawdowns heavily
        # If DD is 10%, penalty is 0.1^2 * 10 = 0.1 (Huge!)
        # With multiplier 4.0: 0.1^2 * 4.0 = 0.04
        dd_penalty = (current_drawdown ** 2) * 4.0
        
        # ===== REWARD SCALING: Scale up tiny log-returns for better Q-value spread =====
        # Log-returns are typically -0.001 to +0.001. Scale to -0.1 to +0.1 range.
        rewards = reward_lr * self.reward_scale
        
        # Apply Drawdown Penalty
        rewards -= dd_penalty * self.reward_scale
        
        # ===== GEN11 UPDATE: Reward holding winners (Fix Overtrading) =====
        if self.in_position.any():
             unrealized_pnl_pct = (current_prices - self.entry_price) / torch.clamp(self.entry_price, min=0.01)
             # Bonus for holding a position that is > 0.5% profitable
             # This encourages longer duration trades vs scalping
             winning_positions = self.in_position & (unrealized_pnl_pct > 0.005) 
             if winning_positions.any():
                 rewards[winning_positions] += 0.001 * self.reward_scale
        
        # ===== REGIME-AWARE REWARD SHAPING =====
        # Detect bear market: price < SMA(lookback) for this step
        # Give BONUS rewards for correct bear market behavior:
        # - HOLD/SELL when flat in downtrend = good (avoided loss)
        # - SELL when in position during downtrend = good (cut losses)
        # - BUY in downtrend that works = extra bonus (caught the reversal)
        lookback_step = torch.clamp(self.current_step - self.trend_lookback, min=0)
        lookback_prices = self.prices[self._row_indices, lookback_step]
        in_downtrend = current_prices < lookback_prices * 0.95  # Price 5%+ below lookback = bear
        
        if in_downtrend.any():
            # Correct behavior in downtrend gets multiplied reward
            # SELL in downtrend while in position = smart exit
            smart_exit = sell_mask & in_downtrend & (sell_pnl_pct < 0)  # Cut losses
            if smart_exit.any():
                # Reduce the loss penalty - cutting losses early is GOOD
                rewards[smart_exit] *= 0.5  # Halve the negative reward for smart exits
            
            # HOLD when flat in downtrend = correctly avoided buying the dip too early
            smart_hold = (effective_actions == 0) & (~self.in_position) & in_downtrend
            if smart_hold.any():
                rewards[smart_hold] += 0.01 * self.reward_scale  # Small bonus for patience
        
        # ===== EXIT REWARD SHAPING: Bonus for profitable exits =====
        profitable_exits = sell_mask & (sell_pnl_pct > 0)
        if profitable_exits.any():
            profit_bonus = torch.clamp(sell_pnl_pct[profitable_exits], max=0.10) * self.exit_profit_bonus * 10 * self.reward_scale
            rewards[profitable_exits] += profit_bonus
        
        # ===== HOLDING LOSS PENALTY: Penalize holding losing positions =====
        if self.in_position.any():
            unrealized_pnl_pct = (current_prices - self.entry_price) / torch.clamp(self.entry_price, min=0.01)
            losing_positions = self.in_position & (unrealized_pnl_pct < -self.loss_threshold_pct)
            if losing_positions.any():
                # Penalty scales with how long we've been holding the loser
                hold_time_factor = torch.clamp(self.steps_in_position[losing_positions].float() / 10.0, max=5.0)
                loss_penalty = self.holding_loss_penalty * hold_time_factor * self.reward_scale
                rewards[losing_positions] -= loss_penalty

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
            max_start = max(self.window_size + 1, self.total_steps - 5)
            self.current_step[dones] = torch.randint(
                self.window_size, max_start, (dones.sum(),), device=self.device
            )
            self.balance[dones] = self.initial_balance
            self.shares[dones] = 0.0
            self.entry_price[dones] = 0.0
            self.in_position[dones] = False
            self.steps_in_position[dones] = 0
            self.high_since_entry[dones] = 0.0  # Reset trailing stop tracker
            self.peak_equity[dones] = self.initial_balance # Reset Peak Equity
            # We don't reset total_reward here to track episode, but for RL training loop we just need 'done' flag
            
        next_obs = self._get_observation()
        
        return next_obs, rewards, dones, {
            'equity': equity_tp1, 
            'action': effective_actions, 
            'entry_price': self.entry_price, 
            'shares': self.shares,
            'realized_pnl': sell_pnl_pct,
            'sell_mask': sell_mask
        }

