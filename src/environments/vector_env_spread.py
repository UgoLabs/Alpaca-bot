"""GPU vector env: call debit spread PnL from precomputed Alpaca spread marks."""
from __future__ import annotations

import torch
from config.settings import TrainingConfig


class VectorizedSpreadEnv:
    """
    One env per underlying symbol. Reward = log-return of spread position equity.

    data_tensor: (N, T, F) normalized features
    spread_marks: (N, T) $ mark per contract (long - short) * 100
    entry_premium: (N, T) debit paid to open on that day
    tradable: (N, T) True when Alpaca had both leg closes
    """

    def __init__(
        self,
        data_tensor,
        spread_marks,
        entry_premium,
        tradable,
        device="cuda",
        position_pct: float = 1.0,
    ):
        self.device = device
        self.data = data_tensor.to(device)
        self.spread_marks = spread_marks.to(device)
        self.entry_premium = entry_premium.to(device)
        self.tradable = tradable.to(device)
        self.num_envs, self.total_steps, self.num_features = self.data.shape

        self.window_size = TrainingConfig.WINDOW_SIZE
        self.initial_balance = 10000.0
        self._row_indices = torch.arange(self.num_envs, device=device)
        self._window_offsets = torch.arange(self.window_size, device=device).view(1, -1)

        self.position_pct = max(0.0, min(1.0, float(position_pct)))
        self.transaction_cost_bps = float(getattr(TrainingConfig, "TRANSACTION_COST_BPS", 0.0))
        self.slippage_bps = float(getattr(TrainingConfig, "SLIPPAGE_BPS", 0.0))
        self._total_cost_rate = (self.transaction_cost_bps + self.slippage_bps) / 10000.0
        self.reward_scale = float(getattr(TrainingConfig, "REWARD_SCALE", 100.0))
        self.invalid_action_penalty = float(
            getattr(TrainingConfig, "INVALID_ACTION_PENALTY", 0.0003)
        )
        self.entry_reward_coef = float(getattr(TrainingConfig, "ENTRY_REWARD_COEF", 0.0))
        self.entry_lookahead = int(getattr(TrainingConfig, "ENTRY_LOOKAHEAD_BARS", 5))
        self.max_hold = int(getattr(TrainingConfig, "MAX_HOLD_BARS", 0) or 0)

        self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.balance = torch.full((self.num_envs,), self.initial_balance, device=device)
        self.in_position = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.premium_paid = torch.zeros(self.num_envs, device=device)
        self.steps_in_position = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.prev_equity = torch.full((self.num_envs,), self.initial_balance, device=device)
        self.reset()

    def _spread_at(self, step_idx: torch.Tensor) -> torch.Tensor:
        return self.spread_marks[self._row_indices, step_idx]

    def _equity(self, step_idx: torch.Tensor) -> torch.Tensor:
        mark = self._spread_at(step_idx)
        held = self.in_position.float() * mark
        return self.balance + held

    def reset(self):
        max_start = max(self.window_size + 1, self.total_steps - 5)
        if max_start <= self.window_size:
            max_start = self.window_size + 1
        self.current_step = torch.randint(
            self.window_size, max_start, (self.num_envs,), device=self.device
        )
        self.balance.fill_(self.initial_balance)
        self.in_position.zero_()
        self.premium_paid.zero_()
        self.steps_in_position.zero_()
        self.prev_equity.fill_(self.initial_balance)
        return self._get_observation()

    def _get_observation(self):
        time_idx = (self.current_step.view(-1, 1) - self.window_size) + self._window_offsets
        time_idx = torch.clamp(time_idx, min=0, max=self.total_steps - 1)
        gather_idx = time_idx.unsqueeze(2).expand(-1, -1, self.num_features)
        return self.data.gather(dim=1, index=gather_idx)

    def step(self, actions):
        rewards = torch.zeros(self.num_envs, device=self.device)
        step = self.current_step
        next_step = torch.clamp(step + 1, max=self.total_steps - 1)

        mark_t = self._spread_at(step)
        mark_tp1 = self._spread_at(next_step)
        prem_t = self.entry_premium[self._row_indices, step]
        can_trade = self.tradable[self._row_indices, step] & (prem_t > 0) & (mark_t > 0)

        invalid_sell_flat = (actions == 2) & (~self.in_position)
        invalid_buy_in_pos = (actions == 1) & (self.in_position)
        invalid_buy_no_data = (actions == 1) & (~self.in_position) & (~can_trade)

        forced_sell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.max_hold > 0 and self.in_position.any():
            forced_sell = self.in_position & (self.steps_in_position >= self.max_hold)

        effective = actions.clone()
        effective[invalid_sell_flat | invalid_buy_in_pos | invalid_buy_no_data] = 0
        if forced_sell.any():
            effective[forced_sell] = 2

        equity_t = self._equity(step)

        buy_intent = (effective == 1) & (~self.in_position) & can_trade
        cost = prem_t * (1.0 + self._total_cost_rate)
        deploy = self.balance * self.position_pct
        buy_mask = buy_intent & (deploy >= cost)
        if buy_mask.any():
            self.balance[buy_mask] -= cost[buy_mask]
            self.premium_paid[buy_mask] = prem_t[buy_mask]
            self.in_position[buy_mask] = True
            self.steps_in_position[buy_mask] = 0

        sell_mask = (effective == 2) & self.in_position
        if sell_mask.any():
            credit = mark_t[sell_mask] * (1.0 - self._total_cost_rate)
            self.balance[sell_mask] += credit
            self.in_position[sell_mask] = False
            self.premium_paid[sell_mask] = 0.0
            self.steps_in_position[sell_mask] = 0

        self.steps_in_position[self.in_position] += 1

        equity_tp1 = self.balance + self.in_position.float() * mark_tp1
        safe_t = torch.clamp(equity_t, min=1e-6)
        rewards = torch.log(torch.clamp(equity_tp1 / safe_t, min=1e-6)) * self.reward_scale

        if self.entry_reward_coef != 0.0 and buy_mask.any():
            fwd = torch.clamp(step + self.entry_lookahead, max=self.total_steps - 1)
            fwd_mark = self._spread_at(fwd)
            fwd_ret = (fwd_mark - prem_t) / torch.clamp(prem_t, min=1e-6)
            rewards[buy_mask] += (
                self.entry_reward_coef * fwd_ret[buy_mask] * self.reward_scale
            )

        if invalid_sell_flat.any() or invalid_buy_in_pos.any() or invalid_buy_no_data.any():
            bad = invalid_sell_flat | invalid_buy_in_pos | invalid_buy_no_data
            rewards[bad] -= self.invalid_action_penalty * self.reward_scale

        self.current_step = next_step
        done = self.current_step >= (self.total_steps - 1)
        return self._get_observation(), rewards, done, {}
