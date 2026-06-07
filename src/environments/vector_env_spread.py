"""GPU vector env: call debit spread PnL from precomputed Alpaca spread marks."""
from __future__ import annotations

import torch
from config.settings import OptionsTraderConfig, TrainingConfig


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


class VectorizedMultiStrategySpreadEnv:
    """
    Multi-strategy spread env: on BUY, open first tradable structure in priority order.

    spread_marks / entry_premium / tradable: (N, T, S)
    is_credit: (S,) bool — credit structures use liability MTM and inverted entry reward.
    """

    def __init__(
        self,
        data_tensor,
        spread_marks,
        entry_premium,
        tradable,
        is_credit,
        device="cuda",
        position_pct: float = 1.0,
        strategy_order: tuple[int, ...] | None = None,
    ):
        self.device = device
        self.data = data_tensor.to(device)
        self.spread_marks = spread_marks.to(device)
        self.entry_premium = entry_premium.to(device)
        self.tradable = tradable.to(device)
        self.is_credit = is_credit.to(device)
        self.num_envs, self.total_steps, self.num_features = self.data.shape
        self.num_strategies = self.spread_marks.shape[2]
        self.strategy_order = strategy_order or tuple(range(self.num_strategies))

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
        self._spread_width = float(getattr(OptionsTraderConfig, "SPREAD_WIDTH", 5.0))

        self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.balance = torch.full((self.num_envs,), self.initial_balance, device=device)
        self.in_position = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.active_strat = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self.premium_stored = torch.zeros(self.num_envs, device=device)
        self.collateral_stored = torch.zeros(self.num_envs, device=device)
        self.steps_in_position = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.reset()

    def _credit_max_risk(self, prem: float) -> float:
        """Max loss per contract for a credit spread (matches live/backtest sizing)."""
        credit_per_share = prem / 100.0
        return max(50.0, (self._spread_width - credit_per_share) * 100.0)

    def _mark_active(self, step_idx: torch.Tensor) -> torch.Tensor:
        marks = torch.zeros(self.num_envs, device=self.device)
        if not self.in_position.any():
            return marks
        rows = self._row_indices[self.in_position]
        steps = step_idx[self.in_position]
        strats = self.active_strat[self.in_position]
        marks[self.in_position] = self.spread_marks[rows, steps, strats]
        return marks

    def _equity(self, step_idx: torch.Tensor) -> torch.Tensor:
        mark = self._mark_active(step_idx)
        eq = self.balance.clone()
        if not self.in_position.any():
            return eq
        held = self.in_position
        credit_pos = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        credit_pos[held] = self.is_credit[self.active_strat[held]]
        eq[held & ~credit_pos] += mark[held & ~credit_pos]
        eq[held & credit_pos] += (
            self.collateral_stored[held & credit_pos] - mark[held & credit_pos]
        )
        return eq

    def reset(self):
        max_start = max(self.window_size + 1, self.total_steps - 5)
        if max_start <= self.window_size:
            max_start = self.window_size + 1
        self.current_step = torch.randint(
            self.window_size, max_start, (self.num_envs,), device=self.device
        )
        self.balance.fill_(self.initial_balance)
        self.in_position.zero_()
        self.active_strat.fill_(-1)
        self.premium_stored.zero_()
        self.collateral_stored.zero_()
        self.steps_in_position.zero_()
        return self._get_observation()

    def _get_observation(self):
        time_idx = (self.current_step.view(-1, 1) - self.window_size) + self._window_offsets
        time_idx = torch.clamp(time_idx, min=0, max=self.total_steps - 1)
        gather_idx = time_idx.unsqueeze(2).expand(-1, -1, self.num_features)
        return self.data.gather(dim=1, index=gather_idx)

    def _pick_strategy(self, buy_intent: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        chosen = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        rows = self._row_indices
        for s in self.strategy_order:
            m = self.spread_marks[rows, step, s]
            p = self.entry_premium[rows, step, s]
            t = self.tradable[rows, step, s]
            ok = (
                buy_intent
                & (chosen < 0)
                & t
                & (p > 0)
                & (m > 0)
            )
            chosen[ok] = s
        return chosen

    def step(self, actions):
        rewards = torch.zeros(self.num_envs, device=self.device)
        step = self.current_step
        next_step = torch.clamp(step + 1, max=self.total_steps - 1)

        buy_intent_raw = (actions == 1) & (~self.in_position)
        invalid_sell_flat = (actions == 2) & (~self.in_position)
        invalid_buy_in_pos = (actions == 1) & self.in_position

        forced_sell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.max_hold > 0 and self.in_position.any():
            forced_sell = self.in_position & (self.steps_in_position >= self.max_hold)

        effective = actions.clone()
        effective[invalid_sell_flat | invalid_buy_in_pos] = 0
        if forced_sell.any():
            effective[forced_sell] = 2

        buy_intent = (effective == 1) & (~self.in_position)
        chosen = self._pick_strategy(buy_intent, step)
        invalid_buy_no_data = buy_intent & (chosen < 0)
        effective[invalid_buy_no_data] = 0

        equity_t = self._equity(step)
        mark_t = self._mark_active(step)

        opened_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        candidates = (buy_intent & (chosen >= 0)).nonzero(as_tuple=False).squeeze(-1)
        if candidates.numel() > 0:
            if candidates.dim() == 0:
                candidates = candidates.unsqueeze(0)
            for i in candidates.tolist():
                s = int(chosen[i].item())
                prem = self.entry_premium[i, step[i], s]
                if prem <= 0:
                    continue
                if bool(self.is_credit[s].item()):
                    collateral = self._credit_max_risk(float(prem.item()))
                    credit_net = prem * (1.0 - self._total_cost_rate)
                    net_reserve = collateral - float(credit_net.item())
                    if net_reserve > self.balance[i] * self.position_pct:
                        continue
                    self.balance[i] += credit_net - collateral
                    self.collateral_stored[i] = collateral
                    self.in_position[i] = True
                    self.active_strat[i] = s
                    self.premium_stored[i] = prem
                    self.steps_in_position[i] = 0
                    opened_mask[i] = True
                else:
                    cost = prem * (1.0 + self._total_cost_rate)
                    if self.balance[i] * self.position_pct >= cost:
                        self.balance[i] -= cost
                        self.in_position[i] = True
                        self.active_strat[i] = s
                        self.premium_stored[i] = prem
                        self.steps_in_position[i] = 0
                        opened_mask[i] = True

        sell_mask = (effective == 2) & self.in_position
        if sell_mask.any():
            for i in sell_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                if isinstance(i, bool):
                    continue
                s = int(self.active_strat[i].item())
                m = self.spread_marks[i, step[i], s]
                if bool(self.is_credit[s].item()):
                    self.balance[i] -= m * (1.0 + self._total_cost_rate)
                    self.balance[i] += self.collateral_stored[i]
                    self.collateral_stored[i] = 0.0
                else:
                    self.balance[i] += m * (1.0 - self._total_cost_rate)
                self.in_position[i] = False
                self.active_strat[i] = -1
                self.premium_stored[i] = 0.0
                self.steps_in_position[i] = 0

        self.steps_in_position[self.in_position] += 1

        equity_tp1 = self._equity(next_step)
        safe_t = torch.clamp(equity_t, min=1e-6)
        rewards = torch.log(torch.clamp(equity_tp1 / safe_t, min=1e-6)) * self.reward_scale

        if self.entry_reward_coef != 0.0 and opened_mask.any():
            for i in opened_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                if isinstance(i, bool):
                    continue
                s = int(self.active_strat[i].item())
                prem = self.premium_stored[i]
                fwd = min(int(step[i].item()) + self.entry_lookahead, self.total_steps - 1)
                fwd_mark = self.spread_marks[i, fwd, s]
                if bool(self.is_credit[s].item()):
                    fwd_ret = (prem - fwd_mark) / max(prem, 1e-6)
                else:
                    fwd_ret = (fwd_mark - prem) / max(prem, 1e-6)
                rewards[i] += self.entry_reward_coef * fwd_ret * self.reward_scale

        if invalid_sell_flat.any() or invalid_buy_in_pos.any() or invalid_buy_no_data.any():
            bad = invalid_sell_flat | invalid_buy_in_pos | invalid_buy_no_data
            rewards[bad] -= self.invalid_action_penalty * self.reward_scale

        self.current_step = next_step
        done = self.current_step >= (self.total_steps - 1)
        return self._get_observation(), rewards, done, {}
