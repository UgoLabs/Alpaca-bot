"""
Portfolio backtest: options model signals → multi-strategy spreads (Alpaca option bars).

Bullish BUY: call_debit → bull_put_credit → long_call (conf gate).
Bearish SELL (not held): put_debit → bear_call_credit when ENABLE_BEARISH_OPENS.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/backtest_options_portfolio.py --disk-cache
  .\\.venv\\Scripts\\python.exe scripts/backtest_options_portfolio.py \\
      --call-debit-only --disk-cache
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OPTIONS_MODEL_PATH, OptionsTraderConfig, TrainingConfig  # noqa: E402

TrainingConfig.WINDOW_SIZE = 60

from scripts.backtest_swing_portfolio import (  # noqa: E402
    WINDOW,
    _load_and_infer,
    _normalize_model_prefix,
    _spy_day_change_pct,
)
from src.data.options_historical import (  # noqa: E402
    OPTIONS_DATA_MIN_DATE,
    OptionSpreadBarCache,
    build_occ_symbol,
    load_swing_calendar,
    make_option_bar_cache_from_disk,
    make_option_bar_cache_from_env,
    pick_expiration,
)
from src.execution.options_spread import spread_width_for_spot  # noqa: E402
from src.execution.options_strategies import (  # noqa: E402
    SpreadKind,
    _LEG_FN,
    widths_to_try,
)

DEFAULT_MODEL = _normalize_model_prefix(str(OPTIONS_MODEL_PATH))
# Pass either the ensemble prefix (models/foo_ep30) or a single leg (…_balanced.pth).
DEFAULT_WATCHLIST = os.path.join("config", "watchlists", "options_backtest_short.txt")

_KIND_FROM_NAME = {k.value: k for k in SpreadKind}


def _strike_step(spot: float, width: float) -> float:
    return width if spot >= 200 else (2.5 if spot < 100 else width)


def _strike_ladder(spot: float, width: float) -> list[float]:
    step = _strike_step(spot, width)
    center = round(spot / step) * step
    if center <= 0:
        center = max(step, spot)
    return [float(center + i * step) for i in range(-8, 9)]


def _widths_for_kind(
    spot: float,
    kind: SpreadKind,
    spread_width: float,
    *,
    scale_width: bool,
) -> list[float]:
    if kind == SpreadKind.LONG_CALL:
        return [0.0]
    return widths_to_try(spot, spread_width, scale=scale_width)


def _plan_spread_at_width(
    sym: str,
    spot: float,
    on: date,
    kind: SpreadKind,
    width: float,
    *,
    spread_width: float,
    target_dte: int,
    min_dte: int,
    max_dte: int,
) -> dict | None:
    leg_fn = _LEG_FN[kind]
    exp = pick_expiration(on, target_dte=target_dte, min_dte=min_dte, max_dte=max_dte)
    right = "C" if kind in (
        SpreadKind.CALL_DEBIT, SpreadKind.BEAR_CALL_CREDIT, SpreadKind.LONG_CALL,
    ) else "P"
    ladder = _strike_ladder(spot, width or spread_width)
    legs = leg_fn(ladder, spot, width)
    if not legs:
        return None
    long_k, short_k = legs
    long_sym = build_occ_symbol(sym, exp, right, long_k)
    if kind == SpreadKind.LONG_CALL:
        return {
            "kind": kind.value,
            "long_sym": long_sym,
            "short_sym": "",
            "long_k": long_k,
            "short_k": short_k,
            "exp": exp,
            "is_credit": False,
            "single_leg": True,
        }
    short_sym = build_occ_symbol(sym, exp, right, short_k)
    is_credit = kind in (SpreadKind.BULL_PUT_CREDIT, SpreadKind.BEAR_CALL_CREDIT)
    return {
        "kind": kind.value,
        "long_sym": long_sym,
        "short_sym": short_sym,
        "long_k": long_k,
        "short_k": short_k,
        "exp": exp,
        "is_credit": is_credit,
        "single_leg": False,
    }


def _plan_spread(
    sym: str,
    spot: float,
    on: date,
    kind: SpreadKind,
    *,
    spread_width: float,
    scale_width: bool,
    target_dte: int,
    min_dte: int,
    max_dte: int,
) -> dict | None:
    for width in _widths_for_kind(spot, kind, spread_width, scale_width=scale_width):
        plan = _plan_spread_at_width(
            sym, spot, on, kind, width,
            spread_width=spread_width,
            target_dte=target_dte,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        if plan:
            return plan
    return None


class OptionsSpreadPortfolio:
    def __init__(
        self,
        capital: float,
        max_positions: int,
        buy_conf: float,
        sell_conf: float,
        tickers: list[str],
        calendar: list[date],
        bar_cache: OptionSpreadBarCache,
        *,
        spread_width: float,
        target_dte: int,
        min_dte: int,
        max_dte: int,
        min_dte_exit: int,
        premium_stop_pct: float,
        min_hold_days: int,
        limit_slippage_pct: float,
        cost_per_side: float,
        profit_target_pct: float = 0.0,
        scale_width_by_price: bool = True,
        max_contracts_per_slot: int = 10,
        bullish_strategies: tuple[str, ...] = ("call_debit",),
        bearish_strategies: tuple[str, ...] = (),
        enable_bearish_opens: bool = False,
        long_call_min_confidence: float = 0.70,
        portfolio_mode: str = "bullish",
        circuit_breakers: bool = False,
        cb_day_half_pct: float = -0.02,
        cb_day_close_pct: float = -0.03,
        cb_week_half_pct: float | None = -0.05,
        cb_peak_halt_pct: float | None = -0.10,
    ):
        self.capital = float(capital)
        self.cash = float(capital)
        self.max_positions = int(max_positions)
        self.buy_conf = float(buy_conf)
        self.sell_conf = float(sell_conf)
        self.tickers = tickers
        self.calendar = calendar
        self.cache = bar_cache
        self.spread_width = float(spread_width)
        self.target_dte = int(target_dte)
        self.min_dte = int(min_dte)
        self.max_dte = int(max_dte)
        self.min_dte_exit = int(min_dte_exit)
        self.premium_stop_pct = float(premium_stop_pct)
        self.profit_target_pct = float(profit_target_pct)
        self.min_hold_days = int(min_hold_days)
        self.slip = 1.0 + float(limit_slippage_pct)
        self.credit_slip = 1.0 - float(limit_slippage_pct)
        self.cost = float(cost_per_side)
        self.scale_width_by_price = bool(scale_width_by_price)
        self.max_contracts_per_slot = max(1, int(max_contracts_per_slot))
        self.bullish_kinds = [
            _KIND_FROM_NAME[s] for s in bullish_strategies if s in _KIND_FROM_NAME
        ]
        self.bearish_kinds = [
            _KIND_FROM_NAME[s] for s in bearish_strategies if s in _KIND_FROM_NAME
        ]
        self.enable_bearish_opens = bool(enable_bearish_opens)
        self.long_call_min_confidence = float(long_call_min_confidence)
        self.portfolio_mode = str(portfolio_mode).lower()
        self.positions: dict[int, dict] = {}
        self.equity_curve: list[float] = []
        self.trade_pnls: list[float] = []
        self.hold_lengths: list[int] = []
        self.n_opens = 0
        self.n_skipped_no_bars = 0
        self.opens_by_kind: dict[str, int] = {}
        self.circuit_breakers = bool(circuit_breakers)
        self.cb_day_half_pct = float(cb_day_half_pct)
        self.cb_day_close_pct = float(cb_day_close_pct)
        self.cb_week_half_pct = (
            float(cb_week_half_pct) if cb_week_half_pct is not None else None
        )
        self.cb_peak_halt_pct = (
            float(cb_peak_halt_pct) if cb_peak_halt_pct is not None else None
        )
        self.peak_equity = float(capital)
        self._day_start_equity = float(capital)
        self._week_start_equity = float(capital)
        self._week_bars = 0
        self._last_on: date | None = None
        self._size_mult = 1.0
        self._halted = False
        self.cb_peak_halts = 0
        self.cb_day_closes = 0
        self.cb_half_size_days = 0

    def _equity_at(self, t: int) -> float:
        return self.cash + self._mtm_open(t)

    def _close_all(self, t: int) -> None:
        on = self.calendar[t]
        for idx in list(self.positions.keys()):
            pos = self.positions[idx]
            value = self._mark(pos, on)
            if value is None:
                value = pos.get("last_mark")
            if value is not None:
                self._close(idx, value, t)

    def _apply_circuit_breakers(self, t: int) -> None:
        if not self.circuit_breakers or self._halted:
            return
        on = self.calendar[t]
        eq = self._equity_at(t)
        self.peak_equity = max(self.peak_equity, eq)
        if (
            self.cb_peak_halt_pct is not None
            and self.peak_equity > 0
            and (eq / self.peak_equity - 1.0) <= self.cb_peak_halt_pct
        ):
            self._halted = True
            self.cb_peak_halts += 1
            self._close_all(t)
            return

        if self._last_on is not None and on != self._last_on:
            if self._day_start_equity > 0:
                day_ret = eq / self._day_start_equity - 1.0
                if day_ret <= self.cb_day_close_pct:
                    self._close_all(t)
                    self.cb_day_closes += 1
                    eq = self._equity_at(t)
                    self._size_mult = 1.0
                elif day_ret <= self.cb_day_half_pct:
                    self._size_mult = 0.5
                    self.cb_half_size_days += 1
                else:
                    self._size_mult = 1.0
            self._week_bars += 1
            if self._week_bars > 5:
                self._week_start_equity = eq
                self._week_bars = 1
            elif self.cb_week_half_pct is not None and self._week_start_equity > 0:
                week_ret = eq / self._week_start_equity - 1.0
                if week_ret <= self.cb_week_half_pct:
                    self._size_mult = min(self._size_mult, 0.5)
            self._day_start_equity = eq
        elif self._last_on is None:
            self._day_start_equity = eq
            self._week_start_equity = eq
            self._week_bars = 1
        self._last_on = on

    def _mark(self, pos: dict, on: date) -> float | None:
        return self.cache.position_mark(
            pos["long_sym"],
            pos.get("short_sym", ""),
            on,
            is_credit=bool(pos.get("is_credit")),
            single_leg=bool(pos.get("single_leg")),
        )

    def _pnl_pct(self, pos: dict, pos_value: float) -> float:
        prem = pos["premium"]
        if prem <= 0:
            return 0.0
        if pos.get("is_credit"):
            return (prem - pos_value) / prem
        return (pos_value - prem) / prem

    def _close(self, idx: int, value: float, step: int):
        pos = self.positions.pop(idx)
        contracts = pos.get("contracts", 1)
        prem = pos["premium"]
        if pos.get("is_credit"):
            close_cost = value * contracts * (1.0 + self.cost)
            self.cash -= close_cost
            self.cash += float(pos.get("collateral", 0.0))
            if prem > 0:
                self.trade_pnls.append((prem - close_cost) / prem)
        else:
            credit = value * contracts * (1.0 - self.cost)
            self.cash += credit
            if prem > 0:
                self.trade_pnls.append((credit - prem) / prem)
        self.hold_lengths.append(step - pos["entry_step"])

    def _mtm_open(self, t: int) -> float:
        on = self.calendar[t]
        total = 0.0
        for pos in self.positions.values():
            m = self._mark(pos, on)
            if m is None:
                m = pos.get("last_mark")
            if m is not None:
                pos["last_mark"] = m
                contracts = pos.get("contracts", 1)
                if pos.get("is_credit"):
                    total += float(pos.get("collateral", 0.0)) - m * contracts
                else:
                    total += m * contracts
        return total

    def _try_open(
        self,
        t: int,
        idx: int,
        conf_val: float,
        spot: float,
        on: date,
        per_slot: float,
        kinds: list[SpreadKind],
    ) -> bool:
        sym = self.tickers[idx]
        for kind in kinds:
            if kind == SpreadKind.LONG_CALL and conf_val < self.long_call_min_confidence:
                continue
            for width in _widths_for_kind(
                spot, kind, self.spread_width, scale_width=self.scale_width_by_price,
            ):
                plan = _plan_spread_at_width(
                    sym,
                    spot,
                    on,
                    kind,
                    width,
                    spread_width=self.spread_width,
                    target_dte=self.target_dte,
                    min_dte=self.min_dte,
                    max_dte=self.max_dte,
                )
                if not plan:
                    continue
                mark = self._mark(plan, on)
                if mark is None or mark <= 0:
                    if not plan.get("single_leg"):
                        self.cache.remember_no_spread(plan["long_sym"], plan["short_sym"], on)
                    self.n_skipped_no_bars += 1
                    self.cache.skipped_no_bars += 1
                    continue

                if plan.get("is_credit"):
                    credit_per = mark * self.credit_slip
                    if credit_per <= 0:
                        continue
                    strike_width = abs(plan["short_k"] - plan["long_k"])
                    credit_per_share = credit_per / 100.0
                    max_risk_per = max(50.0, (strike_width - credit_per_share) * 100.0)
                    n = int(per_slot // max_risk_per)
                    n = max(1, min(n, self.max_contracts_per_slot))
                    prem = credit_per * n
                    collateral = max_risk_per * n
                    credit_net = prem * (1.0 - self.cost)
                    net_reserve = collateral - credit_net
                    if net_reserve > self.cash:
                        n = int(self.cash // max(max_risk_per - credit_per * (1.0 - self.cost), 1.0))
                        if n < 1:
                            continue
                        n = min(n, self.max_contracts_per_slot)
                        prem = credit_per * n
                        collateral = max_risk_per * n
                        credit_net = prem * (1.0 - self.cost)
                        net_reserve = collateral - credit_net
                        if net_reserve > self.cash:
                            continue
                    self.cash += credit_net - collateral
                else:
                    contract_cost = mark * self.slip
                    if contract_cost <= 0:
                        continue
                    n = int(per_slot // contract_cost)
                    n = max(1, min(n, self.max_contracts_per_slot))
                    prem = contract_cost * n
                    if self.cash < prem * (1.0 + self.cost):
                        n = int(self.cash // (contract_cost * (1.0 + self.cost)))
                        if n < 1:
                            continue
                        prem = contract_cost * n
                    self.cash -= prem * (1.0 + self.cost)
                    collateral = 0.0

                self.positions[idx] = {
                    **plan,
                    "premium": prem,
                    "contracts": n,
                    "collateral": collateral,
                    "entry_step": t,
                    "exp_date": plan["exp"],
                    "last_mark": mark,
                }
                self.n_opens += 1
                k = plan["kind"]
                self.opens_by_kind[k] = self.opens_by_kind.get(k, 0) + 1
                return True
        return False

    def step(
        self,
        t: int,
        actions: np.ndarray,
        conf: np.ndarray,
        px: np.ndarray,
        active: np.ndarray,
        *,
        block_new_buys: bool = False,
    ):
        on = self.calendar[t]
        self._apply_circuit_breakers(t)
        if self._halted:
            block_new_buys = True

        for idx in list(self.positions.keys()):
            pos = self.positions[idx]
            value = self._mark(pos, on)
            if value is None:
                value = pos.get("last_mark")
            if value is None:
                continue
            pos["last_mark"] = value

            contracts = pos.get("contracts", 1)
            pos_value = value * contracts
            pnl_pct = self._pnl_pct(pos, pos_value)
            bars_held = t - pos["entry_step"]
            dte_left = (pos["exp_date"] - on).days

            agent_sell = actions[idx] == 2
            sell_ok = self.sell_conf <= 0 or conf[idx] >= self.sell_conf
            exit = False
            if pnl_pct <= -self.premium_stop_pct:
                exit = True
            elif self.profit_target_pct > 0 and pnl_pct >= self.profit_target_pct:
                exit = True
            elif dte_left <= self.min_dte_exit:
                exit = True
            elif agent_sell and sell_ok:
                if not (self.min_hold_days > 0 and bars_held < self.min_hold_days and pnl_pct > 0):
                    exit = True

            if exit:
                self._close(idx, value, t)

        free = self.max_positions - len(self.positions)
        if free > 0 and on >= OPTIONS_DATA_MIN_DATE and not self._halted:
            equity = self.cash + self._mtm_open(t)
            per_slot = (equity / self.max_positions) * self._size_mult

            if self.portfolio_mode == "unified":
                open_kinds = self.bullish_kinds + self.bearish_kinds
            elif self.portfolio_mode == "bearish":
                open_kinds = self.bearish_kinds
            else:
                open_kinds = self.bullish_kinds
            if not block_new_buys and open_kinds:
                buy_cands = np.where((actions == 1) & (conf >= self.buy_conf) & active & (px > 0))[0]
                buy_cands = [c for c in buy_cands if c not in self.positions]
                buy_cands.sort(key=lambda c: conf[c], reverse=True)
                occ_syms: list[str] = []
                fetch_end_max = on
                for idx in buy_cands[: free * 3]:
                    sym = self.tickers[idx]
                    spot = float(px[idx])
                    for kind in open_kinds:
                        if (
                            self.portfolio_mode != "bearish"
                            and kind == SpreadKind.LONG_CALL
                            and conf[idx] < self.long_call_min_confidence
                        ):
                            continue
                        for width in _widths_for_kind(
                            spot, kind, self.spread_width,
                            scale_width=self.scale_width_by_price,
                        ):
                            plan = _plan_spread_at_width(
                                sym, spot, on, kind, width,
                                spread_width=self.spread_width,
                                target_dte=self.target_dte,
                                min_dte=self.min_dte,
                                max_dte=self.max_dte,
                            )
                            if not plan:
                                continue
                            occ_syms.append(plan["long_sym"])
                            if plan.get("short_sym"):
                                occ_syms.append(plan["short_sym"])
                            fetch_end_max = max(fetch_end_max, plan["exp"] + timedelta(days=7))
                if occ_syms:
                    self.cache.ensure_symbols(
                        occ_syms, on - timedelta(days=3), fetch_end_max,
                    )
                opened = 0
                for idx in buy_cands:
                    if opened >= free:
                        break
                    if self._try_open(
                        t, idx, float(conf[idx]), float(px[idx]), on, per_slot, open_kinds,
                    ):
                        opened += 1
                        free -= 1

        self.equity_curve.append(self.cash + self._mtm_open(t))

    def metrics(self) -> dict | None:
        eq = np.array(self.equity_curve)
        if len(eq) < 2:
            return None
        total_ret = eq[-1] / self.capital - 1.0
        years = len(eq) / 252.0
        if years > 0 and eq[-1] > 0:
            cagr = (eq[-1] / self.capital) ** (1 / years) - 1.0
        else:
            cagr = 0.0
        rets = np.diff(eq) / eq[:-1]
        sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252) if rets.std() > 0 else 0.0
        run_max = np.maximum.accumulate(eq)
        max_dd = ((eq - run_max) / run_max).min() * 100
        pnls = np.array(self.trade_pnls)
        wr = (pnls > 0).mean() * 100 if len(pnls) else 0.0
        return {
            "final": eq[-1],
            "total_ret": total_ret * 100,
            "cagr": cagr * 100,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "trades": len(pnls),
            "wr": wr,
            "avg_trade": pnls.mean() * 100 if len(pnls) else 0.0,
            "avg_hold": np.mean(self.hold_lengths) if self.hold_lengths else 0.0,
            "opens": self.n_opens,
            "opens_by_kind": dict(self.opens_by_kind),
            "skipped_no_bars": self.n_skipped_no_bars,
            "api_calls": self.cache.api_calls,
            "fetch_errors": self.cache.fetch_errors,
            "cb_peak_halts": self.cb_peak_halts,
            "cb_day_closes": self.cb_day_closes,
            "cb_half_size_days": self.cb_half_size_days,
            "cb_halted": self._halted,
        }


def run_backtest(
    model_path: str,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    watchlist_path: str | None = DEFAULT_WATCHLIST,
    capital: float = 10_000.0,
    max_positions: int | None = None,
    buy_conf: float | None = None,
    sell_conf: float | None = None,
    spy_fear_block_pct: float | None = None,
    benchmark: str = "SPY",
    option_feed: str = "indicative",
    use_disk_cache: bool = False,
    confidence_temperature: float | None = None,
    call_debit_only: bool = False,
    bearish_only: bool = False,
    unified_only: bool = False,
    circuit_breakers: bool = False,
    cb_day_only: bool = False,
    quiet: bool = False,
):
    cfg = OptionsTraderConfig
    max_positions = max_positions or cfg.MAX_POSITIONS
    if bearish_only:
        buy_conf = buy_conf if buy_conf is not None else float(
            getattr(cfg, "BEARISH_CONFIDENCE_THRESHOLD", 0.65)
        )
    else:
        buy_conf = buy_conf if buy_conf is not None else cfg.CONFIDENCE_THRESHOLD
    sell_conf = sell_conf if sell_conf is not None else cfg.SELL_CONFIDENCE_THRESHOLD
    if spy_fear_block_pct is None and cfg.ENABLE_SPY_FEAR_FILTER:
        spy_fear_block_pct = cfg.SPY_FEAR_BLOCK_PCT
    if confidence_temperature is None:
        confidence_temperature = float(getattr(cfg, "CONFIDENCE_TEMPERATURE", 0.01))

    if test_start_date is None:
        test_start_date = OPTIONS_DATA_MIN_DATE.isoformat()
    elif date.fromisoformat(test_start_date[:10]) < OPTIONS_DATA_MIN_DATE:
        print(
            f"Warning: Alpaca option bars start {OPTIONS_DATA_MIN_DATE}; "
            f"clamping test-start to that date."
        )
        test_start_date = OPTIONS_DATA_MIN_DATE.isoformat()

    print("=" * 72)
    if unified_only:
        title = "UNIFIED OPTIONS BACKTEST"
    elif bearish_only:
        title = "BEARISH OPTIONS BACKTEST"
    else:
        title = "OPTIONS PORTFOLIO BACKTEST"
    print(f"{title} (Alpaca historical option bars only)")
    print(f"  Model: {model_path}")
    print(f"  Watchlist: {watchlist_path or 'all with CSV'}")
    print(f"  Period: {test_start_date} → {test_end_date or 'latest'}")
    print(f"  Option feed: {option_feed}")
    if unified_only:
        bullish = tuple(getattr(cfg, "BULLISH_STRATEGIES", ("call_debit", "bull_put_credit", "long_call")))
        bearish = tuple(getattr(cfg, "BEARISH_STRATEGIES", ("put_debit", "bear_call_credit")))
        portfolio_mode = "unified"
        mode_label = (
            "unified (call_debit -> bull_put -> long_call -> put_debit -> bear_call_credit)"
        )
    elif bearish_only:
        bullish = ()
        bearish = tuple(getattr(cfg, "BEARISH_STRATEGIES", ("put_debit", "bear_call_credit")))
        portfolio_mode = "bearish"
        mode_label = "bearish model (BUY opens put_debit -> bear_call_credit)"
    elif call_debit_only:
        bullish = ("call_debit",)
        bearish = ()
        portfolio_mode = "bullish"
        mode_label = "call_debit only (baseline)"
    else:
        bullish = tuple(getattr(cfg, "BULLISH_STRATEGIES", ("call_debit",)))
        bearish = ()
        portfolio_mode = "bullish"
        mode_label = "multi-strategy bullish (call_debit -> bull_put -> long_call)"
    print(f"  Mode: {mode_label}")
    if unified_only:
        print(f"  Bullish: {bullish}")
        print(f"  Bearish: {bearish}")
    elif bullish:
        print(f"  Bullish: {bullish}")
    if bearish_only:
        print(f"  Bearish: {bearish}")
    print(f"  Buy conf > {buy_conf}, Sell conf > {sell_conf}, slots={max_positions}")
    print(f"  Confidence temperature: {confidence_temperature}")
    print(f"  Spread width=${cfg.SPREAD_WIDTH} (scaled={getattr(cfg, 'SCALE_WIDTH_BY_PRICE', True)}), "
          f"target DTE ~{cfg.TARGET_DTE}, premium stop={cfg.PREMIUM_STOP_PCT:.0%}, "
          f"profit target={getattr(cfg, 'PROFIT_TARGET_PCT', 0.0):.0%}, "
          f"max contracts/slot={getattr(cfg, 'MAX_CONTRACTS_PER_SLOT', 10)}")
    cb_peak = None if cb_day_only else -0.10
    cb_week = None if cb_day_only else -0.05
    if circuit_breakers:
        if cb_day_only:
            print("  Circuit breakers: DAY ONLY (-2% half size, -3% close all)")
        else:
            print(
                "  Circuit breakers: FULL (day -2%/-3%, week -5%, peak -10% halt)"
            )
    if spy_fear_block_pct is not None:
        print(f"  SPY fear filter: block new BUY when SPY day chg < {spy_fear_block_pct}%")
    print("=" * 72)

    if use_disk_cache:
        bar_cache = make_option_bar_cache_from_disk()
        print("  Option marks: disk cache (data/historical_options/*.csv)")
    else:
        try:
            bar_cache = make_option_bar_cache_from_env(feed=option_feed)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return None

    sig = _load_and_infer(
        model_path, test_start_date, test_end_date, watchlist_path,
        confidence_temperature=confidence_temperature,
    )
    if sig is None:
        print("No data.")
        return None

    T, N = sig["T"], sig["N"]
    calendar = load_swing_calendar(T, test_start_date=test_start_date, test_end_date=test_end_date)
    if len(calendar) != T:
        print(f"Calendar length {len(calendar)} != bars {T}")
        return None

    port = OptionsSpreadPortfolio(
        capital,
        max_positions,
        buy_conf,
        sell_conf,
        sig["tickers"],
        calendar,
        bar_cache,
        spread_width=cfg.SPREAD_WIDTH,
        target_dte=cfg.TARGET_DTE,
        min_dte=cfg.MIN_DTE,
        max_dte=cfg.MAX_DTE,
        min_dte_exit=cfg.MIN_DTE_EXIT,
        premium_stop_pct=cfg.PREMIUM_STOP_PCT,
        min_hold_days=cfg.MIN_HOLD_DAYS,
        limit_slippage_pct=cfg.LIMIT_SLIPPAGE_PCT,
        cost_per_side=0.0005,
        profit_target_pct=float(getattr(cfg, "PROFIT_TARGET_PCT", 0.0)),
        scale_width_by_price=bool(getattr(cfg, "SCALE_WIDTH_BY_PRICE", True)),
        max_contracts_per_slot=int(getattr(cfg, "MAX_CONTRACTS_PER_SLOT", 10)),
        bullish_strategies=bullish,
        bearish_strategies=bearish if (bearish_only or unified_only) else (),
        enable_bearish_opens=False,
        long_call_min_confidence=float(getattr(cfg, "LONG_CALL_MIN_CONFIDENCE", 0.70)),
        portfolio_mode=portfolio_mode,
        circuit_breakers=circuit_breakers,
        cb_peak_halt_pct=cb_peak if circuit_breakers else None,
        cb_week_half_pct=cb_week if circuit_breakers else None,
    )

    bench_ret = None
    if benchmark in sig["tickers"]:
        bi = sig["tickers"].index(benchmark)
        bs = sig["prices"][bi, WINDOW:]
        bs = bs[bs > 0]
        if len(bs) > 1:
            bench_ret = (bs[-1] / bs[0] - 1.0) * 100

    n_days = T - WINDOW
    if not quiet:
        if use_disk_cache:
            print(f"Simulating {n_days} trading days over {N} symbols (disk cache)…")
        else:
            print(
                f"Simulating {n_days} trading days — fetching Alpaca option bars on demand "
                f"(this can take several minutes)…"
            )
    for t in tqdm(range(WINDOW, T), desc="Options backtest", unit="day"):
        block = False
        if spy_fear_block_pct is not None:
            block = _spy_day_change_pct(sig["prices"], sig["tickers"], t) < float(spy_fear_block_pct)
        port.step(
            t,
            sig["actions"][t],
            sig["conf"][t],
            sig["prices"][:, t],
            sig["active"][:, t],
            block_new_buys=block,
        )

    m = port.metrics()
    if not m:
        print("Insufficient bars for metrics.")
        return None

    alpha = m["total_ret"] - bench_ret if bench_ret is not None else float("nan")
    print(f"\nResults ({T - WINDOW} trading days, {N} symbols)")
    print(f"  Final equity:     ${m['final']:,.2f}")
    print(f"  Total return:     {m['total_ret']:+.2f}%")
    print(f"  CAGR (approx):    {m['cagr']:+.2f}%")
    print(f"  Max drawdown:     {m['max_dd']:.2f}%")
    print(f"  Sharpe (approx):  {m['sharpe']:.2f}")
    print(f"  Spread opens:     {m['opens']}")
    if m.get("opens_by_kind"):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(m["opens_by_kind"].items()))
        print(f"  Opens by kind:    {parts}")
    print(f"  Skipped (no bars): {m['skipped_no_bars']}")
    print(f"  Alpaca API calls: {m['api_calls']}")
    if m.get("fetch_errors"):
        print(f"  API fetch errors:  {m['fetch_errors']}")
    print(f"  Closed trades:    {m['trades']}")
    print(f"  Win rate:         {m['wr']:.1f}%")
    print(f"  Avg trade (on prem): {m['avg_trade']:+.2f}%")
    print(f"  Avg hold (bars):  {m['avg_hold']:.1f}")
    if bench_ret is not None:
        print(f"  {benchmark} B&H:      {bench_ret:+.2f}%")
        print(f"  Alpha vs {benchmark}:   {alpha:+.2f}%")
    if circuit_breakers:
        print(f"  CB peak halts:    {m.get('cb_peak_halts', 0)}")
        print(f"  CB day closes:    {m.get('cb_day_closes', 0)}")
        print(f"  CB half-size days:{m.get('cb_half_size_days', 0)}")
        if m.get("cb_halted"):
            print("  CB status:        HALTED (peak drawdown limit hit)")
    print("\nMarks and premiums use Alpaca option daily closes only (no synthetic fallback).")
    return m


def main():
    ap = argparse.ArgumentParser(description="Options spread backtest (Alpaca bars only)")
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    ap.add_argument("--test-start-date", default=None, help=f"Default {OPTIONS_DATA_MIN_DATE}")
    ap.add_argument("--test-end-date", default=None)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--confidence-threshold", type=float, default=None)
    ap.add_argument("--sell-confidence-threshold", type=float, default=None)
    ap.add_argument(
        "--confidence-temperature", type=float, default=None,
        help="Softmax temp for confidence (default = OptionsTraderConfig).",
    )
    ap.add_argument("--spy-fear-pct", type=float, default=None)
    ap.add_argument("--no-spy-filter", action="store_true")
    ap.add_argument(
        "--option-feed",
        default="indicative",
        choices=("indicative", "opra"),
        help="indicative=free delayed; opra=subscription",
    )
    ap.add_argument(
        "--disk-cache",
        action="store_true",
        help="Use downloaded OCC CSVs (data/historical_options); no Alpaca API",
    )
    ap.add_argument(
        "--call-debit-only",
        action="store_true",
        help="Baseline: only call debit spreads (no multi-strategy)",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Run call-debit baseline then multi-strategy and print side-by-side",
    )
    ap.add_argument(
        "--bearish",
        action="store_true",
        help="Bearish model backtest: BUY opens put_debit/bear_call_credit, SELL closes",
    )
    ap.add_argument(
        "--unified",
        action="store_true",
        help="Unified model: BUY opens all 5 strategies in priority order",
    )
    ap.add_argument(
        "--circuit-breakers",
        action="store_true",
        help="Portfolio risk layer: day/week loss limits + peak drawdown halt",
    )
    ap.add_argument(
        "--compare-risk",
        action="store_true",
        help="Run baseline then --circuit-breakers and print side-by-side",
    )
    ap.add_argument(
        "--cb-day-only",
        action="store_true",
        help="With --circuit-breakers: day loss rules only (no week/peak halt)",
    )
    ap.add_argument(
        "--compare-risk-grid",
        action="store_true",
        help="Baseline vs day-only CB vs SPY filter vs both",
    )
    args = ap.parse_args()

    spy = None if args.no_spy_filter else args.spy_fear_pct
    if args.compare:
        print("\n>>> BASELINE: call debit only\n")
        m_base = run_backtest(
            args.model_path,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            watchlist_path=args.watchlist,
            capital=args.capital,
            max_positions=args.max_positions,
            buy_conf=args.confidence_threshold,
            sell_conf=args.sell_confidence_threshold,
            spy_fear_block_pct=spy,
            option_feed=args.option_feed,
            use_disk_cache=args.disk_cache,
            confidence_temperature=args.confidence_temperature,
            call_debit_only=True,
        )
        print("\n>>> MULTI-STRATEGY\n")
        m_multi = run_backtest(
            args.model_path,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            watchlist_path=args.watchlist,
            capital=args.capital,
            max_positions=args.max_positions,
            buy_conf=args.confidence_threshold,
            sell_conf=args.sell_confidence_threshold,
            spy_fear_block_pct=spy,
            option_feed=args.option_feed,
            use_disk_cache=args.disk_cache,
            confidence_temperature=args.confidence_temperature,
            call_debit_only=False,
        )
        if m_base and m_multi:
            print("\n" + "=" * 72)
            print("COMPARISON")
            print(f"  Call-debit total return:  {m_base['total_ret']:+.2f}%")
            print(f"  Multi-strategy return:    {m_multi['total_ret']:+.2f}%")
            print(f"  Delta:                    {m_multi['total_ret'] - m_base['total_ret']:+.2f}%")
            print(f"  Call-debit opens:         {m_base['opens']}")
            print(f"  Multi opens:              {m_multi['opens']}")
            print("=" * 72)
        return

    if args.compare_risk:
        common = dict(
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            watchlist_path=args.watchlist,
            capital=args.capital,
            max_positions=args.max_positions,
            buy_conf=args.confidence_threshold,
            sell_conf=args.sell_confidence_threshold,
            spy_fear_block_pct=spy,
            option_feed=args.option_feed,
            use_disk_cache=args.disk_cache,
            confidence_temperature=args.confidence_temperature,
            call_debit_only=args.call_debit_only,
            bearish_only=args.bearish,
            unified_only=args.unified,
        )
        print("\n>>> BASELINE (no circuit breakers)\n")
        m_base = run_backtest(args.model_path, circuit_breakers=False, **common)
        print("\n>>> WITH CIRCUIT BREAKERS\n")
        m_cb = run_backtest(args.model_path, circuit_breakers=True, **common)
        if m_base and m_cb:
            print("\n" + "=" * 72)
            print("RISK COMPARISON")
            print(f"  {'':18} {'Baseline':>12} {'CB ON':>12} {'Delta':>10}")
            print(f"  {'Return %':18} {m_base['total_ret']:+12.2f} {m_cb['total_ret']:+12.2f} "
                  f"{m_cb['total_ret'] - m_base['total_ret']:+10.2f}")
            print(f"  {'Max DD %':18} {m_base['max_dd']:12.2f} {m_cb['max_dd']:12.2f} "
                  f"{m_cb['max_dd'] - m_base['max_dd']:+10.2f}")
            print(f"  {'Sharpe':18} {m_base['sharpe']:12.2f} {m_cb['sharpe']:12.2f} "
                  f"{m_cb['sharpe'] - m_base['sharpe']:+10.2f}")
            print(f"  {'Opens':18} {m_base['opens']:12d} {m_cb['opens']:12d} "
                  f"{m_cb['opens'] - m_base['opens']:+10d}")
            print("=" * 72)
        return

    if args.compare_risk_grid:
        spy_block = float(
            args.spy_fear_pct
            if args.spy_fear_pct is not None
            else OptionsTraderConfig.SPY_FEAR_BLOCK_PCT
        )
        common = dict(
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            watchlist_path=args.watchlist,
            capital=args.capital,
            max_positions=args.max_positions,
            buy_conf=args.confidence_threshold,
            sell_conf=args.sell_confidence_threshold,
            option_feed=args.option_feed,
            use_disk_cache=args.disk_cache,
            confidence_temperature=args.confidence_temperature,
            call_debit_only=args.call_debit_only,
            bearish_only=args.bearish,
            unified_only=args.unified,
            quiet=True,
        )
        variants = [
            ("baseline", dict(spy_fear_block_pct=None, circuit_breakers=False)),
            ("day_cb", dict(spy_fear_block_pct=None, circuit_breakers=True, cb_day_only=True)),
            ("spy", dict(spy_fear_block_pct=spy_block, circuit_breakers=False)),
            ("day_cb+spy", dict(spy_fear_block_pct=spy_block, circuit_breakers=True, cb_day_only=True)),
        ]
        rows: list[tuple[str, dict]] = []
        for label, kw in variants:
            print(f"\n>>> {label}\n")
            m = run_backtest(args.model_path, **common, **kw)
            if m:
                rows.append((label, m))
        if rows:
            print("\n" + "=" * 78)
            print("RISK GRID (200-symbol unified backtest)")
            print(f"  {'Variant':<14} {'Return%':>9} {'MaxDD%':>9} {'Sharpe':>8} {'Opens':>7}")
            print("-" * 78)
            for label, m in rows:
                print(
                    f"  {label:<14} {m['total_ret']:+9.2f} {m['max_dd']:9.2f} "
                    f"{m['sharpe']:8.2f} {m['opens']:7d}"
                )
            base = rows[0][1]
            print("-" * 78)
            for label, m in rows[1:]:
                print(
                    f"  {label} vs baseline: return {m['total_ret'] - base['total_ret']:+.2f}%, "
                    f"DD {m['max_dd'] - base['max_dd']:+.2f}%"
                )
            print("=" * 78)
        return

    run_backtest(
        args.model_path,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        watchlist_path=args.watchlist,
        capital=args.capital,
        max_positions=args.max_positions,
        buy_conf=args.confidence_threshold,
        sell_conf=args.sell_confidence_threshold,
        spy_fear_block_pct=spy,
        option_feed=args.option_feed,
        use_disk_cache=args.disk_cache,
        confidence_temperature=args.confidence_temperature,
        call_debit_only=args.call_debit_only,
        bearish_only=args.bearish,
        unified_only=args.unified,
        circuit_breakers=args.circuit_breakers,
        cb_day_only=args.cb_day_only,
    )


if __name__ == "__main__":
    main()
