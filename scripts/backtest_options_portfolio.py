"""
Portfolio backtest: swing Gen7 signals → call debit spreads marked with Alpaca option bars.

On-demand fetches historical daily option closes (indicative feed, ~Feb 2024+).
Trades are skipped when either leg has no bar on entry; marks/exits use real bars only.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/backtest_options_portfolio.py
  .\\.venv\\Scripts\\python.exe scripts/backtest_options_portfolio.py \\
      --watchlist config/watchlists/day_trade_list.txt \\
      --test-start-date 2024-03-01
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OptionsTraderConfig, TrainingConfig  # noqa: E402

TrainingConfig.WINDOW_SIZE = 60

from scripts.backtest_swing_portfolio import (  # noqa: E402
    WINDOW,
    _load_and_infer,
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

DEFAULT_MODEL = "models/swing_gen7_refined_ep380"
DEFAULT_WATCHLIST = os.path.join("config", "watchlists", "swing_liquid.txt")


def _round_strike(spot: float, width: float = 5.0) -> tuple[float, float]:
    step = width if spot >= 200 else (2.5 if spot < 100 else width)
    long_k = round(spot / step) * step
    if long_k <= 0:
        long_k = max(step, spot)
    return float(long_k), float(long_k + width)


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
        self.min_hold_days = int(min_hold_days)
        self.slip = 1.0 + float(limit_slippage_pct)
        self.cost = float(cost_per_side)
        self.positions: dict[int, dict] = {}
        self.equity_curve: list[float] = []
        self.trade_pnls: list[float] = []
        self.hold_lengths: list[int] = []
        self.n_opens = 0
        self.n_skipped_no_bars = 0

    def _mark(self, pos: dict, on: date) -> float | None:
        return self.cache.spread_mark(pos["long_sym"], pos["short_sym"], on)

    def _close(self, idx: int, value: float, step: int):
        pos = self.positions.pop(idx)
        credit = value * (1.0 - self.cost)
        self.cash += credit
        prem = pos["premium"]
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
                total += m
        return total

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

        for idx in list(self.positions.keys()):
            pos = self.positions[idx]
            value = self._mark(pos, on)
            if value is None:
                value = pos.get("last_mark")
            if value is None:
                continue
            pos["last_mark"] = value

            prem = pos["premium"]
            pnl_pct = (value - prem) / prem if prem > 0 else 0.0
            bars_held = t - pos["entry_step"]
            dte_left = (pos["exp_date"] - on).days

            agent_sell = actions[idx] == 2
            sell_ok = self.sell_conf <= 0 or conf[idx] >= self.sell_conf
            exit = False
            if pnl_pct <= -self.premium_stop_pct:
                exit = True
            elif dte_left <= self.min_dte_exit:
                exit = True
            elif agent_sell and sell_ok:
                if not (self.min_hold_days > 0 and bars_held < self.min_hold_days and pnl_pct > 0):
                    exit = True

            if exit:
                self._close(idx, value, t)

        free = self.max_positions - len(self.positions)
        if free > 0 and not block_new_buys and on >= OPTIONS_DATA_MIN_DATE:
            cands = np.where((actions == 1) & (conf >= self.buy_conf) & active & (px > 0))[0]
            cands = [c for c in cands if c not in self.positions]
            cands.sort(key=lambda c: conf[c], reverse=True)
            equity = self.cash + self._mtm_open(t)
            per_slot = equity / self.max_positions

            plans = []
            fetch_end_max = on
            for idx in cands:
                sym = self.tickers[idx]
                spot = float(px[idx])
                long_k, short_k = _round_strike(spot, self.spread_width)
                exp = pick_expiration(
                    on,
                    target_dte=self.target_dte,
                    min_dte=self.min_dte,
                    max_dte=self.max_dte,
                )
                long_sym = build_occ_symbol(sym, exp, "C", long_k)
                short_sym = build_occ_symbol(sym, exp, "C", short_k)
                fetch_end = exp + timedelta(days=7)
                fetch_end_max = max(fetch_end_max, fetch_end)
                plans.append((idx, long_sym, short_sym, long_k, short_k, exp))

            if plans:
                occ_syms = []
                for _idx, ls, ss, *_ in plans:
                    occ_syms.extend([ls, ss])
                self.cache.ensure_symbols(
                    occ_syms, on - timedelta(days=3), fetch_end_max,
                )

            opened = 0
            for idx, long_sym, short_sym, long_k, short_k, exp in plans:
                if opened >= free:
                    break
                mark = self.cache.spread_mark(long_sym, short_sym, on)
                if mark is None or mark <= 0:
                    self.cache.remember_no_spread(long_sym, short_sym, on)
                    self.n_skipped_no_bars += 1
                    self.cache.skipped_no_bars += 1
                    continue

                prem = min(per_slot, mark * self.slip)
                if self.cash < prem * (1.0 + self.cost):
                    continue

                self.cash -= prem * (1.0 + self.cost)
                self.positions[idx] = {
                    "long_sym": long_sym,
                    "short_sym": short_sym,
                    "long_k": long_k,
                    "short_k": short_k,
                    "premium": prem,
                    "entry_step": t,
                    "exp_date": exp,
                    "last_mark": mark,
                }
                self.n_opens += 1
                opened += 1

        self.equity_curve.append(self.cash + self._mtm_open(t))

    def metrics(self) -> dict | None:
        eq = np.array(self.equity_curve)
        if len(eq) < 2:
            return None
        total_ret = eq[-1] / self.capital - 1.0
        years = len(eq) / 252.0
        cagr = (eq[-1] / self.capital) ** (1 / years) - 1.0 if years > 0 else 0.0
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
            "skipped_no_bars": self.n_skipped_no_bars,
            "api_calls": self.cache.api_calls,
            "fetch_errors": self.cache.fetch_errors,
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
):
    cfg = OptionsTraderConfig
    max_positions = max_positions or cfg.MAX_POSITIONS
    buy_conf = buy_conf if buy_conf is not None else cfg.CONFIDENCE_THRESHOLD
    sell_conf = sell_conf if sell_conf is not None else cfg.SELL_CONFIDENCE_THRESHOLD
    if spy_fear_block_pct is None and cfg.ENABLE_SPY_FEAR_FILTER:
        spy_fear_block_pct = cfg.SPY_FEAR_BLOCK_PCT

    if test_start_date is None:
        test_start_date = OPTIONS_DATA_MIN_DATE.isoformat()
    elif date.fromisoformat(test_start_date[:10]) < OPTIONS_DATA_MIN_DATE:
        print(
            f"Warning: Alpaca option bars start {OPTIONS_DATA_MIN_DATE}; "
            f"clamping test-start to that date."
        )
        test_start_date = OPTIONS_DATA_MIN_DATE.isoformat()

    print("=" * 72)
    print("OPTIONS PORTFOLIO BACKTEST (Alpaca historical option bars only)")
    print(f"  Model: {model_path}")
    print(f"  Watchlist: {watchlist_path or 'all with CSV'}")
    print(f"  Period: {test_start_date} → {test_end_date or 'latest'}")
    print(f"  Option feed: {option_feed}")
    print(f"  Buy conf > {buy_conf}, Sell conf > {sell_conf}, slots={max_positions}")
    print(f"  Spread width=${cfg.SPREAD_WIDTH}, target DTE ~{cfg.TARGET_DTE}, "
          f"premium stop={cfg.PREMIUM_STOP_PCT:.0%}")
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

    sig = _load_and_infer(model_path, test_start_date, test_end_date, watchlist_path)
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
    )

    bench_ret = None
    if benchmark in sig["tickers"]:
        bi = sig["tickers"].index(benchmark)
        bs = sig["prices"][bi, WINDOW:]
        bs = bs[bs > 0]
        if len(bs) > 1:
            bench_ret = (bs[-1] / bs[0] - 1.0) * 100

    n_days = T - WINDOW
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
    args = ap.parse_args()

    spy = None if args.no_spy_filter else args.spy_fear_pct
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
    )


if __name__ == "__main__":
    main()
