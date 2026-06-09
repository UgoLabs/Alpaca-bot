"""
Sweep options_unified_gen380 checkpoints (unified 5-strategy portfolio backtest).

BUY opens: call_debit -> bull_put -> long_call -> put_debit -> bear_call_credit

Usage:
  .\\.venv\\Scripts\\python.exe scripts/sweep_options_unified_from_swing.py --disk-cache --no-spy-filter
  .\\.venv\\Scripts\\python.exe scripts/sweep_options_unified_from_swing.py --only 100,138,150,best --disk-cache
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OptionsTraderConfig, TrainingConfig  # noqa: E402

TrainingConfig.WINDOW_SIZE = 60

from scripts.backtest_options_portfolio import OptionsSpreadPortfolio, WINDOW, _spy_day_change_pct  # noqa: E402
from scripts.sweep_options_from_swing import (  # noqa: E402
    DEFAULT_WATCHLIST,
    _discover_checkpoints,
    _infer,
    _load_base_signals,
)
from src.data.options_historical import (  # noqa: E402
    load_swing_calendar,
    make_option_bar_cache_from_disk,
    make_option_bar_cache_from_env,
)

MODEL_PREFIX = "options_unified_gen380"


def _simulate_unified(
    sig: dict,
    bar_cache,
    *,
    capital: float,
    buy_conf: float,
    sell_conf: float,
    spy_fear_block_pct: float | None,
    calendar: list,
    max_positions: int | None = None,
) -> dict | None:
    cfg = OptionsTraderConfig
    bullish = tuple(getattr(cfg, "BULLISH_STRATEGIES", ("call_debit", "bull_put_credit", "long_call")))
    bearish = tuple(getattr(cfg, "BEARISH_STRATEGIES", ("put_debit", "bear_call_credit")))
    port = OptionsSpreadPortfolio(
        capital,
        max_positions if max_positions is not None else cfg.MAX_POSITIONS,
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
        bearish_strategies=bearish,
        portfolio_mode="unified",
        long_call_min_confidence=float(getattr(cfg, "LONG_CALL_MIN_CONFIDENCE", 0.70)),
    )
    T = sig["T"]
    for t in range(WINDOW, T):
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
    return port.metrics()


def _strip_model_suffix(path: str) -> str:
    for suf in ("_balanced.pth", "_aggressive.pth", "_conservative.pth"):
        if path.endswith(suf):
            return path[: -len(suf)]
    return path


def _run_threshold_sweep(args) -> None:
    model_path = _strip_model_suffix(args.model or f"models/{MODEL_PREFIX}_ep100")
    buy_grid = [float(x) for x in args.buy_grid.split(",") if x.strip()]
    sell_grid = [float(x) for x in args.sell_grid.split(",") if x.strip()]
    spy_fear = None if args.no_spy_filter else OptionsTraderConfig.SPY_FEAR_BLOCK_PCT
    if args.disk_cache:
        bar_cache = make_option_bar_cache_from_disk()
        cache_note = "disk (historical_options/*.csv)"
    else:
        bar_cache = make_option_bar_cache_from_env(feed="indicative")
        cache_note = "Alpaca API on demand"

    print("=" * 88)
    print("UNIFIED THRESHOLD SWEEP (one model, buy x sell grid)")
    print(f"  Model: {model_path}")
    print(f"  Watchlist: {args.watchlist}")
    print(f"  Marks: {cache_note}")
    print(f"  Period: {args.test_start_date or '2024-02-01'} -> {args.test_end_date or 'latest'}")
    print(f"  Buy grid:  {buy_grid}")
    print(f"  Sell grid: {sell_grid}")
    print("=" * 88)

    base = _load_base_signals(
        args.test_start_date or "2024-02-01",
        args.test_end_date,
        args.watchlist,
        force_cpu=args.cpu,
    )
    if base is None:
        print("No swing data.")
        return

    T, N = base["T"], base["N"]
    calendar = load_swing_calendar(
        T, test_start_date=args.test_start_date, test_end_date=args.test_end_date,
    )
    bench_ret = None
    if "SPY" in base["tickers"]:
        bi = base["tickers"].index("SPY")
        bs = base["prices_cpu"][bi, WINDOW:]
        bs = bs[bs > 0]
        if len(bs) > 1:
            bench_ret = (bs[-1] / bs[0] - 1.0) * 100

    try:
        actions, conf = _infer(model_path, base, args.confidence_temperature)
    except Exception as e:
        print(f"  inference failed — {e}")
        return

    sig = dict(
        T=T, N=N,
        tickers=base["tickers"],
        prices=base["prices_cpu"],
        active=base["active_cpu"],
        actions=actions,
        conf=conf,
    )

    rows = []
    combos = [(b, s) for b in buy_grid for s in sell_grid]
    for buy_c, sell_c in tqdm(combos, desc="Buy x Sell grid"):
        m = _simulate_unified(
            sig,
            bar_cache,
            capital=10_000.0,
            buy_conf=buy_c,
            sell_conf=sell_c,
            spy_fear_block_pct=spy_fear,
            calendar=calendar,
        )
        if not m:
            continue
        alpha = (m["total_ret"] - bench_ret) if bench_ret is not None else float("nan")
        rows.append((buy_c, sell_c, m, alpha))

    if not rows:
        print("No results.")
        return

    rows.sort(key=lambda r: r[2]["total_ret"], reverse=True)
    print(f"\n{'Buy':>5} {'Sell':>5} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>7} "
          f"{'Opens':>6} {'WR%':>6} {'AvgTr%':>8} {'Alpha':>8}")
    print("-" * 80)
    for buy_c, sell_c, m, alpha in rows[:15]:
        print(
            f"{buy_c:5.2f} {sell_c:5.2f} {m['total_ret']:+8.2f} {m['max_dd']:8.2f} "
            f"{m['sharpe']:7.2f} {m['opens']:6d} {m['wr']:6.1f} {m['avg_trade']:+8.2f} "
            f"{alpha:+8.2f}"
        )
    if bench_ret is not None:
        best = rows[0]
        kinds = best[2].get("opens_by_kind") or {}
        kind_str = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        print(f"\nSPY B&H: {bench_ret:+.2f}%  |  Best: buy {best[0]:.2f} / sell {best[1]:.2f} "
              f"({best[2]['total_ret']:+.2f}%, Sharpe {best[2]['sharpe']:.2f}, opens {best[2]['opens']})")
        if kind_str:
            print(f"  Opens by kind: {kind_str}")


def _filter_checkpoints(
    checkpoints: list[tuple[str, str]],
    only: str | None,
) -> list[tuple[str, str]]:
    if not only:
        return checkpoints
    want = {x.strip().lower() for x in only.split(",") if x.strip()}
    out = []
    for label, path in checkpoints:
        lab = label.lower()
        if lab in want or lab.replace("ep", "") in want:
            out.append((label, path))
    return out


def main():
    ap = argparse.ArgumentParser(description="Sweep unified options checkpoints")
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    ap.add_argument("--test-start-date", default=None)
    ap.add_argument("--test-end-date", default=None)
    ap.add_argument("--confidence-threshold", type=float, default=None)
    ap.add_argument("--sell-confidence-threshold", type=float, default=None)
    ap.add_argument(
        "--confidence-temperature", type=float,
        default=float(getattr(OptionsTraderConfig, "CONFIDENCE_TEMPERATURE", 0.01)),
    )
    ap.add_argument("--no-spy-filter", action="store_true")
    ap.add_argument("--prefix", default=MODEL_PREFIX)
    ap.add_argument("--disk-cache", action="store_true")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU inference (inflates returns ~30pts; do not use for deploy research)")
    ap.add_argument(
        "--only", default="",
        help="Comma-separated labels (ep50, best, …). Default: all discovered checkpoints",
    )
    ap.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Grid buy x sell thresholds on one model (--model, default ep100)",
    )
    ap.add_argument("--model", default=None, help="Model prefix for --sweep-thresholds")
    ap.add_argument("--buy-grid", default="0.60,0.65,0.70,0.75,0.80,0.85")
    ap.add_argument("--sell-grid", default="0.20,0.30,0.35,0.40,0.50")
    args = ap.parse_args()

    if args.sweep_thresholds:
        return _run_threshold_sweep(args)

    buy_conf = args.confidence_threshold or float(OptionsTraderConfig.CONFIDENCE_THRESHOLD)
    sell_conf = args.sell_confidence_threshold or float(OptionsTraderConfig.SELL_CONFIDENCE_THRESHOLD)

    checkpoints = _filter_checkpoints(_discover_checkpoints(args.prefix), args.only or None)
    if not checkpoints:
        print(f"No checkpoints matched for models/{args.prefix}_*")
        return

    spy_fear = None if args.no_spy_filter else OptionsTraderConfig.SPY_FEAR_BLOCK_PCT
    if args.disk_cache:
        bar_cache = make_option_bar_cache_from_disk()
        cache_note = "disk (historical_options/*.csv)"
    else:
        bar_cache = make_option_bar_cache_from_env(feed="indicative")
        cache_note = "Alpaca API on demand"

    print("=" * 88)
    print("UNIFIED OPTIONS CHECKPOINT SWEEP")
    print(f"  Watchlist: {args.watchlist}")
    print(f"  Marks: {cache_note}")
    print(f"  Period: {args.test_start_date or '2024-02-01'} -> {args.test_end_date or 'latest'}")
    print(f"  BUY conf > {buy_conf}, SELL conf > {sell_conf}")
    print(f"  Strategies: bullish + bearish (5 total)")
    print(f"  Checkpoints: {[c[0] for c in checkpoints]}")
    print("=" * 88)

    base = _load_base_signals(
        args.test_start_date or "2024-02-01",
        args.test_end_date,
        args.watchlist,
        force_cpu=args.cpu,
    )
    if base is None:
        print("No swing data.")
        return

    T, N = base["T"], base["N"]
    calendar = load_swing_calendar(
        T, test_start_date=args.test_start_date, test_end_date=args.test_end_date,
    )
    bench_ret = None
    if "SPY" in base["tickers"]:
        bi = base["tickers"].index("SPY")
        bs = base["prices_cpu"][bi, WINDOW:]
        bs = bs[bs > 0]
        if len(bs) > 1:
            bench_ret = (bs[-1] / bs[0] - 1.0) * 100

    rows = []
    for label, path in tqdm(checkpoints, desc="Unified checkpoints"):
        try:
            actions, conf = _infer(path, base, args.confidence_temperature)
        except Exception as e:
            print(f"  {label}: load failed — {e}")
            continue
        sig = dict(
            T=T, N=N,
            tickers=base["tickers"],
            prices=base["prices_cpu"],
            active=base["active_cpu"],
            actions=actions,
            conf=conf,
        )
        m = _simulate_unified(
            sig,
            bar_cache,
            capital=10_000.0,
            buy_conf=buy_conf,
            sell_conf=sell_conf,
            spy_fear_block_pct=spy_fear,
            calendar=calendar,
        )
        if not m:
            continue
        alpha = (m["total_ret"] - bench_ret) if bench_ret is not None else float("nan")
        rows.append((label, m, alpha))

    if not rows:
        print("No results.")
        return

    rows.sort(key=lambda r: r[1]["total_ret"], reverse=True)
    print(f"\n{'Label':<8} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'Opens':>6} "
          f"{'WR%':>6} {'AvgTr%':>8} {'Alpha':>8}  Opens by kind")
    print("-" * 100)
    for label, m, alpha in rows:
        kinds = m.get("opens_by_kind") or {}
        kind_str = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "-"
        print(
            f"{label:<8} {m['total_ret']:+8.2f} {m['max_dd']:8.2f} {m['sharpe']:7.2f} "
            f"{m['opens']:6d} {m['wr']:6.1f} {m['avg_trade']:+8.2f} {alpha:+8.2f}  {kind_str}"
        )
    if bench_ret is not None:
        best = rows[0]
        print(
            f"\nSPY B&H: {bench_ret:+.2f}%  |  Best: {best[0]} "
            f"({best[1]['total_ret']:+.2f}%, Sharpe {best[1]['sharpe']:.2f})"
        )


if __name__ == "__main__":
    main()
