"""
Grid sweep: max_positions x buy confidence on one model (one inference pass).

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\sweep_options_params.py \\
      --model-path models/options_from_swing_200_ep50 \\
      --watchlist config/watchlists/options_liquid_200.txt \\
      --disk-cache
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OptionsTraderConfig  # noqa: E402

from scripts.backtest_options_portfolio import WINDOW  # noqa: E402
from scripts.sweep_options_from_swing import (  # noqa: E402
    _infer,
    _load_base_signals,
    _simulate,
)
from src.data.options_historical import (  # noqa: E402
    load_swing_calendar,
    make_option_bar_cache_from_disk,
    make_option_bar_cache_from_env,
)


def main():
    ap = argparse.ArgumentParser(description="Options slots x confidence grid (single model)")
    ap.add_argument("--model-path", default="models/options_from_swing_200_ep50")
    ap.add_argument("--watchlist", default="config/watchlists/options_liquid_200.txt")
    ap.add_argument("--test-start-date", default="2024-06-01")
    ap.add_argument("--test-end-date", default="2026-05-30")
    ap.add_argument("--slots-grid", default="20,30,40")
    ap.add_argument("--conf-grid", default="0.65,0.70,0.75,0.80")
    ap.add_argument("--sell-conf", type=float, default=0.35)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--disk-cache", action="store_true")
    ap.add_argument("--no-spy-filter", action="store_true")
    args = ap.parse_args()

    slots_list = [int(x) for x in args.slots_grid.split(",") if x.strip()]
    conf_list = [float(x) for x in args.conf_grid.split(",") if x.strip()]
    spy_fear = None if args.no_spy_filter else OptionsTraderConfig.SPY_FEAR_BLOCK_PCT

    if args.disk_cache:
        bar_cache = make_option_bar_cache_from_disk()
        marks = "disk"
    else:
        bar_cache = make_option_bar_cache_from_env(feed="indicative")
        marks = "API"

    print("=" * 88)
    print("OPTIONS PARAM GRID (one inference, shared marks)")
    print(f"  Model: {args.model_path}")
    print(f"  Watchlist: {args.watchlist}")
    print(f"  Marks: {marks}")
    print(f"  Slots: {slots_list}  |  Buy conf: {conf_list}")
    print("=" * 88)

    base = _load_base_signals(args.test_start_date, args.test_end_date, args.watchlist)
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

    actions, conf = _infer(args.model_path, base)
    sig = dict(
        T=T,
        N=N,
        tickers=base["tickers"],
        prices=base["prices_cpu"],
        active=base["active_cpu"],
        actions=actions,
        conf=conf,
    )

    rows = []
    for slots in slots_list:
        for buy_conf in conf_list:
            m = _simulate(
                sig,
                bar_cache,
                capital=args.capital,
                buy_conf=buy_conf,
                sell_conf=args.sell_conf,
                spy_fear_block_pct=spy_fear,
                calendar=calendar,
                max_positions=slots,
            )
            if not m:
                continue
            alpha = (m["total_ret"] - bench_ret) if bench_ret is not None else float("nan")
            rows.append((slots, buy_conf, m, alpha))

    if not rows:
        print("No results.")
        return

    rows.sort(key=lambda r: r[2]["total_ret"], reverse=True)
    print(
        f"\n{'Slots':>5} {'Conf':>5} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>7} "
        f"{'Opens':>6} {'WR%':>6} {'AvgTr%':>8} {'Alpha':>8}"
    )
    print("-" * 88)
    for slots, buy_conf, m, alpha in rows:
        print(
            f"{slots:5d} {buy_conf:5.2f} {m['total_ret']:+8.2f} {m['max_dd']:8.2f} "
            f"{m['sharpe']:7.2f} {m['opens']:6d} {m['wr']:6.1f} "
            f"{m['avg_trade']:+8.2f} {alpha:+8.2f}"
        )
    if bench_ret is not None:
        best = rows[0]
        print(
            f"\nSPY B&H: {bench_ret:+.2f}%  |  "
            f"Best: {best[0]} slots @ {best[1]:.2f} conf ({best[2]['total_ret']:+.2f}%)"
        )


if __name__ == "__main__":
    main()
