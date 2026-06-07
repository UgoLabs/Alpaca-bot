"""
Rebuild bullish + bearish mark CSVs from cached OCC bars (no API).

Usage:
  .\\.venv\\Scripts\\python.exe scripts/rebuild_multi_strategy_marks.py
  .\\.venv\\Scripts\\python.exe scripts/rebuild_multi_strategy_marks.py \\
      --watchlist config/watchlists/options_liquid_200.txt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OptionsTraderConfig  # noqa: E402
from src.data.options_spread_dataset import rebuild_spread_marks  # noqa: E402

DEFAULT_WATCHLIST = os.path.join(
    "config", "watchlists", getattr(OptionsTraderConfig, "TRAIN_WATCHLIST", "options_liquid_200.txt")
)


def main():
    ap = argparse.ArgumentParser(description="Rebuild multi-strategy options mark CSVs")
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    ap.add_argument(
        "--watchlist-short",
        action="store_true",
        help="Use options_backtest_short.txt (20 symbols) for a quick test run",
    )
    ap.add_argument("--test-start-date", default=None)
    args = ap.parse_args()
    if args.watchlist_short:
        args.watchlist = os.path.join("config", "watchlists", "options_backtest_short.txt")
    start = None
    if args.test_start_date:
        from datetime import date
        start = date.fromisoformat(args.test_start_date[:10])
    rebuild_spread_marks(args.watchlist, start=start)
    print("Done. Bullish: *_spread.csv  Bearish: *_bearish.csv")


if __name__ == "__main__":
    main()
