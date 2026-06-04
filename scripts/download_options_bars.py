"""
Download Alpaca historical option leg bars and build spread mark CSVs for training.

Prereqs: swing 1D CSVs in data/historical_swing/, DAY_API_KEY or SWING_API_KEY in .env

Usage:
  .\\.venv\\Scripts\\python.exe scripts/download_options_bars.py
  .\\.venv\\Scripts\\python.exe scripts/download_options_bars.py \\
      --watchlist config/watchlists/options_backtest_short.txt \\
      --start-date 2024-02-01 --end-date 2026-05-30
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OptionsTraderConfig  # noqa: E402
from src.data.options_historical import OPTIONS_DATA_MIN_DATE, make_option_bar_cache_from_env
from src.data.options_spread_dataset import download_watchlist, rebuild_spread_marks

_DEFAULT_WATCHLIST = os.path.join(
    "config", "watchlists", getattr(OptionsTraderConfig, "TRAIN_WATCHLIST", "options_liquid_200.txt")
)


def main():
    ap = argparse.ArgumentParser(description="Cache Alpaca option bars + spread marks")
    ap.add_argument(
        "--watchlist",
        default=_DEFAULT_WATCHLIST,
    )
    ap.add_argument("--start-date", default=OPTIONS_DATA_MIN_DATE.isoformat())
    ap.add_argument("--end-date", default=None, help="Default: today")
    ap.add_argument("--refresh", action="store_true", help="Re-fetch OCC files even if cached")
    ap.add_argument(
        "--marks-only",
        action="store_true",
        help="Rebuild spread mark CSVs from cached OCC files (no API)",
    )
    args = ap.parse_args()

    start = date.fromisoformat(args.start_date[:10])
    end = date.fromisoformat(args.end_date[:10]) if args.end_date else date.today()

    if args.marks_only:
        rebuild_spread_marks(args.watchlist, start=start, end=end)
        return

    cache = make_option_bar_cache_from_env(feed="indicative")
    download_watchlist(args.watchlist, cache, start=start, end=end, refresh=args.refresh)


if __name__ == "__main__":
    main()
