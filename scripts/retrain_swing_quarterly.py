"""
Quarterly swing model refresh workflow.

Steps:
  1. Refresh CSVs for the liquid watchlist
  2. Fine-tune from the production checkpoint (swing_gen7 or SWING_MODEL_PATH)
  3. Optional portfolio backtest validation on recent OOS window

Usage (PowerShell):
  .\\.venv\\Scripts\\python.exe scripts/retrain_swing_quarterly.py
  .\\.venv\\Scripts\\python.exe scripts/retrain_swing_quarterly.py --episodes 30 --validate

Schedule: run on the first weekend of each quarter (or after major market regime shifts).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SWING_MODEL_PATH, SwingTraderConfig

WATCHLIST = os.path.join("config", "watchlists", SwingTraderConfig.WATCHLIST)
MODEL_PREFIX = "swing_quarterly"


def _model_prefix_from_settings() -> str:
    path = str(SWING_MODEL_PATH)
    for suffix in ("_balanced.pth", "_aggressive.pth", "_conservative.pth"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path.replace(".pth", "")


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n==> {desc}")
    print("    " + " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Quarterly swing retrain + optional validation")
    ap.add_argument("--watchlist", default=WATCHLIST)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--validate", action="store_true",
                    help="Run portfolio backtest after training")
    ap.add_argument("--validate-months", type=int, default=12,
                    help="OOS window length for validation backtest")
    ap.add_argument("--init-from", default="", help="Override init checkpoint prefix")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    init_from = args.init_from.strip() or _model_prefix_from_settings()
    stamp = datetime.now().strftime("%Y%m%d")
    run_prefix = f"{MODEL_PREFIX}_{stamp}"
    out_prefix = f"models/{run_prefix}"

    if not args.skip_download:
        _run(
            [py, "-u", "scripts/build_liquid_watchlist.py", "--output", args.watchlist],
            "Rebuild liquid watchlist from CSVs",
        )
        _run(
            [
                py, "-u", "scripts/download_data.py",
                "--watchlist", args.watchlist,
                "--workers", str(args.workers),
            ],
            "Download / refresh liquid-universe CSVs",
        )

    _run(
        [
            py, "-u", "scripts/train_swing_phase2.py",
            "--episodes", str(args.episodes),
            "--init-from", init_from,
            "--model-prefix", run_prefix,
            "--num-symbols", "0",
        ],
        f"Fine-tune {args.episodes} episodes from {init_from}",
    )

    print(f"\nNew checkpoints saved under prefix: {out_prefix}_*")
    print(f"To promote: update SWING_MODEL_PATH in config/settings.py to:")
    print(f"  {out_prefix}_balanced.pth  (after reviewing ep* / best meta)")

    if args.validate:
        start = (datetime.now() - timedelta(days=30 * args.validate_months)).strftime("%Y-%m-%d")
        _run(
            [
                py, "-u", "scripts/backtest_swing_portfolio.py",
                init_from,
                "--test-start-date", start,
                "--watchlist", args.watchlist,
                "--max-positions", str(SwingTraderConfig.MAX_POSITIONS),
                "--confidence-threshold", str(SwingTraderConfig.CONFIDENCE_THRESHOLD),
                "--sell-confidence-threshold", str(SwingTraderConfig.SELL_CONFIDENCE_THRESHOLD),
            ],
            f"Validate production model OOS from {start}",
        )
        _run(
            [
                py, "-u", "scripts/backtest_swing_portfolio.py",
                f"{out_prefix}_best",
                "--test-start-date", start,
                "--watchlist", args.watchlist,
                "--max-positions", str(SwingTraderConfig.MAX_POSITIONS),
                "--confidence-threshold", str(SwingTraderConfig.CONFIDENCE_THRESHOLD),
                "--sell-confidence-threshold", str(SwingTraderConfig.SELL_CONFIDENCE_THRESHOLD),
            ],
            f"Validate new quarterly model OOS from {start}",
        )


if __name__ == "__main__":
    main()
