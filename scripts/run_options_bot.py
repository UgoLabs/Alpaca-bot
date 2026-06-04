"""
Run swing-signal options bot (call debit spreads, paper only, full swing_liquid universe).

Uses DAY_API_KEY / DAY_API_SECRET on OPTIONS_ALPACA_BASE_URL (defaults to paper).

  .\\.venv\\Scripts\\python.exe scripts/run_options_bot.py
  .\\.venv\\Scripts\\python.exe scripts/run_options_bot.py --watchlist config/watchlists/swing_liquid.txt
"""
from __future__ import annotations

import runpy
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--mode", "options"] + sys.argv[1:]
    runpy.run_module("src.bots.multimodal_trader", run_name="__main__")
