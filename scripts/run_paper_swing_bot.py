"""
Paper swing — identical to live swing (SwingTraderConfig, Gen7 model, swing_liquid).

Only difference: PAPER_SWING_API_KEY / PAPER_SWING_API_SECRET and paper API host.

  .\\.venv\\Scripts\\python.exe scripts/run_paper_swing_bot.py
  .\\.venv\\Scripts\\python.exe scripts/run_paper_swing_bot.py --watchlist config/watchlists/swing_liquid.txt
"""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_DEFAULT_WATCHLIST = os.path.join("config", "watchlists", "swing_liquid.txt")

if __name__ == "__main__":
    extra = sys.argv[1:]
    if not any(a == "--watchlist" or a.startswith("--watchlist=") for a in extra):
        extra = ["--watchlist", _DEFAULT_WATCHLIST] + extra
    sys.argv = [sys.argv[0], "--mode", "paper_swing"] + extra
    runpy.run_module("src.bots.multimodal_trader", run_name="__main__")
