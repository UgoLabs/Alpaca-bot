"""
Measure the ensemble Q-value gap to choose an informative confidence temperature.

Background
----------
Live/backtest "confidence" = softmax(avg_Q / temperature)[chosen_action], where
avg_Q is the mean of the 3 ensemble policy-net Q-vectors. The historical
temperature is 0.01, which divides Q by 0.01 (x100) and saturates the softmax to
a near-hard argmax (~1.0 for almost every pick) -> the CONFIDENCE_THRESHOLD gate
carries little information for options.

This script measures the *raw* gap between the top-1 and top-2 averaged Q-values
across a universe/time window, then shows what confidence distribution each
candidate temperature would produce. Pick the temperature whose confidence
spreads across a usable range (so a threshold like 0.55-0.70 actually selects).

Reads only data/historical_swing + a model checkpoint; changes nothing.

Usage (PowerShell, single line):
  .\.venv\Scripts\python.exe scripts/measure_q_gap.py --model models/options_from_swing_200_ep30
  .\.venv\Scripts\python.exe scripts/measure_q_gap.py --model models/options_from_swing_200_ep30 --watchlist config/watchlists/options_liquid_200.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OptionsTraderConfig, TrainingConfig  # noqa: E402

TrainingConfig.WINDOW_SIZE = 60

from scripts.backtest_options_portfolio import WINDOW  # noqa: E402
from scripts.backtest_swing import load_swing_data  # noqa: E402
from scripts.backtest_swing_portfolio import _load_watchlist_symbols  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402

DEFAULT_WATCHLIST = os.path.join("config", "watchlists", "options_backtest_short.txt")
DEFAULT_MODEL = os.path.join("models", "options_from_swing_200_ep30")
CANDIDATE_TEMPS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
PCTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def _strip_suffix(path: str) -> str:
    for suf in ("_balanced.pth", "_aggressive.pth", "_conservative.pth"):
        if path.endswith(suf):
            return path[: -len(suf)]
    return path


def _collect_avg_q(model_path: str, watchlist: str, start: str, end: str,
                   buys_only: bool, force_cpu: bool = False) -> np.ndarray:
    """Return averaged-Q vectors (rows = active samples, cols = action_dim)."""
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    include = _load_watchlist_symbols(watchlist)
    data, prices, _stop, _atr, tickers = load_swing_data(
        "data/historical_swing",
        test_start_date=start,
        test_end_date=end,
        include_symbols=include,
    )
    if data is None:
        raise SystemExit("No swing data for the requested window/watchlist.")
    N, T, F = data.shape
    print(f"  universe={N} symbols, {T} bars, {F} features on {device}")
    data = data.to(device)
    prices = prices.to(device)
    changed = prices != prices[:, :1]
    active = torch.cummax(changed.int(), dim=1).values.bool()

    agent = EnsembleAgent(time_series_dim=F, vision_channels=F, action_dim=3, device=device)
    agent.load(model_path)
    for sa in agent.agents:
        sa.epsilon = 0.0
        sa.policy_net.eval()
    dummy_ids = torch.zeros((N, 64), dtype=torch.long, device=device)
    dummy_mask = torch.ones((N, 64), dtype=torch.long, device=device)

    rows = []
    for t in range(WINDOW, T):
        obs = data[:, t - WINDOW : t, :]
        with torch.inference_mode():
            q_vecs = [sa.policy_net(obs, dummy_ids, dummy_mask) for sa in agent.agents]
            q_avg = torch.stack(q_vecs, dim=0).float().mean(dim=0)  # (N, action_dim)
        q_np = q_avg.detach().cpu().numpy()
        act_mask = active[:, t].detach().cpu().numpy()
        sel = q_np[act_mask]
        if buys_only and sel.size:
            sel = sel[sel.argmax(axis=1) == 1]  # action 1 == BUY
        if sel.size:
            rows.append(sel)
    if not rows:
        raise SystemExit("No active samples collected.")
    return np.concatenate(rows, axis=0)


def _softmax_conf(q: np.ndarray, temp: float) -> np.ndarray:
    z = q / max(1e-6, temp)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    return p[np.arange(len(p)), q.argmax(axis=1)]


def main():
    ap = argparse.ArgumentParser(description="Measure ensemble Q-gap & temperature effect")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    ap.add_argument("--test-start-date", default="2024-06-01")
    ap.add_argument("--test-end-date", default="2026-05-30")
    ap.add_argument("--buys-only", action="store_true",
                    help="Only measure samples where the chosen action is BUY")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU (recommended while GPU training is running)")
    ap.add_argument("--temps", default=",".join(str(t) for t in CANDIDATE_TEMPS),
                    help="Comma-separated candidate temperatures")
    args = ap.parse_args()

    model_path = _strip_suffix(args.model)
    temps = [float(x) for x in args.temps.split(",") if x.strip()]

    print("=" * 84)
    print("Q-GAP MEASUREMENT")
    print(f"  Model:     {model_path}")
    print(f"  Watchlist: {args.watchlist}")
    print(f"  Period:    {args.test_start_date} -> {args.test_end_date}")
    print(f"  Samples:   {'BUY picks only' if args.buys_only else 'all active picks'}")
    print("=" * 84)

    q = _collect_avg_q(model_path, args.watchlist, args.test_start_date,
                       args.test_end_date, args.buys_only, force_cpu=args.cpu)
    sorted_q = np.sort(q, axis=1)  # ascending
    top1 = sorted_q[:, -1]
    top2 = sorted_q[:, -2]
    gap = top1 - top2

    print(f"\nSamples measured: {len(q):,}")
    print(f"Avg-Q range: min={q.min():.5f}  max={q.max():.5f}  mean={q.mean():.5f}")
    print("\nTop1 - Top2 Q gap percentiles:")
    for p in PCTILES:
        print(f"  p{p:<2d} = {np.percentile(gap, p):.6f}")
    print(f"  mean = {gap.mean():.6f}   std = {gap.std():.6f}")

    print("\nConfidence distribution by temperature "
          "(want a spread, not all ~1.0):")
    header = f"{'temp':>7} | " + " ".join(f"p{p:<2d}".rjust(7) for p in PCTILES) + f" | {'mean':>7} {'%>0.99':>7}"
    print(header)
    print("-" * len(header))
    for t in temps:
        conf = _softmax_conf(q, t)
        cells = " ".join(f"{np.percentile(conf, p):7.4f}" for p in PCTILES)
        frac_sat = float((conf > 0.99).mean()) * 100
        print(f"{t:7.3f} | {cells} | {conf.mean():7.4f} {frac_sat:6.1f}%")

    print("\nGuidance: pick the smallest temperature whose confidence still spans a")
    print("usable range (e.g. p25-p90 straddling 0.5-0.9) so CONFIDENCE_THRESHOLD")
    print("(buy ~0.55-0.70) actually discriminates. %>0.99 near 100% => saturated.")


if __name__ == "__main__":
    main()
