"""
Sweep options_from_swing checkpoints on one options backtest (single data load).

Usage:
  .\\.venv\\Scripts\\python.exe scripts/sweep_options_from_swing.py
  .\\.venv\\Scripts\\python.exe scripts/sweep_options_from_swing.py --confidence-threshold 0.7
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import date

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (  # noqa: E402
    OptionsTraderConfig, SWING_MODEL_PATH, OPTIONS_MODEL_PATH, TrainingConfig,
)

TrainingConfig.WINDOW_SIZE = 60

from scripts.backtest_options_portfolio import (  # noqa: E402
    OptionsSpreadPortfolio,
    WINDOW,
    _spy_day_change_pct,
)
from scripts.backtest_swing import load_swing_data  # noqa: E402
from scripts.backtest_swing_portfolio import _load_watchlist_symbols  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.data.options_historical import (  # noqa: E402
    make_option_bar_cache_from_disk,
    make_option_bar_cache_from_env,
    load_swing_calendar,
)

DEFAULT_WATCHLIST = os.path.join("config", "watchlists", "options_backtest_short.txt")
MODEL_PREFIX = "options_from_swing"


def _discover_checkpoints(prefix: str) -> list[tuple[str, str]]:
    """Return [(label, path_prefix), ...] sorted by episode number."""
    found: list[tuple[int, str, str]] = []
    for path in glob.glob(os.path.join("models", f"{prefix}_ep*_balanced.pth")):
        base = os.path.basename(path)
        marker = f"{prefix}_ep"
        try:
            ep = int(base.split(marker, 1)[1].split("_", 1)[0])
            found.append((ep, f"ep{ep}", os.path.join("models", f"{prefix}_ep{ep}")))
        except Exception:
            continue
    found.sort(key=lambda x: x[0])

    extras = [
        ("best", os.path.join("models", f"{prefix}_best")),
        ("last", os.path.join("models", f"{prefix}_last")),
    ]
    out = [(label, p) for _, label, p in found]
    for label, p in extras:
        if os.path.isfile(f"{p}_balanced.pth"):
            out.append((label, p))
    return out


def _load_base_signals(
    test_start_date: str,
    test_end_date: str | None,
    watchlist_path: str,
    force_cpu: bool = False,
):
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    include = _load_watchlist_symbols(watchlist_path)
    data, prices, _stop, _atr, tickers = load_swing_data(
        "data/historical_swing",
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        include_symbols=include,
    )
    if data is None:
        return None
    N, T, F = data.shape
    data = data.to(device)
    prices = prices.to(device)
    changed = prices != prices[:, :1]
    active = torch.cummax(changed.int(), dim=1).values.bool()
    prices_cpu = prices.detach().cpu().numpy()
    active_cpu = active.detach().cpu().numpy()
    return dict(device=device, data=data, N=N, T=T, F=F, tickers=tickers,
                prices_cpu=prices_cpu, active_cpu=active_cpu)


def _infer(model_path: str, base: dict,
           confidence_temperature: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    device = base["device"]
    agent = EnsembleAgent(
        time_series_dim=base["F"],
        vision_channels=base["F"],
        action_dim=3,
        device=device,
    )
    agent.load(model_path)
    for sa in agent.agents:
        sa.epsilon = 0.0
        sa.policy_net.eval()

    T, N = base["T"], base["N"]
    data = base["data"]
    dummy_ids = torch.zeros((N, 64), dtype=torch.long, device=device)
    dummy_mask = torch.ones((N, 64), dtype=torch.long, device=device)
    actions_all = np.zeros((T, N), dtype=np.int8)
    conf_all = np.zeros((T, N), dtype=np.float32)
    for t in range(WINDOW, T):
        obs = data[:, t - WINDOW : t, :]
        with torch.no_grad():
            actions, conf = agent.batch_act(
                obs, dummy_ids, dummy_mask, return_confidence=True,
                confidence_temperature=confidence_temperature,
            )
        actions_all[t] = actions.detach().cpu().numpy()
        conf_all[t] = conf.detach().cpu().numpy()
    return actions_all, conf_all


def _simulate(
    sig: dict,
    bar_cache,
    *,
    capital: float,
    buy_conf: float,
    sell_conf: float,
    spy_fear_block_pct: float | None,
    calendar: list[date],
    max_positions: int | None = None,
) -> dict | None:
    cfg = OptionsTraderConfig
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
    """Infer once on a single model, then grid over buy x sell thresholds."""
    model_path = _strip_model_suffix(args.model or str(OPTIONS_MODEL_PATH))
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
    print("OPTIONS THRESHOLD SWEEP (one model, one inference, buy x sell grid)")
    print(f"  Model: {model_path}")
    print(f"  Watchlist: {args.watchlist}")
    print(f"  Marks: {cache_note}")
    print(f"  Period: {args.test_start_date} -> {args.test_end_date}")
    print(f"  Confidence temperature: {args.confidence_temperature}")
    print(f"  Buy grid:  {buy_grid}")
    print(f"  Sell grid: {sell_grid}")
    print("=" * 88)

    base = _load_base_signals(
        args.test_start_date, args.test_end_date, args.watchlist, force_cpu=args.cpu,
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
        m = _simulate(
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
    for buy_c, sell_c, m, alpha in rows:
        print(
            f"{buy_c:5.2f} {sell_c:5.2f} {m['total_ret']:+8.2f} {m['max_dd']:8.2f} "
            f"{m['sharpe']:7.2f} {m['opens']:6d} {m['wr']:6.1f} {m['avg_trade']:+8.2f} "
            f"{alpha:+8.2f}"
        )
    if bench_ret is not None:
        best = rows[0]
        print(f"\nSPY B&H: {bench_ret:+.2f}%  |  Best: buy {best[0]:.2f} / sell {best[1]:.2f} "
              f"({best[2]['total_ret']:+.2f}%, Sharpe {best[2]['sharpe']:.2f})")


def main():
    ap = argparse.ArgumentParser(description="Sweep options_from_swing checkpoints")
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    ap.add_argument("--test-start-date", default="2024-06-01")
    ap.add_argument("--test-end-date", default="2026-05-30")
    ap.add_argument("--confidence-threshold", type=float, default=0.7)
    ap.add_argument("--sell-confidence-threshold", type=float, default=0.35)
    ap.add_argument(
        "--confidence-temperature", type=float,
        default=float(getattr(OptionsTraderConfig, "CONFIDENCE_TEMPERATURE", 0.01)),
        help="Softmax temp for confidence (default = OptionsTraderConfig). "
             "Higher spreads confidence out; pick via scripts/measure_q_gap.py.",
    )
    ap.add_argument("--no-spy-filter", action="store_true")
    ap.add_argument("--prefix", default=MODEL_PREFIX)
    ap.add_argument(
        "--include-swing-baseline",
        action="store_true",
        help="Also backtest SWING_MODEL_PATH (ep380)",
    )
    ap.add_argument(
        "--disk-cache",
        action="store_true",
        help="Use downloaded OCC CSVs (required for large watchlists)",
    )
    ap.add_argument(
        "--also",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Extra checkpoint, e.g. s2_best:models/options_from_swing_s2_best",
    )
    ap.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Pin ONE model (--model, default deployed OPTIONS_MODEL_PATH), infer once, "
             "then grid over buy x sell confidence thresholds.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Single model prefix for --sweep-thresholds (default: deployed OPTIONS_MODEL_PATH)",
    )
    ap.add_argument(
        "--buy-grid",
        default="0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated buy confidence thresholds for --sweep-thresholds",
    )
    ap.add_argument(
        "--sell-grid",
        default="0.20,0.35,0.50",
        help="Comma-separated sell confidence thresholds for --sweep-thresholds",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (recommended while GPU training is running)",
    )
    args = ap.parse_args()

    if args.sweep_thresholds:
        return _run_threshold_sweep(args)

    checkpoints = _discover_checkpoints(args.prefix)
    if args.include_swing_baseline:
        swing_p = str(SWING_MODEL_PATH).replace("_balanced.pth", "").replace(
            "_aggressive.pth", ""
        ).replace("_conservative.pth", "")
        if os.path.isfile(f"{swing_p}_balanced.pth"):
            checkpoints.insert(0, ("gen7_ep380", swing_p))
    for item in args.also:
        if ":" not in item:
            continue
        label, path = item.split(":", 1)
        path = path.strip()
        if os.path.isfile(f"{path}_balanced.pth"):
            checkpoints.append((label.strip(), path))

    if not checkpoints:
        print(f"No checkpoints found for models/{args.prefix}_*")
        return

    spy_fear = None if args.no_spy_filter else OptionsTraderConfig.SPY_FEAR_BLOCK_PCT
    if args.disk_cache:
        bar_cache = make_option_bar_cache_from_disk()
        cache_note = "disk (historical_options/*.csv)"
    else:
        bar_cache = make_option_bar_cache_from_env(feed="indicative")
        cache_note = "Alpaca API on demand"

    print("=" * 88)
    print("OPTIONS CHECKPOINT SWEEP (one data load, shared option bar cache)")
    print(f"  Watchlist: {args.watchlist}")
    print(f"  Marks: {cache_note}")
    print(f"  Period: {args.test_start_date} -> {args.test_end_date}")
    print(f"  Buy conf > {args.confidence_threshold}")
    print(f"  Confidence temperature: {args.confidence_temperature}")
    print(f"  Checkpoints: {len(checkpoints)}")
    print("=" * 88)

    base = _load_base_signals(
        args.test_start_date, args.test_end_date, args.watchlist, force_cpu=args.cpu,
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
    for label, path in tqdm(checkpoints, desc="Checkpoints"):
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
        m = _simulate(
            sig,
            bar_cache,
            capital=10_000.0,
            buy_conf=args.confidence_threshold,
            sell_conf=args.sell_confidence_threshold,
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
    print(f"\n{'Label':<14} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'Opens':>6} "
          f"{'WR%':>6} {'AvgTr%':>8} {'Alpha':>8}")
    print("-" * 88)
    for label, m, alpha in rows:
        print(
            f"{label:<14} {m['total_ret']:+8.2f} {m['max_dd']:8.2f} {m['sharpe']:7.2f} "
            f"{m['opens']:6d} {m['wr']:6.1f} {m['avg_trade']:+8.2f} {alpha:+8.2f}"
        )
    if bench_ret is not None:
        print(f"\nSPY B&H: {bench_ret:+.2f}%  |  Best: {rows[0][0]} ({rows[0][1]['total_ret']:+.2f}%)")


if __name__ == "__main__":
    main()
