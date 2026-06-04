"""
Portfolio-level swing backtest (realistic, live-style).

Unlike backtest_swing.py (which runs every symbol as an independent $10k sleeve),
this simulates ONE shared account the way the live bot trades:

  - Single capital pool, MAX_POSITIONS concurrent slots (~equal weight).
  - Entries: agent BUY signal with confidence >= CONFIDENCE_THRESHOLD; when more
    buy signals than free slots, take the HIGHEST-confidence ones.
  - Exits: agent SELL signal with confidence >= SELL_CONFIDENCE_THRESHOLD (0 = any SELL
    action, mirroring the original backtest); optionally a hard percent stop.
  - Transaction cost charged per side; mark-to-market equity each bar.
  - Reports CAGR, max drawdown, Sharpe, exposure, trade stats, and SPY benchmark.

Reuses the exact data + features the production model/live pipeline use (11
features) via backtest_swing.load_swing_data.
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig, SwingTraderConfig  # noqa: E402
TrainingConfig.WINDOW_SIZE = 60

from scripts.backtest_swing import load_swing_data  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402

WINDOW = 60
DEFAULT_WATCHLIST = os.path.join("config", "watchlists", "swing_liquid.txt")


def _load_watchlist_symbols(path: str | None) -> set | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        syms = {ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")}
    syms.add("SPY")
    return syms


def _should_agent_exit(
    pnl_pct: float,
    peak_pct: float,
    bars_held: int,
    agent_sell: bool,
    sell_conf_ok: bool,
    stop_pct: float,
    entry_px: float,
    px: float,
    *,
    winner_trail: bool,
    min_hold_days: int,
    trail_activation_pct: float,
    trail_giveback_pct: float,
) -> bool:
    """Exit rules: hard stop, trailing winner, then gated agent SELL."""
    if stop_pct > 0 and entry_px > 0 and px <= entry_px * (1.0 - stop_pct):
        return True
    if winner_trail and peak_pct >= trail_activation_pct:
        if pnl_pct <= peak_pct - trail_giveback_pct:
            return True
        return False
    if not agent_sell or not sell_conf_ok:
        return False
    if winner_trail and bars_held < min_hold_days and pnl_pct > 0:
        return False
    if winner_trail and pnl_pct > 0 and peak_pct >= trail_activation_pct:
        return False
    return True


def _spy_day_change_pct(prices_cpu, tickers, t: int) -> float:
    if "SPY" not in tickers or t < 1:
        return 0.0
    bi = tickers.index("SPY")
    prev = float(prices_cpu[bi, t - 1])
    curr = float(prices_cpu[bi, t])
    if prev <= 0:
        return 0.0
    return (curr - prev) / prev * 100.0


class Portfolio:
    """One independent shared-capital portfolio state for a given (slots, conf, stop)."""

    def __init__(self, capital, max_positions, confidence_threshold, cost_per_side, stop_pct, n,
                 sell_confidence_threshold=0.0, winner_trail=False, min_hold_days=0,
                 trail_activation_pct=0.05, trail_giveback_pct=0.03):
        self.capital = float(capital)
        self.cash = float(capital)
        self.max_positions = max_positions
        self.conf = confidence_threshold
        self.sell_conf = sell_confidence_threshold  # 0 = any SELL action (no conf gate)
        self.cost = cost_per_side
        self.stop_pct = stop_pct
        self.winner_trail = winner_trail
        self.min_hold_days = min_hold_days
        self.trail_activation_pct = trail_activation_pct
        self.trail_giveback_pct = trail_giveback_pct
        self.shares = np.zeros(n)
        self.entry_px = np.zeros(n)
        self.held = set()
        self.entry_step = {}
        self.peak_pnl = {}
        self.equity_curve = []
        self.trade_pnls = []
        self.hold_lengths = []
        self.n_buys = 0

    def _sell(self, idx, px, step):
        proceeds = self.shares[idx] * px * (1.0 - self.cost)
        cost_basis = self.shares[idx] * self.entry_px[idx]
        self.cash += proceeds
        if cost_basis > 0:
            self.trade_pnls.append((proceeds - cost_basis) / cost_basis)
        self.hold_lengths.append(step - self.entry_step.get(idx, step))
        self.shares[idx] = 0.0
        self.entry_px[idx] = 0.0
        self.held.discard(idx)
        self.entry_step.pop(idx, None)
        self.peak_pnl.pop(idx, None)

    def step(self, t, actions_np, conf_np, px_t, active_t, block_new_buys: bool = False):
        # Exits
        for idx in list(self.held):
            px = px_t[idx]
            if px <= 0:
                continue
            entry = self.entry_px[idx]
            pnl_pct = (px - entry) / entry if entry > 0 else 0.0
            peak = max(self.peak_pnl.get(idx, pnl_pct), pnl_pct)
            self.peak_pnl[idx] = peak
            bars_held = t - self.entry_step.get(idx, t)
            agent_sell = actions_np[idx] == 2
            sell_conf_ok = self.sell_conf <= 0 or conf_np[idx] >= self.sell_conf
            if _should_agent_exit(
                pnl_pct, peak, bars_held, agent_sell, sell_conf_ok,
                self.stop_pct, entry, px,
                winner_trail=self.winner_trail,
                min_hold_days=self.min_hold_days,
                trail_activation_pct=self.trail_activation_pct,
                trail_giveback_pct=self.trail_giveback_pct,
            ):
                self._sell(idx, px, t)
        # Entries
        free = self.max_positions - len(self.held)
        if free > 0 and not block_new_buys:
            cand = np.where(
                (actions_np == 1) & (conf_np >= self.conf) & active_t & (px_t > 0)
            )[0]
            cand = [c for c in cand if c not in self.held]
            cand.sort(key=lambda c: conf_np[c], reverse=True)
            for idx in cand[:free]:
                equity_now = self.cash + float((self.shares * px_t).sum())
                per_slot = equity_now / self.max_positions
                alloc = min(per_slot, self.cash)
                if alloc < px_t[idx]:
                    continue
                qty = np.floor(alloc / (px_t[idx] * (1.0 + self.cost)))
                if qty <= 0:
                    continue
                self.cash -= qty * px_t[idx] * (1.0 + self.cost)
                self.shares[idx] = qty
                self.entry_px[idx] = px_t[idx]
                self.held.add(idx)
                self.entry_step[idx] = t
                self.n_buys += 1
        self.equity_curve.append(self.cash + float((self.shares * px_t).sum()))

    def metrics(self):
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
            "final": eq[-1], "total_ret": total_ret * 100, "cagr": cagr * 100,
            "max_dd": max_dd, "sharpe": sharpe, "trades": len(pnls), "wr": wr,
            "avg_trade": pnls.mean() * 100 if len(pnls) else 0.0,
            "avg_hold": np.mean(self.hold_lengths) if self.hold_lengths else 0.0,
            "buys": self.n_buys,
        }


def _load_and_infer(model_path, test_start_date, test_end_date, watchlist_path=None, ablation=None):
    """Load data + run agent inference once; return signals and prices for reuse."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(f"  universe={N} symbols, {T} bars, {F} features on {device}")
    data = data.to(device)
    prices = prices.to(device)
    changed = (prices != prices[:, :1])
    active = torch.cummax(changed.int(), dim=1).values.bool()

    agent = EnsembleAgent(time_series_dim=F, vision_channels=F, action_dim=3, device=device)
    agent.load(model_path)
    for sa in agent.agents:
        sa.epsilon = 0.0
        sa.policy_net.eval()
    dummy_ids = torch.zeros((N, 64), dtype=torch.long, device=device)
    dummy_mask = torch.ones((N, 64), dtype=torch.long, device=device)

    prices_cpu = prices.detach().cpu().numpy()
    active_cpu = active.detach().cpu().numpy()
    actions_all = np.zeros((T, N), dtype=np.int8)
    conf_all = np.zeros((T, N), dtype=np.float32)
    for t in range(WINDOW, T):
        obs = data[:, t - WINDOW:t, :]
        with torch.no_grad():
            actions, conf = agent.batch_act(
                obs, dummy_ids, dummy_mask, return_confidence=True, ablation=ablation,
            )
        actions_all[t] = actions.detach().cpu().numpy()
        conf_all[t] = conf.detach().cpu().numpy()
    label = ablation or "full"
    return dict(T=T, N=N, prices=prices_cpu, active=active_cpu,
                actions=actions_all, conf=conf_all, tickers=tickers, ablation=label)


def _portfolio_extras(winner_trail=None, min_hold_days=None,
                      trail_activation_pct=None, trail_giveback_pct=None):
    wt = winner_trail if winner_trail is not None else SwingTraderConfig.ENABLE_WINNER_TRAIL
    return dict(
        winner_trail=wt,
        min_hold_days=min_hold_days if min_hold_days is not None else SwingTraderConfig.MIN_HOLD_DAYS,
        trail_activation_pct=trail_activation_pct if trail_activation_pct is not None
        else SwingTraderConfig.TRAIL_ACTIVATION_PCT,
        trail_giveback_pct=trail_giveback_pct if trail_giveback_pct is not None
        else SwingTraderConfig.TRAIL_GIVEBACK_PCT,
    )


def compare_modal(model_path, test_start_date, test_end_date, capital, cost_per_side,
                  max_positions, confidence_threshold, sell_confidence_threshold,
                  benchmark="SPY", watchlist_path=None, spy_fear_block_pct=None, **portfolio_extras):
    """A/B inference: full multimodal vs price-only (ts_only). Zero text ids = production training path."""
    modes = [
        ("full (ts+vision, zero text ids)", None),
        ("no_text (zero text embedding)", "no_text"),
        ("no_vision (zero vision embedding)", "no_vision"),
        ("ts_only (zero vision+text)", "ts_only"),
    ]
    print("\n" + "=" * 90)
    print("MODAL ABLATION")
    print("  Backtest has no historical news; production uses zero text ids (TextHead bypass).")
    print("  no_text should match full. ts_only/no_vision test whether vision fusion carries edge.")
    print("=" * 90)
    bench_ret = None
    for label, abl in modes:
        sig = _load_and_infer(model_path, test_start_date, test_end_date, watchlist_path, ablation=abl)
        if sig is None:
            continue
        T, N = sig["T"], sig["N"]
        if benchmark in sig["tickers"] and bench_ret is None:
            bi = sig["tickers"].index(benchmark)
            bs = sig["prices"][bi, WINDOW:]
            bs = bs[bs > 0]
            if len(bs) > 1:
                bench_ret = bs[-1] / bs[0] - 1.0
        p = Portfolio(
            capital, max_positions, confidence_threshold, cost_per_side, 0.0, N,
            sell_confidence_threshold=sell_confidence_threshold,
            **_portfolio_extras(**portfolio_extras),
        )
        for t in range(WINDOW, T):
            block = False
            if spy_fear_block_pct is not None:
                block = _spy_day_change_pct(sig["prices"], sig["tickers"], t) < float(spy_fear_block_pct)
            p.step(t, sig["actions"][t], sig["conf"][t], sig["prices"][:, t],
                   sig["active"][:, t], block_new_buys=block)
        m = p.metrics()
        if not m:
            continue
        alpha = (m["total_ret"] - bench_ret * 100) if bench_ret is not None else float("nan")
        print(f"{label:40} ret={m['total_ret']:+7.2f}%  Sharpe={m['sharpe']:5.2f}  "
              f"maxDD={m['max_dd']:6.2f}%  trades={m['trades']:4d}  avgHold={m['avg_hold']:5.1f}  "
              f"alpha={alpha:+6.2f}%")
    print("=" * 90)


def sweep_cost(model_path, test_start_date, test_end_date, capital, cost_grid,
               max_positions, confidence_threshold, sell_confidence_threshold,
               benchmark="SPY", watchlist_path=None, spy_fear_block_pct=None, **portfolio_extras):
    """One inference pass; sweep round-trip friction (per side)."""
    sig = _load_and_infer(model_path, test_start_date, test_end_date, watchlist_path)
    if sig is None:
        return
    T, N = sig["T"], sig["N"]
    bench_ret = None
    if benchmark in sig["tickers"]:
        bi = sig["tickers"].index(benchmark)
        bs = sig["prices"][bi, WINDOW:]
        bs = bs[bs > 0]
        if len(bs) > 1:
            bench_ret = bs[-1] / bs[0] - 1.0
    ports = {
        c: Portfolio(
            capital, max_positions, confidence_threshold, c, 0.0, N,
            sell_confidence_threshold=sell_confidence_threshold,
            **_portfolio_extras(**portfolio_extras),
        )
        for c in cost_grid
    }
    for t in range(WINDOW, T):
        block = False
        if spy_fear_block_pct is not None:
            block = _spy_day_change_pct(sig["prices"], sig["tickers"], t) < float(spy_fear_block_pct)
        for p in ports.values():
            p.step(t, sig["actions"][t], sig["conf"][t], sig["prices"][:, t],
                   sig["active"][:, t], block_new_buys=block)
    print("\n" + "=" * 90)
    print(f"COST / SIDE SWEEP  slots={max_positions}  buy>={confidence_threshold:.2f}  "
          f"sell>={sell_confidence_threshold:.2f}")
    print("-" * 90)
    print(f"{'bps/side':>9} {'totRet%':>8} {'Sharpe':>7} {'maxDD%':>8} {'trades':>7} {'avgHold':>8} {'alpha%':>7}")
    for c in cost_grid:
        m = ports[c].metrics()
        if not m:
            continue
        alpha = (m["total_ret"] - bench_ret * 100) if bench_ret is not None else float("nan")
        print(f"{c*10000:>9.0f} {m['total_ret']:>8.2f} {m['sharpe']:>7.2f} {m['max_dd']:>8.2f} "
              f"{m['trades']:>7d} {m['avg_hold']:>8.1f} {alpha:>7.2f}")
    print("=" * 90)


def sweep_trail(model_path, test_start_date, test_end_date, capital, cost_per_side,
                max_positions, confidence_threshold, sell_confidence_threshold,
                act_grid, giveback_grid, min_hold_grid,
                benchmark="SPY", watchlist_path=None, spy_fear_block_pct=None):
    """One inference pass; grid winner-trail params (+ trail-off baseline)."""
    sig = _load_and_infer(model_path, test_start_date, test_end_date, watchlist_path)
    if sig is None:
        return
    T, N = sig["T"], sig["N"]
    bench_ret = None
    if benchmark in sig["tickers"]:
        bi = sig["tickers"].index(benchmark)
        bs = sig["prices"][bi, WINDOW:]
        bs = bs[bs > 0]
        if len(bs) > 1:
            bench_ret = bs[-1] / bs[0] - 1.0

    configs = [("off", dict(winner_trail=False))]
    for act in act_grid:
        for gb in giveback_grid:
            for mh in min_hold_grid:
                label = f"{act*100:.0f}/{gb*100:.0f}/{mh}d"
                configs.append((
                    label,
                    dict(
                        winner_trail=True,
                        trail_activation_pct=float(act),
                        trail_giveback_pct=float(gb),
                        min_hold_days=int(mh),
                    ),
                ))

    ports = {}
    for label, extra in configs:
        ports[label] = Portfolio(
            capital, max_positions, confidence_threshold, cost_per_side, 0.0, N,
            sell_confidence_threshold=sell_confidence_threshold,
            **_portfolio_extras(**extra),
        )

    for t in range(WINDOW, T):
        block = False
        if spy_fear_block_pct is not None:
            block = _spy_day_change_pct(sig["prices"], sig["tickers"], t) < float(spy_fear_block_pct)
        for p in ports.values():
            p.step(t, sig["actions"][t], sig["conf"][t], sig["prices"][:, t],
                   sig["active"][:, t], block_new_buys=block)

    print("\n" + "=" * 100)
    hdr = (f"WINNER TRAIL SWEEP  slots={max_positions}  buy>={confidence_threshold:.2f}  "
           f"sell>={sell_confidence_threshold:.2f}  cost={cost_per_side*10000:.0f}bps/side")
    if bench_ret is not None:
        hdr += f"  SPY {bench_ret*100:+.2f}%"
    print(hdr)
    print("  Columns: act%=trail starts after unrealized gain; gb%=exit giveback from peak; hold=min green days")
    print("-" * 100)
    print(f"{'config':>14} {'totRet%':>8} {'Sharpe':>7} {'maxDD%':>8} {'trades':>7} "
          f"{'avgHold':>8} {'win%':>6} {'alpha%':>7}")
    best = None
    for label, _ in configs:
        m = ports[label].metrics()
        if not m:
            continue
        alpha = (m["total_ret"] - bench_ret * 100) if bench_ret is not None else float("nan")
        print(f"{label:>14} {m['total_ret']:>8.2f} {m['sharpe']:>7.2f} {m['max_dd']:>8.2f} "
              f"{m['trades']:>7d} {m['avg_hold']:>8.1f} {m['wr']:>6.1f} {alpha:>7.2f}")
        if best is None or m["sharpe"] > best[1]["sharpe"]:
            best = (label, m)
    print("=" * 100)
    if best:
        lbl, m = best
        print(f"Best Sharpe: {lbl}  ret={m['total_ret']:+.2f}%  maxDD={m['max_dd']:.2f}%  "
              f"trades={m['trades']}  avgHold={m['avg_hold']:.1f}d")


def sweep_sell(model_path, test_start_date, test_end_date, capital, cost_per_side,
               max_positions, confidence_threshold, sell_conf_grid, benchmark="SPY",
               watchlist_path=None, spy_fear_block_pct=None, **portfolio_extras):
    """Sweep sell-confidence at fixed slots + buy-confidence (one inference pass)."""
    sig = _load_and_infer(model_path, test_start_date, test_end_date, watchlist_path)
    if sig is None:
        print("No data.")
        return
    T, N = sig["T"], sig["N"]
    bench_ret = None
    if benchmark in sig["tickers"]:
        bi = sig["tickers"].index(benchmark)
        bs = sig["prices"][bi, WINDOW:]
        bs = bs[bs > 0]
        if len(bs) > 1:
            bench_ret = bs[-1] / bs[0] - 1.0

    ports = {
        sc: Portfolio(
            capital, max_positions, confidence_threshold, cost_per_side, 0.0, N,
            sell_confidence_threshold=sc,
            **_portfolio_extras(**portfolio_extras),
        )
        for sc in sell_conf_grid
    }
    fear_days = 0
    for t in range(WINDOW, T):
        a = sig["actions"][t]
        cf = sig["conf"][t]
        px = sig["prices"][:, t]
        act = sig["active"][:, t]
        block = False
        if spy_fear_block_pct is not None:
            spy_chg = _spy_day_change_pct(sig["prices"], sig["tickers"], t)
            block = spy_chg < float(spy_fear_block_pct)
            if block:
                fear_days += 1
        for p in ports.values():
            p.step(t, a, cf, px, act, block_new_buys=block)
    if spy_fear_block_pct is not None:
        print(f"  SPY fear filter active on {fear_days}/{T - WINDOW} bars (threshold {spy_fear_block_pct:.1f}%)")

    buy_label = f"buy>={confidence_threshold:.2f}"
    sell_note = "sellConf=0 means any SELL action (no confidence gate)"
    print("\n" + "=" * 90)
    hdr = (f"SELL CONFIDENCE SWEEP  slots={max_positions}  {buy_label}  "
           f"(start {test_start_date}, cost {cost_per_side*100:.3f}%/side")
    if bench_ret is not None:
        hdr += f", SPY {bench_ret*100:+.2f}%)"
    else:
        hdr += ")"
    print(hdr)
    print(sell_note)
    print("-" * 90)
    print(f"{'sellConf':>8} {'totRet%':>8} {'CAGR%':>7} {'maxDD%':>8} "
          f"{'Sharpe':>7} {'trades':>7} {'win%':>6} {'avgT%':>7} {'alpha%':>7}")
    best = None
    for sc in sell_conf_grid:
        m = ports[sc].metrics()
        if not m:
            continue
        alpha = (m["total_ret"] - bench_ret * 100) if bench_ret is not None else float("nan")
        label = f"{sc:.2f}" if sc > 0 else "any"
        print(f"{label:>8} {m['total_ret']:>8.2f} {m['cagr']:>7.2f} {m['max_dd']:>8.2f} "
              f"{m['sharpe']:>7.2f} {m['trades']:>7d} {m['wr']:>6.1f} {m['avg_trade']:>7.2f} {alpha:>7.2f}")
        if best is None or m["sharpe"] > best[1]["sharpe"]:
            best = (sc, m)
    print("=" * 90)
    if best:
        sc, m = best
        lbl = f"{sc:.2f}" if sc > 0 else "any (no gate)"
        print(f"Best Sharpe: sellConf={lbl}  totRet={m['total_ret']:+.2f}%  "
              f"maxDD={m['max_dd']:.2f}%  trades={m['trades']}")


def sweep(model_path, test_start_date, test_end_date, capital, cost_per_side,
          slots_grid, conf_grid, benchmark="SPY", watchlist_path=None, spy_fear_block_pct=None,
          **portfolio_extras):
    sig = _load_and_infer(model_path, test_start_date, test_end_date, watchlist_path)
    if sig is None:
        print("No data.")
        return
    T, N = sig["T"], sig["N"]
    bench_ret = None
    if benchmark in sig["tickers"]:
        bi = sig["tickers"].index(benchmark)
        bs = sig["prices"][bi, WINDOW:]
        bs = bs[bs > 0]
        if len(bs) > 1:
            bench_ret = bs[-1] / bs[0] - 1.0

    configs = [(s, c) for s in slots_grid for c in conf_grid]
    extras = _portfolio_extras(**portfolio_extras)
    ports = {
        (s, c): Portfolio(capital, s, c, cost_per_side, 0.0, N, **extras)
        for (s, c) in configs
    }
    for t in range(WINDOW, T):
        a = sig["actions"][t]
        cf = sig["conf"][t]
        px = sig["prices"][:, t]
        act = sig["active"][:, t]
        block = False
        if spy_fear_block_pct is not None:
            block = _spy_day_change_pct(sig["prices"], sig["tickers"], t) < float(spy_fear_block_pct)
        for p in ports.values():
            p.step(t, a, cf, px, act, block_new_buys=block)

    print("\n" + "=" * 86)
    print(f"PORTFOLIO SWEEP  (start {test_start_date}, cost {cost_per_side*100:.3f}%/side, "
          f"SPY {bench_ret*100:+.2f}%)" if bench_ret is not None else "PORTFOLIO SWEEP")
    print("-" * 86)
    print(f"{'slots':>5} {'conf':>5} {'totRet%':>8} {'CAGR%':>7} {'maxDD%':>8} "
          f"{'Sharpe':>7} {'trades':>7} {'win%':>6} {'avgT%':>7} {'alpha%':>7}")
    for (s, c) in configs:
        m = ports[(s, c)].metrics()
        if not m:
            continue
        alpha = (m["total_ret"] - bench_ret * 100) if bench_ret is not None else float("nan")
        print(f"{s:>5} {c:>5.2f} {m['total_ret']:>8.2f} {m['cagr']:>7.2f} {m['max_dd']:>8.2f} "
              f"{m['sharpe']:>7.2f} {m['trades']:>7d} {m['wr']:>6.1f} {m['avg_trade']:>7.2f} {alpha:>7.2f}")
    print("=" * 86)


def run(
    model_path="models/swing_gen7_refined_ep380",
    test_start_date="2024-01-01",
    test_end_date=None,
    capital=100_000.0,
    max_positions=10,
    confidence_threshold=0.50,
    sell_confidence_threshold=0.0,
    cost_per_side=0.0005,         # 5 bps/side (commission-free + spread/slippage)
    stop_pct=0.0,                 # optional hard stop (0 = off, agent-only exits)
    benchmark="SPY",
    watchlist_path=None,
    spy_fear_block_pct=None,
    winner_trail=None,
    min_hold_days=None,
    trail_activation_pct=None,
    trail_giveback_pct=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Portfolio swing backtest on {device}")
    sell_label = "any SELL" if sell_confidence_threshold <= 0 else f"sellConf>={sell_confidence_threshold:.2f}"
    spy_label = "off" if spy_fear_block_pct is None else f"<{spy_fear_block_pct:.1f}%"
    wt = winner_trail if winner_trail is not None else SwingTraderConfig.ENABLE_WINNER_TRAIL
    trail_lbl = "on" if wt else "off"
    print(f"  capital=${capital:,.0f}  slots={max_positions}  conf>={confidence_threshold}  "
          f"{sell_label}  spyFear={spy_label}  cost={cost_per_side*10000:.0f}bps/side  "
          f"winnerTrail={trail_lbl}  stop={stop_pct*100:.1f}%  start={test_start_date}")

    include = _load_watchlist_symbols(watchlist_path)
    data, prices, _stop, _atr, tickers = load_swing_data(
        "data/historical_swing",
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        include_symbols=include,
    )
    if data is None:
        print("No data.")
        return

    N, T, F = data.shape
    print(f"  universe={N} symbols, {T} bars, {F} features")

    data = data.to(device)
    prices = prices.to(device)

    # "Active" mask: skip the edge-padded leading region (pad repeats col 0).
    changed = (prices != prices[:, :1])
    active = torch.cummax(changed.int(), dim=1).values.bool()  # (N,T)

    agent = EnsembleAgent(time_series_dim=F, vision_channels=F, action_dim=3, device=device)
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Could not load model: {e}")
        return
    for sa in agent.agents:
        sa.epsilon = 0.0
        sa.policy_net.eval()

    dummy_ids = torch.zeros((N, 64), dtype=torch.long, device=device)
    dummy_mask = torch.ones((N, 64), dtype=torch.long, device=device)

    cash = float(capital)
    shares = np.zeros(N)         # shares held per symbol
    entry_px = np.zeros(N)       # avg entry price per held symbol
    held = set()
    entry_step = {}
    peak_pnl = {}
    extras = _portfolio_extras(
        winner_trail=winner_trail,
        min_hold_days=min_hold_days,
        trail_activation_pct=trail_activation_pct,
        trail_giveback_pct=trail_giveback_pct,
    )

    equity_curve = []
    trade_pnls = []              # realized pct returns per closed trade
    hold_lengths = []
    n_buys = 0
    prices_cpu = prices.detach().cpu().numpy()

    def sell(idx, px, step):
        nonlocal cash
        proceeds = shares[idx] * px * (1.0 - cost_per_side)
        cost_basis = shares[idx] * entry_px[idx]
        cash += proceeds
        if cost_basis > 0:
            trade_pnls.append((proceeds - cost_basis) / cost_basis)
        hold_lengths.append(step - entry_step.get(idx, step))
        shares[idx] = 0.0
        entry_px[idx] = 0.0
        held.discard(idx)
        entry_step.pop(idx, None)

    for t in range(WINDOW, T):
        obs = data[:, t - WINDOW:t, :]
        with torch.no_grad():
            actions, conf = agent.batch_act(obs, dummy_ids, dummy_mask, return_confidence=True)
        actions_np = actions.detach().cpu().numpy()
        conf_np = conf.detach().cpu().numpy()
        px_t = prices_cpu[:, t]
        active_t = active[:, t].detach().cpu().numpy()

        # 1) EXITS: hard stop, winner trail, then gated agent SELL.
        for idx in list(held):
            px = px_t[idx]
            if px <= 0:
                continue
            entry = entry_px[idx]
            pnl_pct = (px - entry) / entry if entry > 0 else 0.0
            peak = max(peak_pnl.get(idx, pnl_pct), pnl_pct)
            peak_pnl[idx] = peak
            bars_held = t - entry_step.get(idx, t)
            agent_sell = actions_np[idx] == 2
            sell_conf_ok = sell_confidence_threshold <= 0 or conf_np[idx] >= sell_confidence_threshold
            if _should_agent_exit(
                pnl_pct, peak, bars_held, agent_sell, sell_conf_ok,
                stop_pct, entry, px,
                winner_trail=extras["winner_trail"],
                min_hold_days=extras["min_hold_days"],
                trail_activation_pct=extras["trail_activation_pct"],
                trail_giveback_pct=extras["trail_giveback_pct"],
            ):
                sell(idx, px, t)
                peak_pnl.pop(idx, None)

        # 2) ENTRIES: rank buy signals by confidence, fill free slots.
        block_buys = False
        if spy_fear_block_pct is not None:
            block_buys = _spy_day_change_pct(prices_cpu, tickers, t) < float(spy_fear_block_pct)
        free = max_positions - len(held)
        if free > 0 and not block_buys:
            cand = np.where(
                (actions_np == 1) & (conf_np >= confidence_threshold) & active_t & (px_t > 0)
            )[0]
            cand = [c for c in cand if c not in held]
            cand.sort(key=lambda c: conf_np[c], reverse=True)
            cand = cand[:free]
            if cand:
                equity_now = cash + float((shares * px_t).sum())
                per_slot = equity_now / max_positions
                for idx in cand:
                    alloc = min(per_slot, cash)
                    if alloc < px_t[idx]:
                        continue
                    qty = np.floor(alloc / (px_t[idx] * (1.0 + cost_per_side)))
                    if qty <= 0:
                        continue
                    spend = qty * px_t[idx] * (1.0 + cost_per_side)
                    cash -= spend
                    shares[idx] = qty
                    entry_px[idx] = px_t[idx]
                    held.add(idx)
                    entry_step[idx] = t
                    n_buys += 1

        equity_curve.append(cash + float((shares * px_t).sum()))

    # ---- Metrics ----
    eq = np.array(equity_curve)
    if len(eq) < 2:
        print("Too few steps.")
        return
    total_ret = eq[-1] / capital - 1.0
    years = len(eq) / 252.0
    cagr = (eq[-1] / capital) ** (1 / years) - 1.0 if years > 0 else 0.0
    rets = np.diff(eq) / eq[:-1]
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252) if rets.std() > 0 else 0.0
    run_max = np.maximum.accumulate(eq)
    max_dd = ((eq - run_max) / run_max).min() * 100
    pnls = np.array(trade_pnls)
    wr = (pnls > 0).mean() * 100 if len(pnls) else 0.0

    # Benchmark
    bench_ret = None
    if benchmark in tickers:
        bi = tickers.index(benchmark)
        bseries = prices_cpu[bi, WINDOW:]
        bseries = bseries[bseries > 0]
        if len(bseries) > 1:
            bench_ret = bseries[-1] / bseries[0] - 1.0

    print("\n" + "=" * 60)
    print("PORTFOLIO RESULTS")
    print("-" * 60)
    print(f"  Final equity:     ${eq[-1]:,.0f}  (start ${capital:,.0f})")
    print(f"  Total return:     {total_ret*100:+.2f}%   over {years:.2f}y")
    print(f"  CAGR:             {cagr*100:+.2f}%")
    print(f"  Max drawdown:     {max_dd:.2f}%")
    print(f"  Sharpe (daily):   {sharpe:.2f}")
    print(f"  Closed trades:    {len(pnls)}   win rate {wr:.1f}%   total buys {n_buys}")
    if len(pnls):
        print(f"  Avg trade:        {pnls.mean()*100:+.2f}%   avg hold {np.mean(hold_lengths):.1f} bars")
    if bench_ret is not None:
        print(f"  {benchmark} buy&hold:    {bench_ret*100:+.2f}%   (alpha {(total_ret-bench_ret)*100:+.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Portfolio-level swing backtest")
    ap.add_argument("model_path", nargs="?", default="models/swing_gen7_refined_ep380")
    ap.add_argument("--test-start-date", default="2024-01-01")
    ap.add_argument("--test-end-date", default=None)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--max-positions", type=int, default=SwingTraderConfig.MAX_POSITIONS)
    ap.add_argument("--confidence-threshold", type=float, default=SwingTraderConfig.CONFIDENCE_THRESHOLD)
    ap.add_argument("--sell-confidence-threshold", type=float,
                    default=SwingTraderConfig.SELL_CONFIDENCE_THRESHOLD,
                    help="Min confidence for agent SELL exits (0 = any SELL action)")
    ap.add_argument("--cost-per-side", type=float, default=0.0005)
    ap.add_argument("--stop-pct", type=float, default=0.0)
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--sweep", action="store_true",
                    help="Grid over slots x confidence (shares inference across configs)")
    ap.add_argument("--slots-grid", default="10,15,20")
    ap.add_argument("--conf-grid", default="0.5,0.6,0.7")
    ap.add_argument("--sweep-sell", action="store_true",
                    help="Grid over sell-confidence at fixed --max-positions and --confidence-threshold")
    ap.add_argument("--sell-conf-grid", default="0.25,0.35,0.40",
                    help="Sell confidence levels (0 = any SELL action, no gate)")
    ap.add_argument("--sweep-cost", action="store_true",
                    help="Sweep cost-per-side (see --cost-grid)")
    ap.add_argument("--cost-grid", default="0.0005,0.001,0.0015",
                    help="Per-side costs as decimals (0.001 = 10 bps)")
    ap.add_argument("--compare-modal", action="store_true",
                    help="A/B full vs ts_only inference (no historical news in backtest)")
    ap.add_argument("--no-winner-trail", action="store_true",
                    help="Disable hold-winners trailing exit logic")
    ap.add_argument("--compare-winner-trail", action="store_true",
                    help="Run portfolio with vs without winner trail")
    ap.add_argument("--sweep-trail", action="store_true",
                    help="Grid winner-trail act%% / giveback%% / min-hold (+ off baseline)")
    ap.add_argument("--trail-act-grid", default="0.03,0.05,0.08,0.10",
                    help="Activation %% as decimals (0.05 = +5%% unrealized)")
    ap.add_argument("--trail-giveback-grid", default="0.02,0.03,0.04,0.05",
                    help="Giveback from peak as decimals")
    ap.add_argument("--min-hold-grid", default="0,2,5,10",
                    help="Min hold days before agent SELL on winners")
    ap.add_argument("--trail-activation-pct", type=float, default=None,
                    help="Winner trail activation (default: SwingTraderConfig)")
    ap.add_argument("--trail-giveback-pct", type=float, default=None,
                    help="Winner trail giveback from peak (default: SwingTraderConfig)")
    ap.add_argument("--min-hold-days", type=int, default=None,
                    help="Min days before agent SELL on green positions")
    ap.add_argument("--watchlist", default=DEFAULT_WATCHLIST,
                    help="Symbol list for universe (default: swing_liquid.txt)")
    ap.add_argument("--spy-filter", action="store_true",
                    help="Enable live-style SPY fear filter on new buys")
    ap.add_argument("--spy-fear-pct", type=float, default=SwingTraderConfig.SPY_FEAR_BLOCK_PCT)
    ap.add_argument("--spy-sweep", action="store_true",
                    help="Compare portfolio with vs without SPY fear filter")
    args = ap.parse_args()
    watchlist = args.watchlist if os.path.exists(args.watchlist) else None
    spy_pct = float(args.spy_fear_pct) if args.spy_filter else None
    winner_trail = SwingTraderConfig.ENABLE_WINNER_TRAIL
    if args.no_winner_trail:
        winner_trail = False
    pextra = dict(winner_trail=winner_trail)
    if args.trail_activation_pct is not None:
        pextra["trail_activation_pct"] = args.trail_activation_pct
    if args.trail_giveback_pct is not None:
        pextra["trail_giveback_pct"] = args.trail_giveback_pct
    if args.min_hold_days is not None:
        pextra["min_hold_days"] = args.min_hold_days

    if args.compare_modal:
        compare_modal(
            model_path=args.model_path,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            capital=args.capital,
            cost_per_side=args.cost_per_side,
            max_positions=args.max_positions,
            confidence_threshold=args.confidence_threshold,
            sell_confidence_threshold=args.sell_confidence_threshold,
            benchmark=args.benchmark,
            watchlist_path=watchlist,
            spy_fear_block_pct=spy_pct,
            **pextra,
        )
        sys.exit(0)
    if args.sweep_cost:
        sweep_cost(
            model_path=args.model_path,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            capital=args.capital,
            cost_grid=[float(x) for x in args.cost_grid.split(",")],
            max_positions=args.max_positions,
            confidence_threshold=args.confidence_threshold,
            sell_confidence_threshold=args.sell_confidence_threshold,
            benchmark=args.benchmark,
            watchlist_path=watchlist,
            spy_fear_block_pct=spy_pct,
            **pextra,
        )
        sys.exit(0)
    if args.sweep_trail:
        sweep_trail(
            model_path=args.model_path,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            capital=args.capital,
            cost_per_side=args.cost_per_side,
            max_positions=args.max_positions,
            confidence_threshold=args.confidence_threshold,
            sell_confidence_threshold=args.sell_confidence_threshold,
            act_grid=[float(x) for x in args.trail_act_grid.split(",")],
            giveback_grid=[float(x) for x in args.trail_giveback_grid.split(",")],
            min_hold_grid=[int(x) for x in args.min_hold_grid.split(",")],
            benchmark=args.benchmark,
            watchlist_path=watchlist,
            spy_fear_block_pct=spy_pct,
        )
        sys.exit(0)
    if args.compare_winner_trail:
        for label, wt in [("trail off", False), ("trail on", True)]:
            print(f"\n--- Winner trail: {label} ---")
            run(
                model_path=args.model_path,
                test_start_date=args.test_start_date,
                test_end_date=args.test_end_date,
                capital=args.capital,
                max_positions=args.max_positions,
                confidence_threshold=args.confidence_threshold,
                sell_confidence_threshold=args.sell_confidence_threshold,
                cost_per_side=args.cost_per_side,
                stop_pct=args.stop_pct,
                benchmark=args.benchmark,
                watchlist_path=watchlist,
                spy_fear_block_pct=spy_pct,
                winner_trail=wt,
            )
        sys.exit(0)
    if args.spy_sweep:
        print("SPY FEAR FILTER A/B (same inference, 40 slots / buy 0.70 / sell 0.35)")
        for label, pct in [("off", None), (f"on (<{args.spy_fear_pct:.1f}%)", float(args.spy_fear_pct))]:
            print(f"\n--- {label} ---")
            run(
                model_path=args.model_path,
                test_start_date=args.test_start_date,
                test_end_date=args.test_end_date,
                capital=args.capital,
                max_positions=args.max_positions,
                confidence_threshold=args.confidence_threshold,
                sell_confidence_threshold=args.sell_confidence_threshold,
                cost_per_side=args.cost_per_side,
                stop_pct=args.stop_pct,
                benchmark=args.benchmark,
                watchlist_path=watchlist,
                spy_fear_block_pct=pct,
            )
        sys.exit(0)
    if args.sweep_sell:
        sweep_sell(
            model_path=args.model_path,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            capital=args.capital,
            cost_per_side=args.cost_per_side,
            max_positions=args.max_positions,
            confidence_threshold=args.confidence_threshold,
            sell_conf_grid=[float(x) for x in args.sell_conf_grid.split(",")],
            benchmark=args.benchmark,
            watchlist_path=watchlist,
            spy_fear_block_pct=spy_pct,
            **pextra,
        )
        sys.exit(0)
    if args.sweep:
        sweep(
            model_path=args.model_path,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            capital=args.capital,
            cost_per_side=args.cost_per_side,
            slots_grid=[int(x) for x in args.slots_grid.split(",")],
            conf_grid=[float(x) for x in args.conf_grid.split(",")],
            benchmark=args.benchmark,
            watchlist_path=watchlist,
            spy_fear_block_pct=spy_pct,
            **pextra,
        )
        sys.exit(0)
    run(
        model_path=args.model_path,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        capital=args.capital,
        max_positions=args.max_positions,
        confidence_threshold=args.confidence_threshold,
        sell_confidence_threshold=args.sell_confidence_threshold,
        cost_per_side=args.cost_per_side,
        stop_pct=args.stop_pct,
        benchmark=args.benchmark,
        watchlist_path=watchlist,
        spy_fear_block_pct=spy_pct,
        winner_trail=winner_trail,
    )
