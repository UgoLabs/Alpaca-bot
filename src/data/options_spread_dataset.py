"""
Build and load option spread marks from cached Alpaca option leg bars.

Files:
  data/historical_options/{OCC}.csv — per-contract daily close
  data/historical_options_marks/{SYM}_spread.csv — call debit (legacy) + optional bullish columns
  data/historical_options_marks/{SYM}_bearish.csv — put debit + bear call credit
"""
from __future__ import annotations

import glob
import os
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config.settings import OptionsTraderConfig
from src.data.options_historical import (
    OPTIONS_DATA_MIN_DATE,
    OptionSpreadBarCache,
    build_occ_symbol,
    pick_expiration,
)
from src.data.options_mark_planner import (
    BEARISH_KINDS,
    BULLISH_KINDS,
    plan_spread,
)
from src.execution.options_strategies import SpreadKind

MARKS_DIR = os.path.join("data", "historical_options_marks")
OCC_DIR = os.path.join("data", "historical_options")
SWING_DIR = os.path.join("data", "historical_swing")

# Per-process OCC close cache (rebuild touches each leg once per symbol, not per day).
_OCC_CACHE: dict[str, dict[date, float]] = {}


def _round_strike(spot: float, width: float = 5.0) -> tuple[float, float]:
    step = width if spot >= 200 else (2.5 if spot < 100 else width)
    long_k = round(spot / step) * step
    if long_k <= 0:
        long_k = max(step, spot)
    return float(long_k), float(long_k + width)


def load_watchlist(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f
            if ln.strip() and not ln.startswith("#")
        ]


def _read_swing_close_series(symbol: str) -> pd.Series | None:
    path = os.path.join(SWING_DIR, f"{symbol}_1D.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df["Date"] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index("Date").sort_index()
    close_col = "close" if "close" in df.columns else "Close"
    if close_col not in df.columns:
        return None
    return df[close_col].astype(float)


def clear_occ_cache() -> None:
    _OCC_CACHE.clear()


def _load_occ_closes(occ: str) -> dict[date, float]:
    occ = occ.upper()
    if occ in _OCC_CACHE:
        return _OCC_CACHE[occ]
    path = os.path.join(OCC_DIR, f"{occ}.csv")
    if not os.path.isfile(path):
        _OCC_CACHE[occ] = {}
        return _OCC_CACHE[occ]
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or df.columns[0]
    close_col = cols.get("close")
    if close_col is None:
        _OCC_CACHE[occ] = {}
        return _OCC_CACHE[occ]
    dates = pd.to_datetime(df[date_col]).dt.date
    out = dict(zip(dates, df[close_col].astype(float)))
    _OCC_CACHE[occ] = out
    return out


def _warm_occ_cache_for_symbol(
    symbol: str,
    day_spots: list[tuple[date, float]],
    *,
    kinds: tuple = BULLISH_KINDS + BEARISH_KINDS,
) -> int:
    """Load all OCC CSVs needed for one underlying into memory. Returns leg count."""
    cfg = OptionsTraderConfig
    occ: set[str] = set()
    for on, spot in day_spots:
        plans = _plan_day_strategies(
            symbol, spot, on, kinds,
            spread_width=cfg.SPREAD_WIDTH,
            scale_width=cfg.SCALE_WIDTH_BY_PRICE,
            target_dte=cfg.TARGET_DTE,
            min_dte=cfg.MIN_DTE,
            max_dte=cfg.MAX_DTE,
        )
        for plan in plans.values():
            occ.add(plan["long_sym"])
            if plan.get("short_sym"):
                occ.add(plan["short_sym"])
    for o in occ:
        _load_occ_closes(o)
    return len(occ)


def save_occ_bars(occ: str, closes: dict[date, float]) -> None:
    os.makedirs(OCC_DIR, exist_ok=True)
    if not closes:
        return
    rows = [{"Date": d, "Close": c} for d, c in sorted(closes.items())]
    pd.DataFrame(rows).to_csv(os.path.join(OCC_DIR, f"{occ.upper()}.csv"), index=False)


def plan_occ_symbols_for_symbol(
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
    spread_width: float | None = None,
    target_dte: int | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
) -> tuple[set[str], list[tuple[date, str, str]]]:
    """Return all OCC legs needed and per-day (date, long_occ, short_occ)."""
    cfg = OptionsTraderConfig
    spread_width = spread_width if spread_width is not None else cfg.SPREAD_WIDTH
    target_dte = target_dte if target_dte is not None else cfg.TARGET_DTE
    min_dte = min_dte if min_dte is not None else cfg.MIN_DTE
    max_dte = max_dte if max_dte is not None else cfg.MAX_DTE

    closes = _read_swing_close_series(symbol)
    if closes is None or closes.empty:
        return set(), []

    start = start or OPTIONS_DATA_MIN_DATE
    end = end or date.today()
    occ_set: set[str] = set()
    day_plans: list[tuple[date, str, str]] = []

    for ts, spot in closes.items():
        on = pd.Timestamp(ts).date()
        if on < start or on > end or on < OPTIONS_DATA_MIN_DATE:
            continue
        if spot <= 0 or not np.isfinite(spot):
            continue
        long_k, short_k = _round_strike(float(spot), spread_width)
        exp = pick_expiration(on, target_dte=target_dte, min_dte=min_dte, max_dte=max_dte)
        long_sym = build_occ_symbol(symbol, exp, "C", long_k)
        short_sym = build_occ_symbol(symbol, exp, "C", short_k)
        occ_set.add(long_sym)
        occ_set.add(short_sym)
        day_plans.append((on, long_sym, short_sym))

    return occ_set, day_plans


def _leg_mark(
    plan: dict,
    on: date,
    *,
    slip: float,
) -> tuple[float, float, int]:
    """Return (mark, entry_premium, tradable) for one planned structure."""
    long_sym = plan["long_sym"]
    short_sym = plan.get("short_sym", "")
    if plan.get("single_leg"):
        lc = _load_occ_closes(long_sym).get(on)
        if lc is None or lc <= 0:
            return np.nan, np.nan, 0
        mark = max(0.0, lc * 100.0)
        prem = mark * (1.0 + slip) if mark > 0 else np.nan
        return mark, prem, 1 if mark > 0 and prem and prem > 0 else 0

    lc = _load_occ_closes(long_sym).get(on)
    sc = _load_occ_closes(short_sym).get(on)
    if lc is None or sc is None:
        return np.nan, np.nan, 0
    if plan.get("is_credit"):
        mark = max(0.0, (sc - lc) * 100.0)
        prem = mark * (1.0 - slip) if mark > 0 else np.nan
    else:
        mark = max(0.0, (lc - sc) * 100.0)
        prem = mark * (1.0 + slip) if mark > 0 else np.nan
    tradable = 1 if mark > 0 and prem and prem > 0 else 0
    return mark, prem, tradable


def _plan_day_strategies(
    symbol: str,
    spot: float,
    on: date,
    kinds: tuple[SpreadKind, ...],
    *,
    spread_width: float,
    scale_width: bool,
    target_dte: int,
    min_dte: int,
    max_dte: int,
) -> dict[SpreadKind, dict]:
    out: dict[SpreadKind, dict] = {}
    for kind in kinds:
        plan = plan_spread(
            symbol, spot, on, kind,
            spread_width=spread_width,
            scale_width=scale_width,
            target_dte=target_dte,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        if plan:
            out[kind] = plan
    return out


def build_bullish_marks_csv(
    symbol: str,
    day_spots: list[tuple[date, float]],
    *,
    slip: float = 0.08,
    spread_width: float | None = None,
    target_dte: int | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
    scale_width: bool | None = None,
) -> pd.DataFrame:
    cfg = OptionsTraderConfig
    spread_width = spread_width if spread_width is not None else cfg.SPREAD_WIDTH
    target_dte = target_dte if target_dte is not None else cfg.TARGET_DTE
    min_dte = min_dte if min_dte is not None else cfg.MIN_DTE
    max_dte = max_dte if max_dte is not None else cfg.MAX_DTE
    scale_width = scale_width if scale_width is not None else cfg.SCALE_WIDTH_BY_PRICE

    rows = []
    for on, spot in day_spots:
        plans = _plan_day_strategies(
            symbol, spot, on, BULLISH_KINDS,
            spread_width=spread_width, scale_width=scale_width,
            target_dte=target_dte, min_dte=min_dte, max_dte=max_dte,
        )
        row: dict = {"Date": on}
        cd = plans.get(SpreadKind.CALL_DEBIT)
        if cd:
            m, p, t = _leg_mark(cd, on, slip=slip)
            row.update({
                "spread_mark": m, "entry_premium": p, "tradable": t,
                "long_occ": cd["long_sym"], "short_occ": cd["short_sym"],
            })
        else:
            row.update({
                "spread_mark": np.nan, "entry_premium": np.nan, "tradable": 0,
                "long_occ": "", "short_occ": "",
            })
        for kind, prefix in (
            (SpreadKind.BULL_PUT_CREDIT, "bpc"),
            (SpreadKind.LONG_CALL, "lc"),
        ):
            plan = plans.get(kind)
            if plan:
                m, p, t = _leg_mark(plan, on, slip=slip)
                row[f"{prefix}_mark"] = m
                row[f"{prefix}_premium"] = p
                row[f"{prefix}_tradable"] = t
                row[f"{prefix}_credit"] = 1 if plan.get("is_credit") else 0
            else:
                row[f"{prefix}_mark"] = np.nan
                row[f"{prefix}_premium"] = np.nan
                row[f"{prefix}_tradable"] = 0
                row[f"{prefix}_credit"] = 1 if kind == SpreadKind.BULL_PUT_CREDIT else 0
        rows.append(row)
    return pd.DataFrame(rows)


def build_bearish_marks_csv(
    symbol: str,
    day_spots: list[tuple[date, float]],
    *,
    slip: float = 0.08,
    spread_width: float | None = None,
    target_dte: int | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
    scale_width: bool | None = None,
) -> pd.DataFrame:
    cfg = OptionsTraderConfig
    spread_width = spread_width if spread_width is not None else cfg.SPREAD_WIDTH
    target_dte = target_dte if target_dte is not None else cfg.TARGET_DTE
    min_dte = min_dte if min_dte is not None else cfg.MIN_DTE
    max_dte = max_dte if max_dte is not None else cfg.MAX_DTE
    scale_width = scale_width if scale_width is not None else cfg.SCALE_WIDTH_BY_PRICE

    rows = []
    for on, spot in day_spots:
        plans = _plan_day_strategies(
            symbol, spot, on, BEARISH_KINDS,
            spread_width=spread_width, scale_width=scale_width,
            target_dte=target_dte, min_dte=min_dte, max_dte=max_dte,
        )
        row: dict = {"Date": on}
        for kind, prefix in (
            (SpreadKind.PUT_DEBIT, "pd"),
            (SpreadKind.BEAR_CALL_CREDIT, "bcc"),
        ):
            plan = plans.get(kind)
            if plan:
                m, p, t = _leg_mark(plan, on, slip=slip)
                row[f"{prefix}_mark"] = m
                row[f"{prefix}_premium"] = p
                row[f"{prefix}_tradable"] = t
                row[f"{prefix}_credit"] = 1 if plan.get("is_credit") else 0
            else:
                row[f"{prefix}_mark"] = np.nan
                row[f"{prefix}_premium"] = np.nan
                row[f"{prefix}_tradable"] = 0
                row[f"{prefix}_credit"] = 1 if kind == SpreadKind.BEAR_CALL_CREDIT else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _day_spots_for_symbol(
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[tuple[date, float]]:
    closes = _read_swing_close_series(symbol)
    if closes is None or closes.empty:
        return []
    start = start or OPTIONS_DATA_MIN_DATE
    end = end or date.today()
    out: list[tuple[date, float]] = []
    for ts, spot in closes.items():
        on = pd.Timestamp(ts).date()
        if on < start or on > end or on < OPTIONS_DATA_MIN_DATE:
            continue
        if spot <= 0 or not np.isfinite(spot):
            continue
        out.append((on, float(spot)))
    return out


def plan_all_occ_for_symbol(
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
    kinds: tuple[SpreadKind, ...] = BULLISH_KINDS + BEARISH_KINDS,
) -> set[str]:
    cfg = OptionsTraderConfig
    occ: set[str] = set()
    for on, spot in _day_spots_for_symbol(symbol, start=start, end=end):
        plans = _plan_day_strategies(
            symbol, spot, on, kinds,
            spread_width=cfg.SPREAD_WIDTH,
            scale_width=cfg.SCALE_WIDTH_BY_PRICE,
            target_dte=cfg.TARGET_DTE,
            min_dte=cfg.MIN_DTE,
            max_dte=cfg.MAX_DTE,
        )
        for plan in plans.values():
            occ.add(plan["long_sym"])
            if plan.get("short_sym"):
                occ.add(plan["short_sym"])
    return occ


def build_spread_marks_csv(
    symbol: str,
    day_plans: list[tuple[date, str, str]],
    *,
    slip: float = 0.08,
) -> pd.DataFrame:
    rows = []
    for on, long_sym, short_sym in day_plans:
        lc = _load_occ_closes(long_sym).get(on)
        sc = _load_occ_closes(short_sym).get(on)
        if lc is None or sc is None or lc <= 0:
            rows.append(
                {
                    "Date": on,
                    "spread_mark": np.nan,
                    "entry_premium": np.nan,
                    "tradable": 0,
                    "long_occ": long_sym,
                    "short_occ": short_sym,
                }
            )
            continue
        mark = max(0.0, (lc - sc) * 100.0)
        prem = mark * (1.0 + slip) if mark > 0 else np.nan
        rows.append(
            {
                "Date": on,
                "spread_mark": mark,
                "entry_premium": prem,
                "tradable": 1 if mark > 0 and prem and prem > 0 else 0,
                "long_occ": long_sym,
                "short_occ": short_sym,
            }
        )
    return pd.DataFrame(rows)


def download_symbol_options(
    symbol: str,
    cache: OptionSpreadBarCache,
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool = False,
) -> int:
    """Fetch missing OCC leg bars and write spread marks CSV. Returns OCC files saved."""
    occ_set = plan_all_occ_for_symbol(symbol, start=start, end=end)
    if not occ_set:
        return 0

    fetch_start = start or OPTIONS_DATA_MIN_DATE
    fetch_end = end or date.today()
    range_start = fetch_start - timedelta(days=3)
    range_end = fetch_end + timedelta(days=7)

    occ_list = sorted(occ_set)
    need_api = [
        occ for occ in occ_list
        if refresh or not os.path.isfile(os.path.join(OCC_DIR, f"{occ}.csv"))
    ]
    for i in range(0, len(need_api), 100):
        cache.ensure_symbols(need_api[i : i + 100], range_start, range_end)

    saved = 0
    for occ in occ_list:
        out_path = os.path.join(OCC_DIR, f"{occ}.csv")
        closes = cache._closes.get(occ.upper(), {}) or _load_occ_closes(occ)
        if refresh or not os.path.isfile(out_path):
            if closes:
                save_occ_bars(occ, closes)
                saved += 1

    os.makedirs(MARKS_DIR, exist_ok=True)
    day_spots = _day_spots_for_symbol(symbol, start=start, end=end)
    if day_spots:
        build_bullish_marks_csv(symbol, day_spots).to_csv(
            os.path.join(MARKS_DIR, f"{symbol.upper()}_spread.csv"), index=False
        )
        build_bearish_marks_csv(symbol, day_spots).to_csv(
            os.path.join(MARKS_DIR, f"{symbol.upper()}_bearish.csv"), index=False
        )
    return saved


def rebuild_spread_marks(watchlist_path: str, *, start: date | None = None, end: date | None = None) -> None:
    """Rebuild bullish + bearish mark CSVs from cached OCC files (no API)."""
    symbols = load_watchlist(watchlist_path)
    os.makedirs(MARKS_DIR, exist_ok=True)
    for sym in tqdm(symbols, desc="Rebuild marks"):
        day_spots = _day_spots_for_symbol(sym, start=start, end=end)
        if not day_spots:
            continue
        clear_occ_cache()
        n_legs = _warm_occ_cache_for_symbol(sym, day_spots)
        tqdm.write(f"  {sym}: {len(day_spots)} days, {n_legs} OCC legs cached")
        build_bullish_marks_csv(sym, day_spots).to_csv(
            os.path.join(MARKS_DIR, f"{sym.upper()}_spread.csv"), index=False
        )
        build_bearish_marks_csv(sym, day_spots).to_csv(
            os.path.join(MARKS_DIR, f"{sym.upper()}_bearish.csv"), index=False
        )
    print(f"Rebuilt bullish/bearish marks in {MARKS_DIR}")


def download_watchlist(
    watchlist_path: str,
    cache: OptionSpreadBarCache,
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool = False,
) -> None:
    symbols = load_watchlist(watchlist_path)
    print(f"Downloading option bars for {len(symbols)} symbols -> {OCC_DIR}")
    total_saved = 0
    for sym in tqdm(symbols, desc="Symbols"):
        total_saved += download_symbol_options(sym, cache, start=start, end=end, refresh=refresh)
    print(f"Done. Updated {total_saved} OCC files; marks in {MARKS_DIR}")


def _strategy_specs(mode: str) -> list[tuple[str, str, str, bool]]:
    """(mark_col, prem_col, tradable_col, is_credit) per strategy in priority order."""
    if mode == "bullish":
        return [
            ("spread_mark", "entry_premium", "tradable", False),
            ("bpc_mark", "bpc_premium", "bpc_tradable", True),
            ("lc_mark", "lc_premium", "lc_tradable", False),
        ]
    if mode == "bearish":
        return [
            ("pd_mark", "pd_premium", "pd_tradable", False),
            ("bcc_mark", "bcc_premium", "bcc_tradable", True),
        ]
    if mode == "all":
        return [
            ("spread_mark", "entry_premium", "tradable", False),
            ("bpc_mark", "bpc_premium", "bpc_tradable", True),
            ("lc_mark", "lc_premium", "lc_tradable", False),
            ("pd_mark", "pd_premium", "pd_tradable", False),
            ("bcc_mark", "bcc_premium", "bcc_tradable", True),
        ]
    raise ValueError(f"Unknown mode: {mode}")


def load_options_multi_training_tensors(
    watchlist_path: str,
    *,
    mode: str = "bullish",
    min_date: date | None = None,
) -> (
    tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.BoolTensor,
        torch.BoolTensor,
        list[str],
    ]
    | None
):
    """Load features + (N,T,S) marks for multi-strategy training."""
    from src.core.indicators import add_technical_indicators

    min_date = min_date or OPTIONS_DATA_MIN_DATE
    symbols = load_watchlist(watchlist_path)
    specs = _strategy_specs(mode)
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "rsi", "macd", "macd_signal", "adx", "sma_20", "ema_12",
    ]

    processed = []
    for sym in symbols:
        swing_path = os.path.join(SWING_DIR, f"{sym}_1D.csv")
        spread_path = os.path.join(MARKS_DIR, f"{sym}_spread.csv")
        bearish_path = os.path.join(MARKS_DIR, f"{sym}_bearish.csv")
        if not os.path.isfile(swing_path):
            continue
        if mode == "all":
            if not os.path.isfile(spread_path) or not os.path.isfile(bearish_path):
                continue
        elif mode == "bullish":
            if not os.path.isfile(spread_path):
                continue
        elif mode == "bearish":
            if not os.path.isfile(bearish_path):
                continue
        else:
            continue
        try:
            df = pd.read_csv(swing_path)
            df.columns = [c.lower() for c in df.columns]
            col_map = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col.capitalize()] = df[col]
            if "date" in df.columns:
                df["Date"] = pd.to_datetime(df["date"])
            elif "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            else:
                df["Date"] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index("Date").sort_index()
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df = df[df.index >= pd.to_datetime(min_date)]

            def _read_marks_csv(path: str) -> pd.DataFrame:
                mdf = pd.read_csv(path)
                mdf.columns = [c.lower() for c in mdf.columns]
                if "date" in mdf.columns:
                    mdf["Date"] = pd.to_datetime(mdf["date"])
                else:
                    mdf["Date"] = pd.to_datetime(mdf["Date"])
                mdf = mdf.set_index("Date").sort_index()
                mdf.index = pd.to_datetime(mdf.index).tz_localize(None).normalize()
                return mdf

            if mode == "all":
                spread_marks = _read_marks_csv(spread_path)
                bear_marks = _read_marks_csv(bearish_path)
                if "spread_mark" not in spread_marks.columns or "pd_mark" not in bear_marks.columns:
                    continue
                marks = spread_marks.join(bear_marks, how="outer", rsuffix="_bear")
            elif mode == "bullish":
                marks = _read_marks_csv(spread_path)
                if "spread_mark" not in marks.columns:
                    continue
            else:
                marks = _read_marks_csv(bearish_path)
                if "pd_mark" not in marks.columns:
                    continue

            need_cols = [c for spec in specs for c in spec[:3]]
            have = [c for c in need_cols if c in marks.columns]
            df = df.join(marks[have], how="left")
            for mcol, pcol, tcol, _ in specs:
                if mcol not in df.columns:
                    df[mcol] = 0.0
                if pcol not in df.columns:
                    df[pcol] = 0.0
                if tcol not in df.columns:
                    df[tcol] = 0.0
                df[tcol] = df[tcol].fillna(0).astype(np.float32)
                df[mcol] = df[mcol].fillna(0).astype(np.float32)
                df[pcol] = df[pcol].fillna(0).astype(np.float32)

            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan)
            feat_drop = [c for c in feature_cols if c in df.columns]
            df = df.dropna(subset=feat_drop)
            if len(df) < 200:
                continue

            valid_cols = [c for c in feature_cols if c in df.columns]
            if len(valid_cols) < len(feature_cols):
                continue

            processed.append((sym, df, valid_cols))
        except Exception:
            continue

    if not processed:
        return None

    max_len = max(len(df) for _, df, _ in processed)
    n_strat = len(specs)
    is_credit = torch.BoolTensor([s[3] for s in specs])
    data_list, price_list, mark_list, prem_list, tradable_list, tickers = [], [], [], [], [], []

    for sym, df, valid_cols in processed:
        features = df[valid_cols].values
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        norm_features = np.clip((features - mean) / std, -10, 10)
        norm_features = np.nan_to_num(norm_features, nan=0.0)

        raw_close = df["Close"].values.astype(np.float32)
        marks = np.zeros((len(df), n_strat), dtype=np.float32)
        prems = np.zeros((len(df), n_strat), dtype=np.float32)
        trad = np.zeros((len(df), n_strat), dtype=np.bool_)
        for si, (mcol, pcol, tcol, _) in enumerate(specs):
            marks[:, si] = df[mcol].values.astype(np.float32)
            prems[:, si] = df[pcol].values.astype(np.float32)
            trad[:, si] = (df[tcol].values > 0.5)

        pad = max_len - len(norm_features)
        if pad > 0:
            norm_features = np.pad(norm_features, ((pad, 0), (0, 0)), mode="edge")
            raw_close = np.pad(raw_close, (pad, 0), mode="edge")
            marks = np.pad(marks, ((pad, 0), (0, 0)), mode="constant", constant_values=0)
            prems = np.pad(prems, ((pad, 0), (0, 0)), mode="constant", constant_values=0)
            trad = np.pad(trad, ((pad, 0), (0, 0)), mode="constant", constant_values=False)

        data_list.append(norm_features)
        price_list.append(raw_close)
        mark_list.append(marks)
        prem_list.append(prems)
        tradable_list.append(trad)
        tickers.append(sym)

    if not data_list:
        return None

    return (
        torch.FloatTensor(np.array(data_list)),
        torch.FloatTensor(np.array(price_list)),
        torch.FloatTensor(np.array(mark_list)),
        torch.FloatTensor(np.array(prem_list)),
        torch.BoolTensor(np.array(tradable_list)),
        is_credit,
        tickers,
    )


def load_options_training_tensors(
    watchlist_path: str,
    *,
    min_date: date | None = None,
) -> (
    tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.BoolTensor,
        list[str],
    ]
    | None
):
    """Legacy loader: call-debit only (first strategy column)."""
    loaded = load_options_multi_training_tensors(
        watchlist_path, mode="bullish", min_date=min_date,
    )
    if loaded is None:
        return None
    data, prices, marks, prems, tradable, _is_credit, tickers = loaded
    return (
        data,
        prices,
        marks[:, :, 0],
        prems[:, :, 0],
        tradable[:, :, 0],
        tickers,
    )
