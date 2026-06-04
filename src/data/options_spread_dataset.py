"""
Build and load call-debit spread marks from cached Alpaca option leg bars.

Files:
  data/historical_options/{OCC}.csv          — per-contract daily close
  data/historical_options_marks/{SYM}_spread.csv — aligned spread marks per underlying day
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

MARKS_DIR = os.path.join("data", "historical_options_marks")
OCC_DIR = os.path.join("data", "historical_options")
SWING_DIR = os.path.join("data", "historical_swing")


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


def _load_occ_closes(occ: str) -> dict[date, float]:
    path = os.path.join(OCC_DIR, f"{occ.upper()}.csv")
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    close_col = "close" if "close" in df.columns else None
    if close_col is None:
        return {}
    out: dict[date, float] = {}
    for _, row in df.iterrows():
        d = pd.to_datetime(row[date_col]).date()
        out[d] = float(row[close_col])
    return out


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
    occ_set, day_plans = plan_occ_symbols_for_symbol(symbol, start=start, end=end)
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
    marks_df = build_spread_marks_csv(symbol, day_plans)
    marks_df.to_csv(os.path.join(MARKS_DIR, f"{symbol.upper()}_spread.csv"), index=False)
    return saved


def rebuild_spread_marks(watchlist_path: str, *, start: date | None = None, end: date | None = None) -> None:
    """Rebuild *_spread.csv from cached OCC files (no API)."""
    symbols = load_watchlist(watchlist_path)
    os.makedirs(MARKS_DIR, exist_ok=True)
    for sym in tqdm(symbols, desc="Rebuild marks"):
        _, day_plans = plan_occ_symbols_for_symbol(sym, start=start, end=end)
        if not day_plans:
            continue
        build_spread_marks_csv(sym, day_plans).to_csv(
            os.path.join(MARKS_DIR, f"{sym.upper()}_spread.csv"), index=False
        )
    print(f"Rebuilt spread marks in {MARKS_DIR}")


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
    """Load swing features + spread marks tensors (same padding as swing training)."""
    from src.core.indicators import add_technical_indicators

    min_date = min_date or OPTIONS_DATA_MIN_DATE
    symbols = load_watchlist(watchlist_path)
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "rsi", "macd", "macd_signal", "adx", "sma_20", "ema_12",
    ]

    processed = []
    for sym in symbols:
        swing_path = os.path.join(SWING_DIR, f"{sym}_1D.csv")
        marks_path = os.path.join(MARKS_DIR, f"{sym}_spread.csv")
        if not os.path.isfile(swing_path) or not os.path.isfile(marks_path):
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

            marks = pd.read_csv(marks_path)
            marks.columns = [c.lower() for c in marks.columns]
            if "date" in marks.columns:
                marks["Date"] = pd.to_datetime(marks["date"])
            else:
                marks["Date"] = pd.to_datetime(marks["Date"])
            marks = marks.set_index("Date").sort_index()
            marks.index = pd.to_datetime(marks.index).tz_localize(None).normalize()
            mark_cols = [c for c in ("spread_mark", "entry_premium", "tradable") if c in marks.columns]
            if len(mark_cols) < 3:
                alt = {c.lower(): c for c in marks.columns}
                mark_cols = [alt.get(x, x) for x in ("spread_mark", "entry_premium", "tradable")]
            df = df.join(marks[mark_cols], how="left")
            if "tradable" not in df.columns and "Tradable" in df.columns:
                df["tradable"] = df["Tradable"]
            df["tradable"] = df["tradable"].fillna(0).astype(np.float32)
            df["spread_mark"] = df["spread_mark"].fillna(0).astype(np.float32)
            df["entry_premium"] = df["entry_premium"].fillna(0).astype(np.float32)

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
    data_list, price_list, mark_list, prem_list, tradable_list, tickers = [], [], [], [], [], []

    for sym, df, valid_cols in processed:
        features = df[valid_cols].values
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        norm_features = np.clip((features - mean) / std, -10, 10)
        norm_features = np.nan_to_num(norm_features, nan=0.0)

        raw_close = df["Close"].values.astype(np.float32)
        spread_mark = df["spread_mark"].values.astype(np.float32)
        entry_premium = df["entry_premium"].values.astype(np.float32)
        tradable = (df["tradable"].values > 0.5).astype(np.bool_)

        pad = max_len - len(norm_features)
        if pad > 0:
            norm_features = np.pad(norm_features, ((pad, 0), (0, 0)), mode="edge")
            raw_close = np.pad(raw_close, (pad, 0), mode="edge")
            spread_mark = np.pad(spread_mark, (pad, 0), mode="constant", constant_values=0)
            entry_premium = np.pad(entry_premium, (pad, 0), mode="constant", constant_values=0)
            tradable = np.pad(tradable, (pad, 0), mode="constant", constant_values=False)

        data_list.append(norm_features)
        price_list.append(raw_close)
        mark_list.append(spread_mark)
        prem_list.append(entry_premium)
        tradable_list.append(tradable)
        tickers.append(sym)

    if not data_list:
        return None

    return (
        torch.FloatTensor(np.array(data_list)),
        torch.FloatTensor(np.array(price_list)),
        torch.FloatTensor(np.array(mark_list)),
        torch.FloatTensor(np.array(prem_list)),
        torch.BoolTensor(np.array(tradable_list)),
        tickers,
    )
