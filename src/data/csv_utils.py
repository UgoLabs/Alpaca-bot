"""Helpers for swing CSV files (yfinance -> local disk)."""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame or empty if unusable."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    col_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns={c: col_map.get(str(c).lower(), c) for c in df.columns})
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        return pd.DataFrame()

    out = df[OHLCV_COLS].copy()
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out.index.name = "Date"
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    return out


def read_swing_csv(path: str) -> pd.DataFrame:
    """Load a swing CSV, recovering from legacy duplicate-column corruption."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Legacy files stored duplicate Price columns (Close + Close.1 from racey yf.download).
    if "Close.1" in df.columns:
        close = df["Close"].copy()
        if close.isna().all() or (close.notna() & df["Close.1"].notna()).any():
            # Prefer .1 when primary Close is NaN or both exist (primary is often wrong ticker).
            mask = close.isna() & df["Close.1"].notna()
            close = close.where(~mask, df["Close.1"])
        for base in ("Open", "High", "Low", "Volume"):
            alt = f"{base}.1"
            if alt not in df.columns:
                continue
            col = df[base].copy()
            m = col.isna() & df[alt].notna()
            df[base] = col.where(~m, df[alt])
        df["Close"] = close

    return normalize_ohlcv(df)


def csv_header_col_count(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as f:
        return len(f.readline().strip().split(","))


def is_corrupt_csv(path: str, reference_close: Optional[float] = None) -> bool:
    """True if file structure or (optional) last close looks wrong."""
    if not os.path.exists(path):
        return True
    if csv_header_col_count(path) > 7:
        return True
    df = read_swing_csv(path)
    if df.empty or len(df) < 10:
        return True
    last = float(df["Close"].iloc[-1])
    if last <= 0:
        return True
    if reference_close is not None and reference_close > 0:
        ratio = last / reference_close
        if ratio < 0.85 or ratio > 1.15:
            return True
    return False


def save_swing_csv(df: pd.DataFrame, path: str) -> None:
    out = normalize_ohlcv(df)
    if out.empty:
        raise ValueError("Refusing to save empty/invalid OHLCV frame")
    out.to_csv(path)
