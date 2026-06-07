"""
On-demand Alpaca historical option bars for spread backtests.

Uses Market Data API (indicative feed by default). Coverage from ~Feb 2024.
No synthetic/intrinsic fallback — missing bars return None.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

# Alpaca historical options availability
OPTIONS_DATA_MIN_DATE = date(2024, 2, 1)
# Free tier: cannot query the latest ~15 minutes without OPRA / Algo Trader Plus
INDICATIVE_END_DELAY = timedelta(minutes=20)


def occ_root_for_alpaca(underlying: str) -> str:
    """Alpaca option symbols: 1-5 letter root only (no dots), e.g. BRK.B -> BRKB."""
    root = underlying.upper().strip().replace(".", "")
    root = "".join(c for c in root if c.isalpha())[:5]
    return root


def build_occ_symbol(underlying: str, exp: date, right: str, strike: float) -> str:
    """OCC symbol, e.g. AAPL250620C00110000."""
    root = occ_root_for_alpaca(underlying)
    if not root:
        raise ValueError(f"invalid underlying for OCC: {underlying!r}")
    yymmdd = exp.strftime("%y%m%d")
    r = right.upper()[0]
    strike_int = int(round(float(strike) * 1000))
    return f"{root}{yymmdd}{r}{strike_int:08d}"


def pick_expiration(
    entry: date,
    *,
    target_dte: int = 30,
    min_dte: int = 14,
    max_dte: int = 45,
) -> date:
    """Calendar expiration ~target DTE, snapped to Friday, clamped to [min_dte, max_dte]."""
    raw = entry + timedelta(days=int(target_dte))
    days_to_fri = (4 - raw.weekday()) % 7
    if days_to_fri > 3:
        days_to_fri -= 7
    exp = raw + timedelta(days=days_to_fri)
    dte = (exp - entry).days
    if dte < min_dte:
        exp = entry + timedelta(days=min_dte)
        days_to_fri = (4 - exp.weekday()) % 7
        if days_to_fri > 3:
            days_to_fri -= 7
        exp = exp + timedelta(days=days_to_fri)
    elif dte > max_dte:
        exp = entry + timedelta(days=max_dte)
        days_to_fri = (4 - exp.weekday()) % 7
        if days_to_fri > 3:
            days_to_fri -= 7
        exp = exp + timedelta(days=days_to_fri)
    return exp


def load_swing_calendar(
    T: int,
    *,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    anchor_symbol: str = "SPY",
) -> list[date]:
    """Bar index -> calendar date (SPY timeline, same front-pad as load_swing_data)."""
    path = os.path.join("data", "historical_swing", f"{anchor_symbol}_1D.csv")
    if not os.path.isfile(path):
        end = date.today()
        start = end - timedelta(days=max(T * 2, 400))
        idx = pd.bdate_range(start=start, end=end)
        if len(idx) < T:
            pad = [idx[0]] * (T - len(idx))
            idx = pd.DatetimeIndex(list(pad) + list(idx))
        else:
            idx = idx[-T:]
        return [d.date() for d in idx]

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df["Date"] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index("Date").sort_index()
    if test_start_date:
        df = df[df.index >= pd.to_datetime(test_start_date)]
    if test_end_date:
        df = df[df.index <= pd.to_datetime(test_end_date)]
    dates = list(df.index)
    if len(dates) < T:
        pad = [dates[0]] * (T - len(dates))
        dates = pad + dates
    elif len(dates) > T:
        dates = dates[-T:]
    return [pd.Timestamp(d).date() for d in dates]


class OptionSpreadBarCache:
    """Fetch and cache daily option closes; spread mark = (long - short) * 100."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "indicative",
        request_pause_sec: float = 0.15,
    ):
        self._client = OptionHistoricalDataClient(api_key, secret_key)
        self.feed = feed
        self.pause = request_pause_sec
        self._closes: dict[str, dict[date, float]] = {}
        self._fetched_span: dict[str, tuple[date, date]] = {}
        self._no_spread_on: set[tuple[str, str, date]] = set()
        self.api_calls = 0
        self.skipped_no_bars = 0
        self.fetch_errors = 0
        self._logged_feed_note = False

    def _safe_end_datetime(self, end: date) -> datetime:
        """Avoid 'recent OPRA' 403 without ?feed= (not supported on /options/bars)."""
        utc = timezone.utc
        now = datetime.now(utc)
        today = now.date()
        cutoff = now - INDICATIVE_END_DELAY
        # Cap anything that could include today's session (incl. future exp+7 windows).
        if end >= today - timedelta(days=1):
            return cutoff
        return datetime.combine(end, datetime.max.time()).replace(tzinfo=utc)

    def _start_datetime(self, start: date) -> datetime:
        return datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)

    def _needs_fetch(self, sym: str, start: date, end: date) -> bool:
        sym = sym.upper()
        span = self._fetched_span.get(sym)
        if span is None:
            return True
        return span[0] > start or span[1] < end

    def _record_span(self, sym: str, start: date, end: date) -> None:
        sym = sym.upper()
        prev = self._fetched_span.get(sym)
        if prev is None:
            self._fetched_span[sym] = (start, end)
        else:
            self._fetched_span[sym] = (min(prev[0], start), max(prev[1], end))

    def _fetch_range(self, symbols: list[str], start: date, end: date) -> None:
        if not symbols:
            return
        start_dt = self._start_datetime(start)
        end_dt = self._safe_end_datetime(end)
        if not self._logged_feed_note:
            print(
                f"  Option bars: no feed param (/options/bars rejects it). "
                f"Recent ranges end at now-{INDICATIVE_END_DELAY} (free tier)."
            )
            self._logged_feed_note = True

        req = OptionBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
        )
        try:
            resp = self._client.get_option_bars(req)
            self.api_calls += 1
        except Exception as e:
            self.fetch_errors += 1
            if self.fetch_errors <= 3:
                print(f"   [warn] Option bars fetch failed ({symbols[0]}...): {e}")
            elif self.fetch_errors == 4:
                print("   [warn] Further option bar errors suppressed (see fetch_errors in summary).")
            # Do not record span — allow retry on later days / next run.
            return
        finally:
            time.sleep(self.pause)

        data = getattr(resp, "data", None) or resp
        if not isinstance(data, dict):
            for sym in symbols:
                self._record_span(sym, start, end)
            return
        for sym, rows in data.items():
            bucket = self._closes.setdefault(sym.upper(), {})
            for row in rows or []:
                ts = row.get("timestamp") if isinstance(row, dict) else getattr(row, "timestamp", None)
                close = row.get("close") if isinstance(row, dict) else getattr(row, "close", None)
                if ts is None or close is None:
                    continue
                d = ts.date() if isinstance(ts, datetime) else pd.Timestamp(ts).date()
                bucket[d] = float(close)
        for sym in symbols:
            self._record_span(sym, start, end)

    def ensure_symbols(self, symbols: list[str], start: date, end: date) -> None:
        """Batch-fetch up to 100 OCC symbols per API call."""
        uniq = []
        seen: set[str] = set()
        for s in symbols:
            u = s.upper()
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        need = [s for s in uniq if self._needs_fetch(s, start, end)]
        for i in range(0, len(need), 100):
            self._fetch_range(need[i : i + 100], start, end)

    def ensure_range(self, long_sym: str, short_sym: str, start: date, end: date) -> None:
        self.ensure_symbols([long_sym, short_sym], start, end)

    def spread_mark(self, long_sym: str, short_sym: str, on: date) -> Optional[float]:
        """Debit spread mark: (long − short) × 100 per contract."""
        return self.position_mark(long_sym, short_sym, on, is_credit=False)

    def position_mark(
        self,
        long_sym: str,
        short_sym: str,
        on: date,
        *,
        is_credit: bool = False,
        single_leg: bool = False,
    ) -> Optional[float]:
        """Per-contract $ mark; credit spreads use (short − long) × 100."""
        long_sym = long_sym.upper()
        if single_leg or not short_sym:
            lc = self._closes.get(long_sym, {}).get(on)
            if lc is None:
                return None
            return max(0.0, lc * 100.0)
        short_sym = short_sym.upper()
        key = (long_sym, short_sym, on)
        if key in self._no_spread_on:
            return None
        lc = self._closes.get(long_sym, {}).get(on)
        sc = self._closes.get(short_sym, {}).get(on)
        if lc is None or sc is None:
            return None
        if is_credit:
            return max(0.0, (sc - lc) * 100.0)
        return max(0.0, (lc - sc) * 100.0)

    def remember_no_spread(self, long_sym: str, short_sym: str, on: date) -> None:
        self._no_spread_on.add((long_sym.upper(), short_sym.upper(), on))


class DiskOptionSpreadBarCache:
    """Spread marks from data/historical_options/{OCC}.csv (no API; use after download_options_bars)."""

    def __init__(self) -> None:
        self._closes: dict[str, dict[date, float]] = {}
        self._no_spread_on: set[tuple[str, str, date]] = set()
        self.api_calls = 0
        self.skipped_no_bars = 0
        self.fetch_errors = 0

    def _leg_closes(self, occ: str) -> dict[date, float]:
        occ = occ.upper()
        if occ not in self._closes:
            from src.data.options_spread_dataset import _load_occ_closes

            self._closes[occ] = _load_occ_closes(occ)
        return self._closes[occ]

    def ensure_symbols(self, symbols: list[str], start: date, end: date) -> None:
        for sym in symbols:
            self._leg_closes(sym)

    def ensure_range(self, long_sym: str, short_sym: str, start: date, end: date) -> None:
        self.ensure_symbols([long_sym, short_sym], start, end)

    def spread_mark(self, long_sym: str, short_sym: str, on: date) -> Optional[float]:
        return self.position_mark(long_sym, short_sym, on, is_credit=False)

    def position_mark(
        self,
        long_sym: str,
        short_sym: str,
        on: date,
        *,
        is_credit: bool = False,
        single_leg: bool = False,
    ) -> Optional[float]:
        long_sym = long_sym.upper()
        if single_leg or not short_sym:
            lc = self._leg_closes(long_sym).get(on)
            if lc is None:
                return None
            return max(0.0, lc * 100.0)
        short_sym = short_sym.upper()
        key = (long_sym, short_sym, on)
        if key in self._no_spread_on:
            return None
        lc = self._leg_closes(long_sym).get(on)
        sc = self._leg_closes(short_sym).get(on)
        if lc is None or sc is None:
            return None
        if is_credit:
            return max(0.0, (sc - lc) * 100.0)
        return max(0.0, (lc - sc) * 100.0)

    def remember_no_spread(self, long_sym: str, short_sym: str, on: date) -> None:
        self._no_spread_on.add((long_sym.upper(), short_sym.upper(), on))


def make_option_bar_cache_from_disk() -> DiskOptionSpreadBarCache:
    return DiskOptionSpreadBarCache()


def make_option_bar_cache_from_env(*, feed: str = "indicative") -> OptionSpreadBarCache:
    from dotenv import load_dotenv

    load_dotenv()
    key = os.getenv("DAY_API_KEY") or os.getenv("SWING_API_KEY")
    secret = os.getenv("DAY_API_SECRET") or os.getenv("SWING_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "Set DAY_API_KEY/DAY_API_SECRET (or SWING_*) for Alpaca option market data."
        )
    return OptionSpreadBarCache(key, secret, feed=feed)
