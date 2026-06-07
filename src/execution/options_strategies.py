"""Option spread kinds, strike selection, and spread descriptors (live + backtest)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Optional


class SpreadKind(str, Enum):
    CALL_DEBIT = "call_debit"
    BULL_PUT_CREDIT = "bull_put_credit"
    PUT_DEBIT = "put_debit"
    BEAR_CALL_CREDIT = "bear_call_credit"
    LONG_CALL = "long_call"


@dataclass
class OptionSpread:
    underlying: str
    kind: SpreadKind
    long_symbol: str
    short_symbol: str  # empty for LONG_CALL
    expiration: date
    long_strike: float
    short_strike: float
    est_premium: float  # debit paid (+) or credit received (stored as +credit; is_credit=True)
    is_credit: bool = False

    @property
    def est_debit(self) -> float:
        """Backward-compatible alias (debit spreads only)."""
        return self.est_premium

    @property
    def label(self) -> str:
        if self.kind == SpreadKind.LONG_CALL:
            return f"LONG CALL {self.long_strike}"
        if self.kind == SpreadKind.CALL_DEBIT:
            return f"CALL DEBIT {self.long_strike}/{self.short_strike}"
        if self.kind == SpreadKind.BULL_PUT_CREDIT:
            return f"BULL PUT CREDIT {self.short_strike}/{self.long_strike}"
        if self.kind == SpreadKind.PUT_DEBIT:
            return f"PUT DEBIT {self.long_strike}/{self.short_strike}"
        if self.kind == SpreadKind.BEAR_CALL_CREDIT:
            return f"BEAR CALL CREDIT {self.short_strike}/{self.long_strike}"
        return self.kind.value


# Backward-compatible alias
CallDebitSpread = OptionSpread


def widths_to_try(spot: float, base_width: float, *, scale: bool) -> list[float]:
    from src.execution.options_spread import spread_width_for_spot

    primary = spread_width_for_spot(spot, base_width, scale=scale)
    widths: list[float] = []
    for w in (primary, 5.0, 2.5, 1.0):
        if w > 0 and w not in widths:
            widths.append(w)
    return widths


def _legs_call_debit(strikes: list[float], spot: float, width: float) -> tuple[float, float] | None:
    long_strike = min(strikes, key=lambda k: abs(k - spot))
    short_candidates = [k for k in strikes if k >= long_strike + width - 0.01]
    if not short_candidates:
        return None
    return long_strike, min(short_candidates)


def _legs_bull_put_credit(strikes: list[float], spot: float, width: float) -> tuple[float, float] | None:
    """Return (long_lower, short_higher)."""
    below = [k for k in strikes if k <= spot + 0.01]
    if not below:
        return None
    short_strike = max(below)
    long_candidates = [k for k in strikes if k <= short_strike - width + 0.01]
    if not long_candidates:
        return None
    long_strike = max(long_candidates)
    if long_strike >= short_strike:
        return None
    return long_strike, short_strike


def _legs_put_debit_bear(strikes: list[float], spot: float, width: float) -> tuple[float, float] | None:
    """Return (long_higher, short_lower)."""
    long_strike = min(strikes, key=lambda k: abs(k - spot))
    short_candidates = [k for k in strikes if k <= long_strike - width + 0.01]
    if not short_candidates:
        return None
    short_strike = max(short_candidates)
    if short_strike >= long_strike:
        return None
    return long_strike, short_strike


def _legs_bear_call_credit(strikes: list[float], spot: float, width: float) -> tuple[float, float] | None:
    """Return (long_higher, short_lower) — short is lower strike call sold."""
    above = [k for k in strikes if k >= spot - 0.01]
    if not above:
        return None
    short_strike = min(above)
    long_candidates = [k for k in strikes if k >= short_strike + width - 0.01]
    if not long_candidates:
        return None
    long_strike = min(long_candidates)
    if long_strike <= short_strike:
        return None
    return long_strike, short_strike


def _legs_long_call(strikes: list[float], spot: float) -> tuple[float, float] | None:
    strike = min(strikes, key=lambda k: abs(k - spot))
    return strike, strike


_LEG_FN = {
    SpreadKind.CALL_DEBIT: lambda s, spot, w: _legs_call_debit(s, spot, w),
    SpreadKind.BULL_PUT_CREDIT: lambda s, spot, w: _legs_bull_put_credit(s, spot, w),
    SpreadKind.PUT_DEBIT: lambda s, spot, w: _legs_put_debit_bear(s, spot, w),
    SpreadKind.BEAR_CALL_CREDIT: lambda s, spot, w: _legs_bear_call_credit(s, spot, w),
    SpreadKind.LONG_CALL: lambda s, spot, w: _legs_long_call(s, spot),
}


def match_all_on_chain(
    chain: list,
    spot: float,
    kind: SpreadKind,
    *,
    base_width: float,
    scale_width: bool,
    min_open_interest: int,
) -> list[tuple[tuple[float, float], float, dict[float, Any], dict[float, Any]]]:

    def _oi(c) -> float:
        try:
            return float(getattr(c, "open_interest", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    liquid = [c for c in chain if _oi(c) >= min_open_interest]
    full_by_strike = {float(c.strike_price): c for c in chain}
    if len(full_by_strike) < 1:
        return []
    if kind != SpreadKind.LONG_CALL and len(full_by_strike) < 2:
        return []

    leg_fn = _LEG_FN[kind]
    width_list = [0.0] if kind == SpreadKind.LONG_CALL else widths_to_try(
        spot, base_width, scale=scale_width
    )
    out: list[tuple[tuple[float, float], float, dict[float, Any], dict[float, Any]]] = []
    seen: set[tuple[float, float]] = set()

    for use_liquid in (True, False):
        if use_liquid and liquid:
            by_strike = {float(c.strike_price): c for c in liquid}
        else:
            by_strike = full_by_strike
        strikes = sorted(by_strike.keys())
        if len(strikes) < (1 if kind == SpreadKind.LONG_CALL else 2):
            continue
        for width in width_list:
            legs = leg_fn(strikes, spot, width)
            if legs and legs not in seen:
                seen.add(legs)
                used_w = 0.0 if kind == SpreadKind.LONG_CALL else (
                    legs[1] - legs[0] if kind in (SpreadKind.CALL_DEBIT, SpreadKind.BEAR_CALL_CREDIT)
                    else legs[0] - legs[1] if kind in (SpreadKind.BULL_PUT_CREDIT, SpreadKind.PUT_DEBIT)
                    else width
                )
                if kind == SpreadKind.PUT_DEBIT:
                    used_w = legs[0] - legs[1]
                elif kind == SpreadKind.BULL_PUT_CREDIT:
                    used_w = legs[1] - legs[0]
                out.append((legs, abs(used_w) or width, by_strike, full_by_strike))
        if out:
            break
    return out


def build_spread(
    kind: SpreadKind,
    u: str,
    expiration: date,
    legs: tuple[float, float],
    by_strike: dict[float, Any],
    full_by_strike: dict[float, Any],
) -> Optional[OptionSpread]:
    long_strike, short_strike = legs
    if kind == SpreadKind.LONG_CALL:
        c = by_strike.get(long_strike) or full_by_strike.get(long_strike)
        if not c:
            return None
        px = float(c.close_price or 0) or 0.05
        return OptionSpread(
            underlying=u,
            kind=kind,
            long_symbol=str(c.symbol),
            short_symbol="",
            expiration=expiration,
            long_strike=long_strike,
            short_strike=long_strike,
            est_premium=px,
            is_credit=False,
        )

    long_c = by_strike.get(long_strike) or full_by_strike.get(long_strike)
    short_c = by_strike.get(short_strike) or full_by_strike.get(short_strike)
    if not long_c or not short_c:
        return None
    long_px = float(long_c.close_price or 0) or 0.05
    short_px = float(short_c.close_price or 0) or 0.05

    if kind in (SpreadKind.CALL_DEBIT, SpreadKind.PUT_DEBIT):
        est = max(0.05, long_px - short_px)
        return OptionSpread(
            underlying=u, kind=kind,
            long_symbol=str(long_c.symbol), short_symbol=str(short_c.symbol),
            expiration=expiration, long_strike=long_strike, short_strike=short_strike,
            est_premium=est, is_credit=False,
        )
    # credit spreads: sell short_c, buy long_c
    est = max(0.05, short_px - long_px)
    return OptionSpread(
        underlying=u, kind=kind,
        long_symbol=str(long_c.symbol), short_symbol=str(short_c.symbol),
        expiration=expiration, long_strike=long_strike, short_strike=short_strike,
        est_premium=est, is_credit=True,
    )
