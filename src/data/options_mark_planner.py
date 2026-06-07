"""Strike/expiry planning for options marks (training, backtest, download)."""
from __future__ import annotations

from datetime import date

from src.data.options_historical import build_occ_symbol, pick_expiration
from src.execution.options_spread import spread_width_for_spot
from src.execution.options_strategies import SpreadKind, _LEG_FN, widths_to_try


def _strike_step(spot: float, width: float) -> float:
    return width if spot >= 200 else (2.5 if spot < 100 else width)


def _strike_ladder(spot: float, width: float) -> list[float]:
    step = _strike_step(spot, width)
    center = round(spot / step) * step
    if center <= 0:
        center = max(step, spot)
    return [float(center + i * step) for i in range(-8, 9)]


def plan_spread(
    sym: str,
    spot: float,
    on: date,
    kind: SpreadKind,
    *,
    spread_width: float,
    scale_width: bool,
    target_dte: int,
    min_dte: int,
    max_dte: int,
) -> dict | None:
    width_list = [0.0] if kind == SpreadKind.LONG_CALL else widths_to_try(
        spot, spread_width, scale=scale_width
    )
    leg_fn = _LEG_FN[kind]
    exp = pick_expiration(on, target_dte=target_dte, min_dte=min_dte, max_dte=max_dte)
    right = "C" if kind in (
        SpreadKind.CALL_DEBIT, SpreadKind.BEAR_CALL_CREDIT, SpreadKind.LONG_CALL,
    ) else "P"

    for width in width_list:
        ladder = _strike_ladder(spot, width or spread_width)
        legs = leg_fn(ladder, spot, width)
        if not legs:
            continue
        long_k, short_k = legs
        long_sym = build_occ_symbol(sym, exp, right, long_k)
        if kind == SpreadKind.LONG_CALL:
            return {
                "kind": kind.value,
                "long_sym": long_sym,
                "short_sym": "",
                "long_k": long_k,
                "short_k": short_k,
                "exp": exp,
                "is_credit": False,
                "single_leg": True,
            }
        short_sym = build_occ_symbol(sym, exp, right, short_k)
        is_credit = kind in (SpreadKind.BULL_PUT_CREDIT, SpreadKind.BEAR_CALL_CREDIT)
        return {
            "kind": kind.value,
            "long_sym": long_sym,
            "short_sym": short_sym,
            "long_k": long_k,
            "short_k": short_k,
            "exp": exp,
            "is_credit": is_credit,
            "single_leg": False,
        }
    return None


BULLISH_KINDS = (
    SpreadKind.CALL_DEBIT,
    SpreadKind.BULL_PUT_CREDIT,
    SpreadKind.LONG_CALL,
)

BEARISH_KINDS = (
    SpreadKind.PUT_DEBIT,
    SpreadKind.BEAR_CALL_CREDIT,
)
