"""
Alpaca call debit spreads driven by swing signals (paper only).

Uses alpaca-py TradingClient + option contracts API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    ContractType,
    OrderClass,
    OrderSide,
    OrderType,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
    OptionLegRequest,
)

_OCC_RE = re.compile(r"^([A-Z]{1,5})(\d{6})([CP])(\d{8})$")


@dataclass
class CallDebitSpread:
    underlying: str
    long_symbol: str
    short_symbol: str
    expiration: date
    long_strike: float
    short_strike: float
    est_debit: float  # per-share net premium (limit_price for mleg)


def parse_occ_symbol(occ: str) -> Optional[tuple[str, date, str, float]]:
    """Return (underlying, expiration, right, strike) from OCC symbol."""
    m = _OCC_RE.match((occ or "").strip().upper())
    if not m:
        return None
    root, yymmdd, right, strike_raw = m.groups()
    exp = datetime.strptime(yymmdd, "%y%m%d").date()
    strike = int(strike_raw) / 1000.0
    return root, exp, right, strike


def _paper_only_guard(base_url: str) -> None:
    if "paper" not in (base_url or "").lower():
        raise RuntimeError(
            "Options bot is paper-only. Set ALPACA_BASE_URL=https://paper-api.alpaca.markets"
        )


class OptionsSpreadBroker:
    """Open/close call debit spreads on Alpaca (multi-leg)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str,
        target_dte: int = 30,
        min_dte: int = 14,
        max_dte: int = 45,
        spread_width: float = 5.0,
        limit_slippage_pct: float = 0.08,
    ):
        _paper_only_guard(base_url)
        paper = True
        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.target_dte = int(target_dte)
        self.min_dte = int(min_dte)
        self.max_dte = int(max_dte)
        self.spread_width = float(spread_width)
        self.limit_slippage_pct = float(limit_slippage_pct)
        # underlying -> CallDebitSpread last opened (hint for close)
        self._spread_book: dict[str, CallDebitSpread] = {}

    def underlyings_with_open_spreads(self, positions: Iterable[Any]) -> set[str]:
        """Unique underlyings with at least one option leg (spread or single)."""
        out: set[str] = set()
        for pos in positions:
            sym = getattr(pos, "symbol", "") or ""
            ac = (getattr(pos, "asset_class", "") or "").lower()
            if "option" in ac or _OCC_RE.match(sym.upper()):
                parsed = parse_occ_symbol(sym)
                if parsed:
                    out.add(parsed[0])
        return out

    def spread_unrealized_pct(self, positions: Iterable[Any], underlying: str) -> float:
        """Aggregate unrealized P/L % for all option legs on this underlying."""
        u = underlying.upper()
        cost = 0.0
        mkt = 0.0
        for pos in positions:
            sym = (getattr(pos, "symbol", "") or "").upper()
            parsed = parse_occ_symbol(sym)
            if not parsed or parsed[0] != u:
                continue
            qty = abs(float(getattr(pos, "qty", 0) or 0))
            if qty <= 0:
                continue
            entry = float(getattr(pos, "avg_entry_price", 0) or 0)
            current = float(getattr(pos, "current_price", 0) or 0)
            cost += entry * qty * 100
            mkt += current * qty * 100
        if cost <= 0:
            return 0.0
        return (mkt - cost) / cost

    def days_to_expiry(self, positions: Iterable[Any], underlying: str) -> Optional[int]:
        u = underlying.upper()
        exp: Optional[date] = None
        for pos in positions:
            parsed = parse_occ_symbol(getattr(pos, "symbol", "") or "")
            if parsed and parsed[0] == u:
                exp = parsed[1]
                break
        if not exp:
            spread = self._spread_book.get(u)
            if spread:
                exp = spread.expiration
        if not exp:
            return None
        return (exp - date.today()).days

    def pick_call_debit_spread(self, underlying: str, spot: float) -> Optional[CallDebitSpread]:
        """Select ATM long call + higher strike short for a debit spread."""
        u = underlying.upper()
        today = date.today()
        exp_gte = today + timedelta(days=self.min_dte)
        exp_lte = today + timedelta(days=self.max_dte)

        req = GetOptionContractsRequest(
            underlying_symbols=[u],
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
            type=ContractType.CALL,
            status="active",
        )
        try:
            page = self.client.get_option_contracts(req)
        except Exception as e:
            print(f"   ⚠️ {u}: option chain fetch failed: {e}")
            return None

        contracts = list(getattr(page, "option_contracts", None) or [])
        if not contracts:
            print(f"   ⚠️ {u}: no call contracts in DTE window.")
            return None

        # Pick expiration closest to target DTE
        by_exp: dict[date, list] = {}
        for c in contracts:
            exp = c.expiration_date
            if isinstance(exp, datetime):
                exp = exp.date()
            by_exp.setdefault(exp, []).append(c)

        def dte_dist(exp: date) -> int:
            return abs((exp - today).days - self.target_dte)

        best_exp = min(by_exp.keys(), key=dte_dist)
        chain = by_exp[best_exp]
        strikes = sorted({float(c.strike_price) for c in chain})
        if not strikes:
            return None

        long_strike = min(strikes, key=lambda k: abs(k - spot))
        short_strike = long_strike + self.spread_width
        short_candidates = [k for k in strikes if k >= short_strike - 0.01]
        if not short_candidates:
            print(f"   ⚠️ {u}: no short strike >= {short_strike:.2f} on {best_exp}.")
            return None
        short_strike = min(short_candidates)

        long_c = next((c for c in chain if float(c.strike_price) == long_strike), None)
        short_c = next((c for c in chain if float(c.strike_price) == short_strike), None)
        if not long_c or not short_c:
            print(f"   ⚠️ {u}: could not match spread legs {long_strike}/{short_strike}.")
            return None

        long_px = float(long_c.close_price or 0)
        short_px = float(short_c.close_price or 0)
        if long_px <= 0:
            long_px = 0.05
        est_debit = max(0.05, long_px - short_px)

        spread = CallDebitSpread(
            underlying=u,
            long_symbol=str(long_c.symbol),
            short_symbol=str(short_c.symbol),
            expiration=best_exp,
            long_strike=long_strike,
            short_strike=short_strike,
            est_debit=est_debit,
        )
        return spread

    def _limit_debit(self, est_debit: float) -> float:
        slip = 1.0 + self.limit_slippage_pct
        return round(est_debit * slip, 2)

    def open_call_debit_spread(
        self,
        underlying: str,
        spot: float,
        max_debit_dollars: float,
    ) -> bool:
        spread = self.pick_call_debit_spread(underlying, spot)
        if not spread:
            return False

        limit_px = self._limit_debit(spread.est_debit)
        cost_est = limit_px * 100.0
        if cost_est > max_debit_dollars:
            print(
                f"   ⚠️ {spread.underlying}: spread ~${cost_est:.0f} exceeds slot "
                f"${max_debit_dollars:.0f} — skip."
            )
            return False

        legs = [
            OptionLegRequest(
                symbol=spread.long_symbol,
                side=OrderSide.BUY,
                ratio_qty=1,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
            OptionLegRequest(
                symbol=spread.short_symbol,
                side=OrderSide.SELL,
                ratio_qty=1,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
        ]
        order = LimitOrderRequest(
            order_class=OrderClass.MLEG,
            qty=1,
            time_in_force=TimeInForce.DAY,
            type=OrderType.LIMIT,
            limit_price=limit_px,
            legs=legs,
        )
        print(
            f"   📈 {spread.underlying}: CALL DEBIT {spread.long_strike}/{spread.short_strike} "
            f"exp {spread.expiration} limit ${limit_px:.2f} (~${cost_est:.0f})"
        )
        try:
            self.client.submit_order(order)
            self._spread_book[spread.underlying] = spread
            print(f"   ✅ MLeg open submitted for {spread.underlying}")
            return True
        except Exception as e:
            print(f"   ❌ {spread.underlying}: open spread failed: {e}")
            return False

    def close_call_debit_spread(self, underlying: str) -> bool:
        u = underlying.upper()
        spread = self._spread_book.get(u)
        long_sym = short_sym = None
        if spread:
            long_sym, short_sym = spread.long_symbol, spread.short_symbol
        else:
            # Infer from open positions
            try:
                positions = self.client.get_all_positions()
            except Exception:
                positions = []
            long_legs = []
            short_legs = []
            for pos in positions:
                parsed = parse_occ_symbol(getattr(pos, "symbol", "") or "")
                if not parsed or parsed[0] != u or parsed[2] != "C":
                    continue
                qty = float(getattr(pos, "qty", 0) or 0)
                if qty > 0:
                    long_legs.append(pos.symbol)
                elif qty < 0:
                    short_legs.append(pos.symbol)
            if long_legs and short_legs:
                long_sym = long_legs[0]
                short_sym = short_legs[0]

        if not long_sym or not short_sym:
            print(f"   ⚠️ {u}: no spread legs found to close.")
            return False

        legs = [
            OptionLegRequest(
                symbol=long_sym,
                side=OrderSide.SELL,
                ratio_qty=1,
                position_intent=PositionIntent.SELL_TO_CLOSE,
            ),
            OptionLegRequest(
                symbol=short_sym,
                side=OrderSide.BUY,
                ratio_qty=1,
                position_intent=PositionIntent.BUY_TO_CLOSE,
            ),
        ]
        # Credit limit for closing — use small credit floor; broker may need net credit
        order = LimitOrderRequest(
            order_class=OrderClass.MLEG,
            qty=1,
            time_in_force=TimeInForce.DAY,
            type=OrderType.LIMIT,
            limit_price=0.01,
            legs=legs,
        )
        print(f"   🔻 {u}: CLOSE call debit spread ({long_sym} / {short_sym})")
        try:
            self.client.submit_order(order)
            self._spread_book.pop(u, None)
            print(f"   ✅ MLeg close submitted for {u}")
            return True
        except Exception as e:
            print(f"   ❌ {u}: close spread failed: {e}")
            return False
