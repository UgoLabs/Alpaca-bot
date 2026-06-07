"""
Alpaca option spreads driven by swing signals (paper only).

Supports call debit, bull put credit, put debit, bear call credit, and long call.

Uses alpaca-py TradingClient + option contracts API.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from src.execution.options_strategies import (
    CallDebitSpread,
    OptionSpread,
    SpreadKind,
    build_spread,
    match_all_on_chain,
    widths_to_try,
)

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    ContractType,
    OrderClass,
    OrderSide,
    OrderType,
    PositionIntent,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OptionLegRequest,
)

_OCC_RE = re.compile(r"^([A-Z]{1,5})(\d{6})([CP])(\d{8})$")


def spread_width_for_spot(spot: float, base_width: float = 5.0, *, scale: bool = True) -> float:
    """Strike width ($) scaled by underlying price so width stays ~1-2% of spot.

    Shared by the live broker and the backtest to keep strike selection in parity.
    """
    if not scale:
        return float(base_width)
    spot = float(spot)
    if spot < 100.0:
        return min(2.5, base_width)
    if spot > 250.0:
        return base_width * 2.0
    return float(base_width)


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
    """Open/close option spreads on Alpaca (multi-leg + single-leg)."""

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
        scale_width_by_price: bool = True,
        min_open_interest: int = 10,
        max_contracts_per_slot: int = 10,
        close_use_market: bool = True,
    ):
        _paper_only_guard(base_url)
        paper = True
        self.client = TradingClient(api_key, api_secret, paper=paper)
        self._quote_client = OptionHistoricalDataClient(api_key, api_secret)
        self.target_dte = int(target_dte)
        self.min_dte = int(min_dte)
        self.max_dte = int(max_dte)
        self.spread_width = float(spread_width)
        self.limit_slippage_pct = float(limit_slippage_pct)
        self.scale_width_by_price = bool(scale_width_by_price)
        self.min_open_interest = int(min_open_interest)
        self.max_contracts_per_slot = max(1, int(max_contracts_per_slot))
        self.close_use_market = bool(close_use_market)
        self._spread_book: dict[str, OptionSpread] = {}

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

    def underlyings_with_pending_opens(self) -> set[str]:
        """Underlyings with an open (unfilled) MLEG buy-to-open spread order."""
        out: set[str] = set()
        for orders in self._pending_open_orders_by_underlying().values():
            if orders:
                out.add(self._underlying_from_order(orders[0]))
        return {u for u in out if u}

    def _underlying_from_order(self, order: Any) -> str:
        for leg in getattr(order, "legs", None) or []:
            sym = getattr(leg, "symbol", "") or ""
            parsed = parse_occ_symbol(sym)
            if parsed:
                return parsed[0]
        return ""

    def _pending_open_orders_by_underlying(self) -> dict[str, list[Any]]:
        """Open MLEG orders keyed by underlying (from OCC leg symbols)."""
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = list(self.client.get_orders(filter=req) or [])
        except Exception:
            return {}
        by_u: dict[str, list[Any]] = {}
        for order in orders:
            if str(getattr(order, "order_class", "") or "").lower() != "mleg":
                continue
            u = self._underlying_from_order(order)
            if not u:
                continue
            by_u.setdefault(u.upper(), []).append(order)
        return by_u

    def _cancel_pending_opens(self, underlying: str) -> int:
        u = underlying.upper()
        cancelled = 0
        for order in self._pending_open_orders_by_underlying().get(u, []):
            oid = getattr(order, "id", None)
            if not oid:
                continue
            try:
                self.client.cancel_order_by_id(oid)
                cancelled += 1
            except Exception as e:
                print(f"   ⚠️ {u}: cancel open order {oid} failed: {e}")
        return cancelled

    @staticmethod
    def _order_matches_spread(order: Any, spread: OptionSpread) -> bool:
        if spread.kind == SpreadKind.LONG_CALL:
            legs = getattr(order, "legs", None) or []
            if legs:
                return any(
                    (getattr(leg, "symbol", "") or "").upper() == spread.long_symbol.upper()
                    for leg in legs
                )
            sym = (getattr(order, "symbol", "") or "").upper()
            return sym == spread.long_symbol.upper()
        long_sym = short_sym = None
        for leg in getattr(order, "legs", None) or []:
            sym = (getattr(leg, "symbol", "") or "").upper()
            side = str(getattr(leg, "side", "") or "").lower()
            if side == "buy":
                long_sym = sym
            elif side == "sell":
                short_sym = sym
        return (
            long_sym == spread.long_symbol.upper()
            and short_sym == spread.short_symbol.upper()
        )

    def _spread_debit_estimate(
        self,
        long_sym: str,
        short_sym: str,
        fallback: float,
    ) -> tuple[float, str]:
        """Debit per share: prefer long ask − short bid; else mid; else fallback close."""
        try:
            resp = self._quote_client.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=[long_sym, short_sym])
            )
        except Exception:
            return max(0.05, fallback), "close"

        def _q(sym: str) -> Any:
            if hasattr(resp, "get"):
                return resp.get(sym)
            return getattr(resp, sym, None)

        long_q = _q(long_sym)
        short_q = _q(short_sym)
        if not long_q or not short_q:
            return max(0.05, fallback), "close"

        long_ask = float(getattr(long_q, "ask_price", 0) or 0)
        short_bid = float(getattr(short_q, "bid_price", 0) or 0)
        if long_ask > 0 and short_bid >= 0:
            return max(0.05, long_ask - short_bid), "quote"

        long_bid = float(getattr(long_q, "bid_price", 0) or 0)
        short_ask = float(getattr(short_q, "ask_price", 0) or 0)
        if long_ask > 0 and long_bid > 0 and short_ask > 0 and short_bid >= 0:
            long_mid = (long_ask + long_bid) / 2.0
            short_mid = (short_ask + short_bid) / 2.0
            return max(0.05, long_mid - short_mid), "mid"

        return max(0.05, fallback), "close"

    def _spread_credit_estimate(
        self,
        short_sym: str,
        long_sym: str,
        fallback: float,
    ) -> tuple[float, str]:
        """Credit per share: short bid − long ask."""
        try:
            resp = self._quote_client.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=[long_sym, short_sym])
            )
        except Exception:
            return max(0.05, fallback), "close"

        def _q(sym: str) -> Any:
            if hasattr(resp, "get"):
                return resp.get(sym)
            return getattr(resp, sym, None)

        short_q = _q(short_sym)
        long_q = _q(long_sym)
        if not short_q or not long_q:
            return max(0.05, fallback), "close"
        short_bid = float(getattr(short_q, "bid_price", 0) or 0)
        long_ask = float(getattr(long_q, "ask_price", 0) or 0)
        if short_bid > 0 and long_ask >= 0:
            return max(0.05, short_bid - long_ask), "quote"
        return max(0.05, fallback), "close"

    def spread_unrealized_pct(self, positions: Iterable[Any], underlying: str) -> float:
        """Net unrealized P/L % on the debit spread (signed qty: long +, short -)."""
        u = underlying.upper()
        net_cost = 0.0
        net_mkt = 0.0
        for pos in positions:
            sym = (getattr(pos, "symbol", "") or "").upper()
            parsed = parse_occ_symbol(sym)
            if not parsed or parsed[0] != u:
                continue
            qty = float(getattr(pos, "qty", 0) or 0)
            if qty == 0:
                continue
            entry = float(getattr(pos, "avg_entry_price", 0) or 0)
            current = float(getattr(pos, "current_price", 0) or 0)
            net_cost += entry * qty * 100.0
            net_mkt += current * qty * 100.0
        # Debit spread: net_cost is premium paid (positive). Credit books flip sign.
        basis = abs(net_cost)
        if basis <= 0:
            return 0.0
        return (net_mkt - net_cost) / basis

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

    def _widths_to_try(self, spot: float) -> list[float]:
        """Primary scaled width, then common narrower fallbacks for sparse strike grids."""
        primary = spread_width_for_spot(
            spot, self.spread_width, scale=self.scale_width_by_price
        )
        widths: list[float] = []
        for w in (primary, 5.0, 2.5, 1.0):
            if w > 0 and w not in widths:
                widths.append(w)
        return widths

    @staticmethod
    def _legs_for_width(
        strikes: list[float],
        spot: float,
        width: float,
    ) -> tuple[float, float] | None:
        long_strike = min(strikes, key=lambda k: abs(k - spot))
        short_target = long_strike + width
        short_candidates = [k for k in strikes if k >= short_target - 0.01]
        if not short_candidates:
            return None
        return long_strike, min(short_candidates)

    @staticmethod
    def _next_strike_up(strikes: list[float], spot: float) -> tuple[float, float] | None:
        """Minimum viable spread: long ATM + nearest higher listed strike."""
        long_strike = min(strikes, key=lambda k: abs(k - spot))
        higher = [k for k in strikes if k > long_strike + 0.01]
        if not higher:
            return None
        return long_strike, min(higher)

    def _fetch_call_contracts(
        self,
        underlying: str,
        exp_gte: date,
        exp_lte: date,
    ) -> list:
        """Paginate option contract search so sparse names get the full chain."""
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
            type=ContractType.CALL,
            status="active",
        )
        out: list = []
        page_token: str | None = None
        while True:
            if page_token:
                req = GetOptionContractsRequest(
                    underlying_symbols=[underlying],
                    expiration_date_gte=exp_gte,
                    expiration_date_lte=exp_lte,
                    type=ContractType.CALL,
                    status="active",
                    page_token=page_token,
                )
            page = self.client.get_option_contracts(req)
            batch = list(getattr(page, "option_contracts", None) or [])
            out.extend(batch)
            page_token = getattr(page, "next_page_token", None)
            if not page_token:
                break
        return out

    def _fetch_put_contracts(self, underlying: str, exp_gte: date, exp_lte: date) -> list:
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
            type=ContractType.PUT,
            status="active",
        )
        out: list = []
        page_token: str | None = None
        while True:
            if page_token:
                req = GetOptionContractsRequest(
                    underlying_symbols=[underlying],
                    expiration_date_gte=exp_gte,
                    expiration_date_lte=exp_lte,
                    type=ContractType.PUT,
                    status="active",
                    page_token=page_token,
                )
            page = self.client.get_option_contracts(req)
            batch = list(getattr(page, "option_contracts", None) or [])
            out.extend(batch)
            page_token = getattr(page, "next_page_token", None)
            if not page_token:
                break
        return out

    def _fetch_contracts_for_kind(
        self, underlying: str, kind: SpreadKind, exp_gte: date, exp_lte: date
    ) -> list:
        if kind in (SpreadKind.CALL_DEBIT, SpreadKind.BEAR_CALL_CREDIT, SpreadKind.LONG_CALL):
            return self._fetch_call_contracts(underlying, exp_gte, exp_lte)
        return self._fetch_put_contracts(underlying, exp_gte, exp_lte)

    def pick_spread_within_budget(
        self,
        underlying: str,
        spot: float,
        kind: SpreadKind,
        max_budget_dollars: float,
    ) -> Optional[tuple[OptionSpread, str]]:
        u = underlying.upper()
        today = date.today()
        exp_gte = today + timedelta(days=self.min_dte)
        exp_lte = today + timedelta(days=self.max_dte)
        try:
            contracts = self._fetch_contracts_for_kind(u, kind, exp_gte, exp_lte)
        except Exception as e:
            print(f"   ⚠️ {u}: option chain fetch failed ({kind.value}): {e}")
            return None
        if not contracts:
            print(f"   ⚠️ {u}: no {kind.value} contracts in DTE window.")
            return None

        by_exp: dict[date, list] = {}
        for c in contracts:
            exp = c.expiration_date
            if isinstance(exp, datetime):
                exp = exp.date()
            by_exp.setdefault(exp, []).append(c)

        def dte_dist(exp: date) -> int:
            return abs((exp - today).days - self.target_dte)

        width_rank = {w: i for i, w in enumerate(widths_to_try(
            spot, self.spread_width, scale=self.scale_width_by_price
        ))}
        candidates: list = []
        for exp in by_exp:
            dist = dte_dist(exp)
            for legs, width, by_strike, full_by_strike in match_all_on_chain(
                by_exp[exp], spot, kind,
                base_width=self.spread_width,
                scale_width=self.scale_width_by_price,
                min_open_interest=self.min_open_interest,
            ):
                wr = 0 if kind == SpreadKind.LONG_CALL else width_rank.get(width, 99)
                candidates.append((dist, wr, width, exp, legs, by_strike, full_by_strike))

        if not candidates:
            return None
        candidates.sort(key=lambda row: (row[0], row[1]))
        cheapest_cost = float("inf")
        cheapest_src = "close"
        cheapest_prem = 0.0

        for _dist, _wr, width, exp, legs, by_strike, full_by_strike in candidates:
            spread = build_spread(kind, u, exp, legs, by_strike, full_by_strike)
            if not spread:
                continue
            if spread.is_credit:
                prem, price_src = self._spread_credit_estimate(
                    spread.short_symbol, spread.long_symbol, spread.est_premium
                )
                spread.est_premium = prem
                width_dollars = abs(spread.short_strike - spread.long_strike)
                max_loss = max(50.0, (width_dollars - prem) * 100.0)
                contract_cost = max_loss
            elif spread.kind == SpreadKind.LONG_CALL:
                prem, price_src = self._spread_debit_estimate(
                    spread.long_symbol, spread.long_symbol, spread.est_premium
                )
                # single leg: use ask from quote on long only
                try:
                    resp = self._quote_client.get_option_latest_quote(
                        OptionLatestQuoteRequest(symbol_or_symbols=[spread.long_symbol])
                    )
                    q = resp.get(spread.long_symbol) if hasattr(resp, "get") else getattr(
                        resp, spread.long_symbol, None
                    )
                    if q and float(getattr(q, "ask_price", 0) or 0) > 0:
                        prem = float(q.ask_price)
                        price_src = "quote"
                except Exception:
                    pass
                spread.est_premium = prem
                limit_px = self._limit_debit(prem)
                contract_cost = limit_px * 100.0
            else:
                prem, price_src = self._spread_debit_estimate(
                    spread.long_symbol, spread.short_symbol, spread.est_premium
                )
                spread.est_premium = prem
                limit_px = self._limit_debit(prem)
                contract_cost = limit_px * 100.0

            cheapest_cost = min(cheapest_cost, contract_cost)
            cheapest_prem = prem
            cheapest_src = price_src
            if contract_cost <= max_budget_dollars:
                return spread, price_src

        print(
            f"   ⚠️ {u}: no {kind.value} fits slot ${max_budget_dollars:.0f} "
            f"({len(candidates)} tried; cheapest ~${cheapest_cost:.0f} "
            f"[{cheapest_src} ${cheapest_prem:.2f}]) — skip.",
        )
        return None

    def _match_spread_on_chain(
        self,
        chain: list,
        spot: float,
    ) -> tuple[tuple[float, float], float | None, dict[float, Any]] | None:
        """Return (legs, used_width, by_strike) or None."""

        def _oi(c) -> float:
            try:
                return float(getattr(c, "open_interest", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        liquid = [c for c in chain if _oi(c) >= self.min_open_interest]
        full_by_strike = {float(c.strike_price): c for c in chain}
        if len(full_by_strike) < 2:
            return None

        primary_w = spread_width_for_spot(
            spot, self.spread_width, scale=self.scale_width_by_price
        )
        legs: tuple[float, float] | None = None
        used_width: float | None = None
        by_strike = full_by_strike

        for use_liquid in (True, False):
            if use_liquid and liquid:
                by_strike = {float(c.strike_price): c for c in liquid}
            else:
                by_strike = full_by_strike
            strikes = sorted(by_strike.keys())
            if len(strikes) < 2:
                continue

            for width in self._widths_to_try(spot):
                legs = self._legs_for_width(strikes, spot, width)
                if legs:
                    used_width = width
                    break
            if legs:
                break

            legs = self._next_strike_up(strikes, spot)
            if legs:
                used_width = legs[1] - legs[0]
                break

        if not legs:
            return None
        return legs, used_width, by_strike

    def _match_all_spreads_on_chain(
        self,
        chain: list,
        spot: float,
    ) -> list[tuple[tuple[float, float], float, dict[float, Any], dict[float, Any]]]:
        """All viable (legs, width, by_strike, full_by_strike) pairs on one expiry."""

        def _oi(c) -> float:
            try:
                return float(getattr(c, "open_interest", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        liquid = [c for c in chain if _oi(c) >= self.min_open_interest]
        full_by_strike = {float(c.strike_price): c for c in chain}
        if len(full_by_strike) < 2:
            return []

        out: list[tuple[tuple[float, float], float, dict[float, Any], dict[float, Any]]] = []
        seen: set[tuple[float, float]] = set()

        for use_liquid in (True, False):
            if use_liquid and liquid:
                by_strike = {float(c.strike_price): c for c in liquid}
            else:
                by_strike = full_by_strike
            strikes = sorted(by_strike.keys())
            if len(strikes) < 2:
                continue

            for width in self._widths_to_try(spot):
                legs = self._legs_for_width(strikes, spot, width)
                if legs and legs not in seen:
                    seen.add(legs)
                    out.append((legs, width, by_strike, full_by_strike))

            legs = self._next_strike_up(strikes, spot)
            if legs and legs not in seen:
                seen.add(legs)
                out.append((legs, legs[1] - legs[0], by_strike, full_by_strike))

        return out

    def _build_call_debit_spread(
        self,
        u: str,
        expiration: date,
        long_strike: float,
        short_strike: float,
        by_strike: dict[float, Any],
        full_by_strike: dict[float, Any],
    ) -> Optional[CallDebitSpread]:
        long_c = by_strike.get(long_strike) or full_by_strike.get(long_strike)
        short_c = by_strike.get(short_strike) or full_by_strike.get(short_strike)
        if not long_c or not short_c:
            return None
        long_px = float(long_c.close_price or 0)
        short_px = float(short_c.close_price or 0)
        if long_px <= 0:
            long_px = 0.05
        est_debit = max(0.05, long_px - short_px)
        return OptionSpread(
            underlying=u,
            kind=SpreadKind.CALL_DEBIT,
            long_symbol=str(long_c.symbol),
            short_symbol=str(short_c.symbol),
            expiration=expiration,
            long_strike=long_strike,
            short_strike=short_strike,
            est_premium=est_debit,
        )

    def pick_call_debit_spread_within_budget(
        self,
        underlying: str,
        spot: float,
        max_debit_dollars: float,
    ) -> Optional[tuple[CallDebitSpread, str]]:
        """Pick a spread whose quote-based limit fits one contract in the slot budget."""
        u = underlying.upper()
        today = date.today()
        exp_gte = today + timedelta(days=self.min_dte)
        exp_lte = today + timedelta(days=self.max_dte)

        try:
            contracts = self._fetch_call_contracts(u, exp_gte, exp_lte)
        except Exception as e:
            print(f"   ⚠️ {u}: option chain fetch failed: {e}")
            return None

        if not contracts:
            print(f"   ⚠️ {u}: no call contracts in DTE window.")
            return None

        by_exp: dict[date, list] = {}
        for c in contracts:
            exp = c.expiration_date
            if isinstance(exp, datetime):
                exp = exp.date()
            by_exp.setdefault(exp, []).append(c)

        def dte_dist(exp: date) -> int:
            return abs((exp - today).days - self.target_dte)

        primary_w = spread_width_for_spot(
            spot, self.spread_width, scale=self.scale_width_by_price
        )
        width_rank = {w: i for i, w in enumerate(self._widths_to_try(spot))}

        candidates: list[
            tuple[int, int, float, date, tuple[float, float], dict[float, Any], dict[float, Any]]
        ] = []
        for exp in by_exp:
            dist = dte_dist(exp)
            for legs, width, by_strike, full_by_strike in self._match_all_spreads_on_chain(
                by_exp[exp], spot
            ):
                candidates.append(
                    (dist, width_rank.get(width, 99), width, exp, legs, by_strike, full_by_strike)
                )

        if not candidates:
            print(
                f"   ⚠️ {u}: no viable call spread in DTE window "
                f"({len(by_exp)} expiries; tried widths {self._widths_to_try(spot)} "
                f"+ next strike up).",
            )
            return None

        candidates.sort(key=lambda row: (row[0], row[1]))
        best_exp = min(by_exp.keys(), key=dte_dist)
        cheapest_cost = float("inf")
        cheapest_src = "close"
        cheapest_debit = 0.0

        for dist, _wr, width, exp, legs, by_strike, full_by_strike in candidates:
            long_strike, short_strike = legs
            spread = self._build_call_debit_spread(
                u, exp, long_strike, short_strike, by_strike, full_by_strike
            )
            if not spread:
                continue
            est_debit, price_src = self._spread_debit_estimate(
                spread.long_symbol, spread.short_symbol, spread.est_debit
            )
            limit_px = self._limit_debit(est_debit)
            contract_cost = limit_px * 100.0
            cheapest_cost = min(cheapest_cost, contract_cost)
            cheapest_debit = est_debit
            cheapest_src = price_src
            if contract_cost <= max_debit_dollars:
                if exp != best_exp:
                    print(
                        f"   ℹ️ {u}: using exp {exp} "
                        f"(target ~{self.target_dte}d was {best_exp})",
                    )
                if abs(width - primary_w) > 0.01:
                    print(
                        f"   ℹ️ {u}: {long_strike}/{short_strike} on {exp} "
                        f"(width ${width:.1f} fits slot; primary ${primary_w:.1f} ~"
                        f"${self._limit_debit(est_debit) * 100:.0f})",
                    )
                spread.est_premium = est_debit
                return spread, price_src

        print(
            f"   ⚠️ {u}: no spread fits slot ${max_debit_dollars:.0f} "
            f"({len(candidates)} tried; cheapest ~${cheapest_cost:.0f} "
            f"[{cheapest_src} debit ${cheapest_debit:.2f}]) — skip.",
        )
        return None

    def pick_call_debit_spread(self, underlying: str, spot: float) -> Optional[CallDebitSpread]:
        """Select ATM long call + higher strike short for a debit spread."""
        u = underlying.upper()
        today = date.today()
        exp_gte = today + timedelta(days=self.min_dte)
        exp_lte = today + timedelta(days=self.max_dte)

        try:
            contracts = self._fetch_call_contracts(u, exp_gte, exp_lte)
        except Exception as e:
            print(f"   ⚠️ {u}: option chain fetch failed: {e}")
            return None

        if not contracts:
            print(f"   ⚠️ {u}: no call contracts in DTE window.")
            return None

        by_exp: dict[date, list] = {}
        for c in contracts:
            exp = c.expiration_date
            if isinstance(exp, datetime):
                exp = exp.date()
            by_exp.setdefault(exp, []).append(c)

        def dte_dist(exp: date) -> int:
            return abs((exp - today).days - self.target_dte)

        primary_w = spread_width_for_spot(
            spot, self.spread_width, scale=self.scale_width_by_price
        )
        best_exp = min(by_exp.keys(), key=dte_dist)
        legs: tuple[float, float] | None = None
        used_width: float | None = None
        by_strike: dict[float, Any] = {}
        chosen_exp = best_exp
        full_by_strike: dict[float, Any] = {}

        for exp in sorted(by_exp.keys(), key=dte_dist):
            chain = by_exp[exp]
            full_by_strike = {float(c.strike_price): c for c in chain}
            matched = self._match_spread_on_chain(chain, spot)
            if matched:
                legs, used_width, by_strike = matched
                chosen_exp = exp
                if exp != best_exp:
                    print(
                        f"   ℹ️ {u}: using exp {chosen_exp} "
                        f"(target ~{self.target_dte}d was {best_exp})",
                    )
                break

        if not legs:
            print(
                f"   ⚠️ {u}: no viable call spread in DTE window "
                f"({len(by_exp)} expiries; tried widths {self._widths_to_try(spot)} "
                f"+ next strike up).",
            )
            return None

        long_strike, short_strike = legs
        if used_width is not None and abs(used_width - primary_w) > 0.01:
            print(
                f"   ℹ️ {u}: {long_strike}/{short_strike} on {chosen_exp} "
                f"(width ${used_width:.1f}, primary ${primary_w:.1f} unavailable)",
            )

        long_c = by_strike.get(long_strike)
        short_c = by_strike.get(short_strike)
        if not long_c or not short_c:
            long_c = full_by_strike.get(long_strike)
            short_c = full_by_strike.get(short_strike)
        if not long_c or not short_c:
            print(f"   ⚠️ {u}: could not match spread legs {long_strike}/{short_strike}.")
            return None

        long_px = float(long_c.close_price or 0)
        short_px = float(short_c.close_price or 0)
        if long_px <= 0:
            long_px = 0.05
        est_debit = max(0.05, long_px - short_px)

        return OptionSpread(
            underlying=u,
            kind=SpreadKind.CALL_DEBIT,
            long_symbol=str(long_c.symbol),
            short_symbol=str(short_c.symbol),
            expiration=chosen_exp,
            long_strike=long_strike,
            short_strike=short_strike,
            est_premium=est_debit,
        )

    def _limit_debit(self, est_debit: float) -> float:
        slip = 1.0 + self.limit_slippage_pct
        return round(est_debit * slip, 2)

    def _limit_credit(self, est_credit: float) -> float:
        slip = 1.0 - self.limit_slippage_pct
        return -round(max(0.01, est_credit * slip), 2)

    def _contract_cost(self, spread: OptionSpread) -> float:
        if spread.is_credit:
            width_dollars = abs(spread.short_strike - spread.long_strike)
            return max(50.0, (width_dollars - spread.est_premium) * 100.0)
        return self._limit_debit(spread.est_premium) * 100.0

    def _open_legs(self, spread: OptionSpread) -> list[OptionLegRequest]:
        if spread.kind == SpreadKind.LONG_CALL:
            return []
        if spread.is_credit:
            return [
                OptionLegRequest(
                    symbol=spread.short_symbol,
                    side=OrderSide.SELL,
                    ratio_qty=1,
                    position_intent=PositionIntent.SELL_TO_OPEN,
                ),
                OptionLegRequest(
                    symbol=spread.long_symbol,
                    side=OrderSide.BUY,
                    ratio_qty=1,
                    position_intent=PositionIntent.BUY_TO_OPEN,
                ),
            ]
        return [
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

    def _close_legs(self, spread: OptionSpread, long_sym: str, short_sym: str) -> list[OptionLegRequest]:
        if spread.kind == SpreadKind.LONG_CALL:
            return []
        if spread.is_credit:
            return [
                OptionLegRequest(
                    symbol=short_sym,
                    side=OrderSide.BUY,
                    ratio_qty=1,
                    position_intent=PositionIntent.BUY_TO_CLOSE,
                ),
                OptionLegRequest(
                    symbol=long_sym,
                    side=OrderSide.SELL,
                    ratio_qty=1,
                    position_intent=PositionIntent.SELL_TO_CLOSE,
                ),
            ]
        return [
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

    def _submit_open(self, spread: OptionSpread, max_budget_dollars: float) -> bool:
        if spread.is_credit:
            limit_px = self._limit_credit(spread.est_premium)
            price_src = "quote"
            est = spread.est_premium
        else:
            est = spread.est_premium
            limit_px = self._limit_debit(est)
            price_src = "quote"
        contract_cost = self._contract_cost(spread)
        if contract_cost <= 0:
            print(f"   ⚠️ {spread.underlying}: non-positive spread cost — skip.")
            return False

        pending = self._pending_open_orders_by_underlying().get(spread.underlying, [])
        if pending:
            same_legs = [o for o in pending if self._order_matches_spread(o, spread)]
            old_limit = float(getattr(same_legs[0], "limit_price", 0) or 0) if same_legs else 0.0
            reprice_gap = max(0.02, abs(old_limit) * 0.02)
            better = (
                (spread.is_credit and limit_px < old_limit - reprice_gap)
                or (not spread.is_credit and limit_px > old_limit + reprice_gap)
            )
            if same_legs and better:
                n = self._cancel_pending_opens(spread.underlying)
                print(
                    f"   ℹ️ {spread.underlying}: cancelled {n} stale open order(s) "
                    f"(${old_limit:.2f} -> ${limit_px:.2f})",
                )
            else:
                lims = ", ".join(
                    f"${float(getattr(o, 'limit_price', 0) or 0):.2f}" for o in pending[:3]
                )
                print(
                    f"   🔇 {spread.underlying}: open order already pending "
                    f"({len(pending)} @ {lims}) — skip.",
                )
                return False

        qty = int(max_budget_dollars // contract_cost)
        qty = max(1, min(qty, self.max_contracts_per_slot))
        slip_pct = int(round(self.limit_slippage_pct * 100))

        if spread.kind == SpreadKind.LONG_CALL:
            cost_est = limit_px * 100.0 * qty
            print(
                f"   📈 {spread.underlying}: {spread.label} exp {spread.expiration} "
                f"x{qty} limit ${limit_px:.2f} (~${cost_est:.0f}) [quote ${est:.2f} +{slip_pct}%]",
            )
            try:
                order = LimitOrderRequest(
                    symbol=spread.long_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    limit_price=limit_px,
                    time_in_force=TimeInForce.DAY,
                )
                self.client.submit_order(order)
                self._spread_book[spread.underlying] = spread
                print(f"   ✅ Open submitted for {spread.underlying} (x{qty})")
                return True
            except Exception as e:
                print(f"   ❌ {spread.underlying}: open failed: {e}")
                return False

        legs = self._open_legs(spread)
        cost_est = (abs(limit_px) if spread.is_credit else limit_px) * 100.0 * qty
        prem_label = "credit" if spread.is_credit else "debit"
        print(
            f"   📈 {spread.underlying}: {spread.label} exp {spread.expiration} "
            f"x{qty} limit ${limit_px:.2f} (~${cost_est:.0f}) "
            f"[{prem_label} ${est:.2f}]",
        )
        try:
            order = LimitOrderRequest(
                order_class=OrderClass.MLEG,
                qty=qty,
                time_in_force=TimeInForce.DAY,
                type=OrderType.LIMIT,
                limit_price=limit_px,
                legs=legs,
            )
            self.client.submit_order(order)
            self._spread_book[spread.underlying] = spread
            print(f"   ✅ MLeg open submitted for {spread.underlying} (x{qty})")
            return True
        except Exception as e:
            print(f"   ❌ {spread.underlying}: open failed: {e}")
            return False

    def open_spread(
        self,
        underlying: str,
        spot: float,
        kind: SpreadKind,
        max_budget_dollars: float,
    ) -> bool:
        picked = self.pick_spread_within_budget(underlying, spot, kind, max_budget_dollars)
        if not picked:
            return False
        spread, _src = picked
        return self._submit_open(spread, max_budget_dollars)

    def open_bullish_spread(
        self,
        underlying: str,
        spot: float,
        max_budget_dollars: float,
        strategies: Iterable[str],
        confidence: float = 0.0,
        long_call_min_confidence: float = 0.70,
    ) -> bool:
        kind_map = {
            "call_debit": SpreadKind.CALL_DEBIT,
            "bull_put_credit": SpreadKind.BULL_PUT_CREDIT,
            "long_call": SpreadKind.LONG_CALL,
        }
        for name in strategies:
            kind = kind_map.get(str(name).lower())
            if kind is None:
                continue
            if kind == SpreadKind.LONG_CALL and confidence < long_call_min_confidence:
                continue
            if self.open_spread(underlying, spot, kind, max_budget_dollars):
                return True
        return False

    def open_bearish_spread(
        self,
        underlying: str,
        spot: float,
        max_budget_dollars: float,
        strategies: Iterable[str],
    ) -> bool:
        kind_map = {
            "put_debit": SpreadKind.PUT_DEBIT,
            "bear_call_credit": SpreadKind.BEAR_CALL_CREDIT,
        }
        for name in strategies:
            kind = kind_map.get(str(name).lower())
            if kind is None:
                continue
            if self.open_spread(underlying, spot, kind, max_budget_dollars):
                return True
        return False

    def open_call_debit_spread(
        self,
        underlying: str,
        spot: float,
        max_debit_dollars: float,
    ) -> bool:
        return self.open_spread(
            underlying, spot, SpreadKind.CALL_DEBIT, max_debit_dollars
        )

    def close_spread(self, underlying: str) -> bool:
        u = underlying.upper()
        spread = self._spread_book.get(u)
        try:
            positions = list(self.client.get_all_positions())
        except Exception:
            positions = []

        long_sym = short_sym = None
        long_qty = short_qty = 0.0
        single_long = None
        single_qty = 0.0
        for pos in positions:
            sym = getattr(pos, "symbol", "") or ""
            parsed = parse_occ_symbol(sym)
            if not parsed or parsed[0] != u:
                continue
            qty = float(getattr(pos, "qty", 0) or 0)
            if parsed[2] == "C":
                if qty > 0:
                    long_sym, long_qty = sym, abs(qty)
                elif qty < 0:
                    short_sym, short_qty = sym, abs(qty)
            elif parsed[2] == "P":
                if qty > 0:
                    long_sym, long_qty = sym, abs(qty)
                elif qty < 0:
                    short_sym, short_qty = sym, abs(qty)

        if spread and spread.kind == SpreadKind.LONG_CALL:
            single_long = spread.long_symbol
            single_qty = long_qty or 1.0
        elif long_sym and not short_sym and long_qty > 0:
            single_long = long_sym
            single_qty = long_qty

        if single_long:
            qty = int(max(1, round(single_qty)))
            print(f"   🔻 {u}: CLOSE {spread.label if spread else 'LONG CALL'} x{qty}")
            try:
                if self.close_use_market:
                    order = MarketOrderRequest(
                        symbol=single_long,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                else:
                    order = LimitOrderRequest(
                        symbol=single_long,
                        qty=qty,
                        side=OrderSide.SELL,
                        type=OrderType.LIMIT,
                        limit_price=0.01,
                        time_in_force=TimeInForce.DAY,
                    )
                self.client.submit_order(order)
                self._spread_book.pop(u, None)
                print(f"   ✅ Close submitted for {u}")
                return True
            except Exception as e:
                print(f"   ❌ {u}: close failed: {e}")
                return False

        if spread:
            long_sym = long_sym or spread.long_symbol
            short_sym = short_sym or spread.short_symbol
        if not long_sym or not short_sym:
            print(f"   ⚠️ {u}: no spread legs found to close.")
            return False
        if not spread:
            pl = parse_occ_symbol(long_sym)
            ps = parse_occ_symbol(short_sym)
            kind = SpreadKind.CALL_DEBIT
            is_credit = False
            if pl and ps:
                if pl[2] == "P" and pl[3] > ps[3]:
                    kind = SpreadKind.PUT_DEBIT
                elif pl[2] == "P" and pl[3] < ps[3]:
                    kind = SpreadKind.BULL_PUT_CREDIT
                    is_credit = True
                elif pl[2] == "C" and pl[3] > ps[3]:
                    kind = SpreadKind.BEAR_CALL_CREDIT
                    is_credit = True
            spread = OptionSpread(
                underlying=u,
                kind=kind,
                long_symbol=long_sym,
                short_symbol=short_sym,
                expiration=pl[1] if pl else date.today(),
                long_strike=pl[3] if pl else 0.0,
                short_strike=ps[3] if ps else 0.0,
                est_premium=0.05,
                is_credit=is_credit,
            )

        qty = int(max(1, round(max(long_qty, short_qty) or 1)))
        legs = self._close_legs(spread, long_sym, short_sym)
        if self.close_use_market:
            order = MarketOrderRequest(
                order_class=OrderClass.MLEG,
                qty=qty,
                time_in_force=TimeInForce.DAY,
                legs=legs,
            )
        else:
            order = LimitOrderRequest(
                order_class=OrderClass.MLEG,
                qty=qty,
                time_in_force=TimeInForce.DAY,
                type=OrderType.LIMIT,
                limit_price=0.01,
                legs=legs,
            )
        kind = "MKT" if self.close_use_market else "LMT"
        label = spread.label if spread else "spread"
        print(f"   🔻 {u}: CLOSE {label} x{qty} [{kind}] ({long_sym} / {short_sym})")
        try:
            self.client.submit_order(order)
            self._spread_book.pop(u, None)
            print(f"   ✅ MLeg close submitted for {u}")
            return True
        except Exception as e:
            print(f"   ❌ {u}: close spread failed: {e}")
            return False

    def close_call_debit_spread(self, underlying: str) -> bool:
        return self.close_spread(underlying)
