"""
Diagnose Alpaca account status and liquidate all equity positions.

403 errors on close usually mean:
  - account_blocked / trading_blocked
  - liquidation-only restriction (PDT, ACH return, compliance)
  - open orders locking shares (cancel orders first)
  - paper vs live account mismatch

Usage:
  .venv\\Scripts\\python.exe scripts/diagnose_and_liquidate.py
  .venv\\Scripts\\python.exe scripts/diagnose_and_liquidate.py --liquidate
"""
import argparse
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_trade_api as tradeapi
from config.settings import SwingTraderCreds, ALPACA_BASE_URL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liquidate", action="store_true", help="Cancel open orders and close all positions")
    ap.add_argument("--mode", choices=["swing", "options"], default="swing")
    args = ap.parse_args()

    if args.mode == "options":
        from config.settings import OptionsTraderCreds, OPTIONS_ALPACA_BASE_URL
        creds = OptionsTraderCreds
        api_url = OPTIONS_ALPACA_BASE_URL
    else:
        creds = SwingTraderCreds
        api_url = ALPACA_BASE_URL

    if not creds.API_KEY or not creds.API_SECRET:
        key_hint = "OPTIONS_API_KEY / OPTIONS_API_SECRET" if args.mode == "options" else "SWING_API_KEY / SWING_API_SECRET"
        print(f"❌ Missing API keys in .env ({key_hint})")
        sys.exit(1)

    api = tradeapi.REST(
        str(creds.API_KEY),
        str(creds.API_SECRET),
        str(api_url),
        api_version="v2",
    )

    print("=" * 60)
    print("ALPACA ACCOUNT DIAGNOSTIC")
    print("=" * 60)
    print(f"  API URL: {api_url}")
    print(f"  Mode:    {args.mode}")

    try:
        acct = api.get_account()
    except Exception as e:
        print(f"\n❌ Could not reach Alpaca API: {e}")
        print("   Check: API keys, ALPACA_BASE_URL (paper vs live), network.")
        sys.exit(1)

    print(f"\n  Account ID:        {acct.id}")
    print(f"  Status:            {acct.status}")
    print(f"  Equity:            ${float(acct.equity):,.2f}")
    print(f"  Cash:              ${float(acct.cash):,.2f}")
    print(f"  Buying power:      ${float(acct.buying_power):,.2f}")
    print(f"  Pattern day trader:{acct.pattern_day_trader}")
    print(f"  Daytrade count:    {getattr(acct, 'daytrade_count', '?')}")
    print(f"  Trading blocked:   {acct.trading_blocked}")
    print(f"  Account blocked:   {acct.account_blocked}")
    print(f"  Transfers blocked: {getattr(acct, 'transfers_blocked', '?')}")

    if acct.trading_blocked or acct.account_blocked:
        print("\n⚠️  ACCOUNT IS BLOCKED — manual/API trades may return 403.")
        print("   Email support@alpaca.markets with your account ID.")

    positions = api.list_positions()
    print(f"\n  Open positions:    {len(positions)}")
    for p in positions:
        print(f"    • {p.symbol}: {p.qty} @ ${float(p.current_price):.2f} "
              f"(P/L ${float(p.unrealized_pl):+.2f})")

    orders = api.list_orders(status="open")
    print(f"\n  Open orders:       {len(orders)}")
    for o in orders[:10]:
        print(f"    • {o.side} {o.qty} {o.symbol} ({o.status})")
    if len(orders) > 10:
        print(f"    ... and {len(orders) - 10} more")

    if not args.liquidate:
        print("\n" + "-" * 60)
        print("Dry run only. To liquidate via API:")
        print("  .venv\\Scripts\\python.exe scripts/diagnose_and_liquidate.py --liquidate")
        print("\nIf the Alpaca WEBSITE gives 403:")
        print("  1. Confirm you're on the same account (paper vs live) as the bot")
        print("  2. Check Dashboard -> Account -> Restrictions / PDT flag")
        print("  3. Cancel all open orders first, then close positions")
        print("  4. Contact support@alpaca.markets if account_blocked=True")
        return

    print("\n" + "=" * 60)
    print("LIQUIDATING...")
    print("=" * 60)

    # Step 1: cancel open orders (shares may be reserved)
    try:
        cancelled = api.cancel_all_orders()
        n = len(cancelled) if cancelled else 0
        print(f"  Cancelled {n} open order(s).")
    except Exception as e:
        print(f"  ⚠️ Cancel orders: {e}")

    # Step 2: close all positions (Alpaca native liquidate endpoint)
    try:
        result = api.close_all_positions()
        if not result:
            print("  No positions to close.")
        else:
            for item in result:
                sym = getattr(item, "symbol", None) or (item.get("symbol") if isinstance(item, dict) else "?")
                status = getattr(item, "status", None) or (item.get("status") if isinstance(item, dict) else "?")
                body = getattr(item, "body", None) or (item.get("body") if isinstance(item, dict) else None)
                if status and int(status) >= 400:
                    print(f"  ❌ {sym}: HTTP {status} — {body}")
                else:
                    print(f"  ✅ {sym}: close order submitted (HTTP {status})")
    except Exception as e:
        print(f"  ❌ close_all_positions failed: {e}")
        print("\n  Trying one-by-one close_position...")
        for p in positions:
            try:
                api.close_position(p.symbol)
                print(f"  ✅ Closed {p.symbol}")
            except Exception as e2:
                print(f"  ❌ {p.symbol}: {e2}")

    # Verify
    try:
        remaining = api.list_positions()
        print(f"\n  Remaining positions: {len(remaining)}")
        acct2 = api.get_account()
        print(f"  Cash after:          ${float(acct2.cash):,.2f}")
        print(f"  Equity after:        ${float(acct2.equity):,.2f}")
    except Exception as e:
        print(f"  Could not verify: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
