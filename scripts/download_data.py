"""
Script to download massive historical data to local SSD using yfinance.
Ensures 10 years of Daily data for Swing Trading.

By default only refreshes CSVs older than --max-age-hours (avoids re-downloading
thousands of symbols on every bot restart). Use --force for a full refresh.
"""
import os
import sys
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.market_fetcher import MarketDataFetcher
from src.data.csv_utils import is_corrupt_csv, save_swing_csv, csv_header_col_count

# Configuration
DATA_DIR = "data/historical_swing"
TIMEFRAME = "1d"
PERIOD = "10y"
DEFAULT_SYMBOLS_FILE = "config/watchlists/my_portfolio.txt"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _needs_update(file_path: str, max_age_hours: float) -> bool:
    if not os.path.exists(file_path):
        return True
    age_sec = time.time() - os.path.getmtime(file_path)
    return age_sec > max_age_hours * 3600.0


def download_data(
    watchlist_path=None,
    max_age_hours: float = 20.0,
    force: bool = False,
    repair: bool = False,
    workers: int = 2,
):
    symbols_file = watchlist_path if watchlist_path else DEFAULT_SYMBOLS_FILE
    fetcher = MarketDataFetcher()

    print(f"🚀 Swing data update (yfinance Ticker.history)", flush=True)
    print(f"📂 Target: {os.path.abspath(DATA_DIR)}", flush=True)
    print(f"⏱️  Timeframe: {TIMEFRAME} | Period: {PERIOD}", flush=True)
    if force:
        print("   Mode: FORCE (re-download all)", flush=True)
    elif repair:
        print("   Mode: REPAIR (re-download corrupt CSVs only)", flush=True)
    else:
        print(f"   Mode: incremental (refresh if older than {max_age_hours:.0f}h)", flush=True)
    print(f"   Workers: {workers} (sequential Ticker.history — avoids yf.download races)", flush=True)

    ensure_dir(DATA_DIR)

    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"❌ Could not find {symbols_file}", flush=True)
        return

    print(f"📋 Watchlist: {len(symbols)} symbols", flush=True)

    if force:
        todo = symbols
        skipped = 0
    elif repair:
        todo = []
        skipped = 0
        for sym in symbols:
            path = os.path.join(DATA_DIR, f"{sym}_1D.csv")
            if is_corrupt_csv(path):
                todo.append(sym)
            else:
                skipped += 1
    else:
        todo = []
        skipped = 0
        for sym in symbols:
            path = os.path.join(DATA_DIR, f"{sym}_1D.csv")
            stale = _needs_update(path, max_age_hours)
            bad_header = csv_header_col_count(path) > 7 if os.path.exists(path) else False
            if stale or bad_header:
                todo.append(sym)
            else:
                skipped += 1

    if not todo:
        print(f"✅ All {len(symbols)} CSVs are fresh — nothing to download.", flush=True)
        return

    print(f"   ↳ Downloading {len(todo)} symbols ({skipped} skipped as fresh)", flush=True)

    ok = 0
    failed = []

    def download_symbol(symbol):
        try:
            file_path = os.path.join(DATA_DIR, f"{symbol}_1D.csv")
            df = fetcher.get_history_yfinance(symbol, period=PERIOD, interval=TIMEFRAME)
            if df is None or df.empty:
                if os.path.exists(file_path) and is_corrupt_csv(file_path):
                    os.remove(file_path)
                return ('fail', symbol, 'no data')
            save_swing_csv(df, file_path)
            # Verify round-trip
            if is_corrupt_csv(file_path):
                os.remove(file_path)
                return ('fail', symbol, 'saved file still corrupt')
            return ('ok', symbol, None)
        except Exception as e:
            return ('fail', symbol, str(e))

    n_workers = max(1, min(workers, len(todo)))
    print(f"🚀 Parallel download ({n_workers} workers)", flush=True)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(download_symbol, sym): sym for sym in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Downloading"):
            status, sym, err = future.result()
            if status == 'ok':
                ok += 1
            else:
                failed.append((sym, err))

    print(f"\n✅ Download Complete. OK={ok}  Failed={len(failed)}  Skipped={skipped}", flush=True)
    if failed:
        # Summarize failures (don't spam thousands of lines)
        sample = failed[:8]
        for sym, err in sample:
            print(f"   ⚠️ {sym}: {err}", flush=True)
        if len(failed) > 8:
            print(f"   ... and {len(failed) - 8} more failures", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchlist", type=str, help="Path to watchlist file", default=None)
    parser.add_argument("--max-age-hours", type=float, default=20.0,
                        help="Skip CSVs updated within this many hours (default 20)")
    parser.add_argument("--force", action="store_true", help="Re-download every symbol")
    parser.add_argument("--repair", action="store_true",
                        help="Re-download only CSVs with corrupt structure/prices")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel download workers (default 2; keep low for yfinance stability)")
    args = parser.parse_args()

    download_data(
        args.watchlist,
        max_age_hours=args.max_age_hours,
        force=args.force,
        repair=args.repair,
        workers=args.workers,
    )
