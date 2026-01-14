"""Download recent Alpaca crypto bars to CSV for offline training.

Saves files into `data/historical` with names like `BTCUSD_1Min.csv`.
This matches the expectations in `scripts/train_crypto_bot.py`.

Usage (Windows PowerShell):
  .\.venv311\Scripts\python.exe scripts\download_crypto_data.py --timeframe 1Min --lookback-days 60
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
import alpaca_trade_api as tradeapi

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CryptoTraderCreds, ALPACA_BASE_URL  # noqa: E402


DATA_DIR = os.path.join("data", "historical")
WATCHLIST_PATH = os.path.join("config", "watchlists", "crypto_watchlist.txt")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_symbols() -> list[str]:
    if not os.path.exists(WATCHLIST_PATH):
        raise FileNotFoundError(f"Missing watchlist: {WATCHLIST_PATH}")

    symbols: list[str] = []
    with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Accept BTC/USD or BTCUSD; normalize to BTC/USD for the API call
            if "/" not in s and s.upper().endswith("USD") and len(s) > 3:
                s = f"{s[:-3]}/USD"
            symbols.append(s.upper())

    if not symbols:
        raise ValueError("crypto_watchlist.txt is empty")

    return symbols


def _file_name_for_symbol(symbol: str, timeframe: str) -> str:
    # BTC/USD -> BTCUSD_1Min.csv
    safe = symbol.replace("/", "")
    return f"{safe}_{timeframe}.csv"


def _bars_to_training_csv(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure a real timestamp column exists for training feature engineering.
    out = df.reset_index()
    if "timestamp" not in out.columns:
        # Alpaca sometimes uses 'time' or the index column name.
        for candidate in ("time", "t", "index"):
            if candidate in out.columns:
                out = out.rename(columns={candidate: "timestamp"})
                break
        else:
            # fallback: rename the first column
            out = out.rename(columns={out.columns[0]: "timestamp"})

    return out


def download_crypto_bars(timeframe: str, lookback_days: int, workers: int) -> None:
    _ensure_dir(DATA_DIR)

    symbols = _load_symbols()

    if not CryptoTraderCreds.API_KEY or not CryptoTraderCreds.API_SECRET:
        raise ValueError("Missing CRYPTO_API_KEY / CRYPTO_API_SECRET in .env")

    api = tradeapi.REST(
        CryptoTraderCreds.API_KEY,
        CryptoTraderCreds.API_SECRET,
        ALPACA_BASE_URL,
        api_version="v2",
    )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))

    start_iso = start.isoformat()
    end_iso = end.isoformat()

    print(f"📥 Downloading crypto bars -> {os.path.abspath(DATA_DIR)}")
    print(f"📋 Symbols: {len(symbols)} | Timeframe: {timeframe} | Lookback: {lookback_days}d")

    def _download_one(sym: str) -> str:
        out_path = os.path.join(DATA_DIR, _file_name_for_symbol(sym, timeframe))

        # If file exists and is non-trivial, skip (user can delete to re-download)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return f"Skipped {sym}"

        try:
            bars = api.get_crypto_bars(sym, timeframe, start=start_iso, end=end_iso).df
            if bars is None or bars.empty:
                return f"No Data {sym}"

            df = _bars_to_training_csv(bars)
            df.to_csv(out_path, index=False)
            return f"Done {sym} ({len(df)} bars)"
        except Exception as e:
            return f"Error {sym}: {e}"

    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(_download_one, s) for s in symbols]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            _ = f.result()

    print("✅ Crypto download complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Alpaca crypto bars for training")
    parser.add_argument("--timeframe", type=str, default="1Min", help="Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)")
    parser.add_argument("--lookback-days", type=int, default=60, help="How many days back to download")
    parser.add_argument("--workers", type=int, default=12, help="Parallel download workers")
    args = parser.parse_args()

    download_crypto_bars(args.timeframe, args.lookback_days, args.workers)


if __name__ == "__main__":
    main()
