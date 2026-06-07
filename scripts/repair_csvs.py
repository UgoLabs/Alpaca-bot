"""
Scan and repair corrupt swing CSV files in data/historical_swing.

Corruption was caused by parallel yf.download() calls mixing tickers into one file.
download_data.py now uses Ticker.history() via MarketDataFetcher.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.csv_utils import is_corrupt_csv, read_swing_csv, csv_header_col_count

DATA_DIR = "data/historical_swing"
DEFAULT_SYMBOLS_FILE = "config/watchlists/my_portfolio.txt"


def load_symbols(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def scan(symbols: list[str]) -> dict:
    stats = {"missing": [], "wide_header": [], "corrupt": [], "ok": []}
    for sym in symbols:
        path = os.path.join(DATA_DIR, f"{sym}_1D.csv")
        if not os.path.exists(path):
            stats["missing"].append(sym)
            continue
        if csv_header_col_count(path) > 7:
            stats["wide_header"].append(sym)
        if is_corrupt_csv(path):
            stats["corrupt"].append(sym)
        else:
            stats["ok"].append(sym)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Scan/repair swing CSV files")
    parser.add_argument("--watchlist", default=DEFAULT_SYMBOLS_FILE)
    parser.add_argument("--scan-only", action="store_true", help="Report only; do not download")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    symbols = load_symbols(args.watchlist)
    stats = scan(symbols)

    print(f"Symbols: {len(symbols)}")
    print(f"  OK:           {len(stats['ok'])}")
    print(f"  Corrupt:      {len(stats['corrupt'])}")
    print(f"  Wide header:  {len(stats['wide_header'])}")
    print(f"  Missing:      {len(stats['missing'])}")

    if stats["corrupt"][:10]:
        print(f"  Sample bad:   {', '.join(stats['corrupt'][:10])}")

    if args.scan_only:
        return

    if not stats["corrupt"] and not stats["missing"]:
        print("Nothing to repair.")
        return

    from scripts.download_data import download_data

    download_data(
        watchlist_path=args.watchlist,
        force=False,
        repair=True,
        workers=args.workers,
    )

    # Re-scan
    after = scan(symbols)
    print(f"\nAfter repair: OK={len(after['ok'])}  still corrupt={len(after['corrupt'])}")


if __name__ == "__main__":
    main()
