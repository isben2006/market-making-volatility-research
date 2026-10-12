"""
Download one day of Binance BTCUSDT perp bookTicker + aggTrades into data/raw/.
The walk-forward scripts download what they need themselves; this is for the
single-day scripts (signature plot, legacy).

Usage (from src/): python -m scripts.download_data --date 2024-01-10
"""
import argparse

from mmvol.config import RAW, SYMBOL
from mmvol.data import DATA_TYPES, download_and_extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--date", default="2024-01-10", help="YYYY-MM-DD, must be within 2023-05-16..2024-03-30")
    args = ap.parse_args()

    for dt in DATA_TYPES:
        path = download_and_extract(args.symbol, args.date, dt, RAW)
        print(f"  -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
