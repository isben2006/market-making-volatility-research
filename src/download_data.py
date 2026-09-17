"""
Pull one day of free Binance USDS-M futures data for the AS skeleton.

Data source notes (checked 2026-08-08):
  - data.binance.vision does NOT publish `bookTicker` for spot BTCUSDT at all
    (only aggTrades / trades / klines exist under data/spot/daily/).
  - data.binance.vision publishes `bookTicker` for USDS-M futures (data/futures/um/)
    but ONLY from 2023-05-16 through 2024-03-30 -- it was discontinued after that.
  - So: this whole skeleton runs on BTCUSDT USDS-M futures, and any future
    "different regimes" work is constrained to that ~10.5 month window unless
    you start capturing live bookTicker/depth streams yourself going forward.

This script just downloads + unzips one day of:
  - aggTrades  (executed trades -> used for the trade-through fill rule)
  - bookTicker (best bid/ask, tick-by-tick -> used as the reference mid price)

Usage:
    python src/download_data.py --date 2024-01-10
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/futures/um/daily/{data_type}/{symbol}/{symbol}-{data_type}-{date}.zip"
DATA_TYPES = ["aggTrades", "bookTicker"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_and_extract(symbol: str, date: str, data_type: str, out_dir: Path) -> Path:
    url = BASE_URL.format(data_type=data_type, symbol=symbol, date=date)
    zip_path = out_dir / f"{symbol}-{data_type}-{date}.zip"
    csv_path = out_dir / f"{symbol}-{data_type}-{date}.csv"

    if csv_path.exists():
        print(f"[skip] {csv_path.name} already present")
        return csv_path

    print(f"[download] {url}")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        raise

    print(f"[unzip] {zip_path.name} -> {csv_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()  # keep only the csv, no need to keep the zip around
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2024-01-10", help="YYYY-MM-DD, must be within 2023-05-16..2024-03-30")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for dt in DATA_TYPES:
        path = download_and_extract(args.symbol, args.date, dt, RAW_DIR)
        print(f"  -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
