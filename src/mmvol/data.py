"""
Binance public-archive data: download, and load into the shapes the
simulator and estimators use.

Source notes (checked 2026-08-08): data.binance.vision publishes `bookTicker`
(every best bid/ask change) for USDⓈ-M futures only from 2023-05-16 through
2024-03-30, and never for spot. That window bounds the study.
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from mmvol.config import DAY_SECONDS, RAW, SYMBOL

BASE_URL = "https://data.binance.vision/data/futures/um/daily/{data_type}/{symbol}/{symbol}-{data_type}-{date}.zip"
DATA_TYPES = ["aggTrades", "bookTicker"]
DOWNLOAD_TIMEOUT_SEC = 60
DOWNLOAD_ATTEMPTS = 3


def download_and_extract(symbol: str, date: str, data_type: str, out_dir: Path = RAW) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(data_type=data_type, symbol=symbol, date=date)
    zip_path = out_dir / f"{symbol}-{data_type}-{date}.zip"
    csv_path = out_dir / f"{symbol}-{data_type}-{date}.csv"

    if csv_path.exists():
        print(f"[skip] {csv_path.name} already present")
        return csv_path

    print(f"[download] {url}")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            # timeout applies to each read: a stalled connection (e.g. after sleep) fails instead of hanging
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as resp, open(zip_path, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            break
        except Exception as e:
            zip_path.unlink(missing_ok=True)
            print(f"  attempt {attempt}/{DOWNLOAD_ATTEMPTS} FAILED: {e!r}", file=sys.stderr)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise

    print(f"[unzip] {zip_path.name} -> {csv_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    return csv_path


def load_events(date: str):
    """Raw best bid/ask updates and trades, timestamps in ms since midnight UTC."""
    bt = pd.read_csv(RAW / f"{SYMBOL}-bookTicker-{date}.csv",
                     usecols=["best_bid_price", "best_ask_price", "transaction_time"],
                     dtype={"best_bid_price": "float64", "best_ask_price": "float64", "transaction_time": "int64"})
    tr = pd.read_csv(RAW / f"{SYMBOL}-aggTrades-{date}.csv", usecols=["price", "quantity", "transact_time",
                     "is_buyer_maker"], dtype={"price": "float64", "quantity": "float64", "transact_time": "int64"})
    if tr["is_buyer_maker"].dtype != bool:
        tr["is_buyer_maker"] = tr["is_buyer_maker"].astype(str).str.strip().str.lower().eq("true")

    day0 = int(pd.Timestamp(f"{date} 00:00:00", tz="UTC").timestamp() * 1000)
    bt_ts = bt["transaction_time"].values - day0
    order = np.argsort(bt_ts, kind="stable")
    book = {"ts": bt_ts[order], "bid": bt["best_bid_price"].values[order], "ask": bt["best_ask_price"].values[order]}

    tr_ts = tr["transact_time"].values - day0
    order = np.argsort(tr_ts, kind="stable")
    tr = tr.iloc[order]
    sell = tr["is_buyer_maker"].values  # buyer was resting => a seller hit the bid
    trades = {
        "sell_ts": tr_ts[order][sell], "sell_px": tr["price"].values[sell], "sell_qty": tr["quantity"].values[sell],
        "buy_ts": tr_ts[order][~sell], "buy_px": tr["price"].values[~sell], "buy_qty": tr["quantity"].values[~sell],
    }
    return book, trades


GAP_SEC = 30  # a quote gap this long with trades inside it is a recording failure (a healthy book updates many times a second)


def book_quality(book: dict, trades: dict) -> dict:
    """Coverage of the bookTicker data and its gaps. Binance's archive has holes:
    stretches with no quote updates while trades keep printing (e.g. 14 minutes
    and 268,529 trades on 2024-01-03). The simulator would quote around a frozen
    price through them and get filled on every trade, so days with such gaps
    are excluded (scripts/check_data_quality.py)."""
    bts = np.asarray(book["ts"], dtype=float)
    tts = np.sort(np.r_[trades["sell_ts"], trades["buy_ts"]].astype(float))
    edges = np.r_[0.0, bts, DAY_SECONDS * 1000.0]  # day start and end count as gap edges
    gaps = np.diff(edges)
    i = int(np.argmax(gaps))
    long_gaps = np.where(gaps > GAP_SEC * 1000)[0]
    trades_in = [int(np.searchsorted(tts, edges[j + 1]) - np.searchsorted(tts, edges[j])) for j in long_gaps]
    return {"book_first_h": bts[0] / 3.6e6, "book_last_h": bts[-1] / 3.6e6,
            "max_gap_s": gaps[i] / 1000, "max_gap_at_h": edges[i] / 3.6e6,
            "n_long_gaps": len(long_gaps), "trades_in_long_gaps": int(sum(trades_in)),
            "ok": not any(n > 0 for n in trades_in)}


def mid_1s_from_book(book: dict, index: pd.DatetimeIndex) -> pd.Series:
    """Last mid of each second (forward-filled, back-filled before the first
    update). Identical to load_book_1s's mid, without reading the CSV again."""
    ts = book["ts"]
    last = np.clip(np.searchsorted(ts, (np.arange(DAY_SECONDS) + 1) * 1000, side="left") - 1, 0, len(ts) - 1)
    return pd.Series((book["bid"][last] + book["ask"][last]) / 2.0, index=index)


def load_book_1s(date: str):
    """Per-second last mid, best bid and best ask, read from the bookTicker CSV."""
    path = RAW / f"{SYMBOL}-bookTicker-{date}.csv"
    print(f"[load] {path.name}")
    df = pd.read_csv(
        path,
        usecols=["best_bid_price", "best_ask_price", "transaction_time"],
        dtype={"best_bid_price": "float64", "best_ask_price": "float64", "transaction_time": "int64"},
    )
    df["mid"] = (df["best_bid_price"] + df["best_ask_price"]) / 2.0
    ts = pd.to_datetime(df["transaction_time"], unit="ms", utc=True)
    book = pd.DataFrame({"mid": df["mid"].values, "bid": df["best_bid_price"].values,
                          "ask": df["best_ask_price"].values}, index=ts).sort_index()

    full_index = pd.date_range(f"{date} 00:00:00+00:00", periods=DAY_SECONDS, freq="1s")
    book_1s = book.resample("1s").last().reindex(full_index).ffill().bfill()
    print(f"  -> {len(df):,} L1 updates resampled to {len(book_1s):,} 1s bars "
          f"(mid range {book_1s['mid'].min():.1f} - {book_1s['mid'].max():.1f})")
    return book_1s["mid"], book_1s["bid"], book_1s["ask"]
