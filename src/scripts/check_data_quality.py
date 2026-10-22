"""
Check every study day's bookTicker data for gaps (mmvol.data.book_quality).

Binance's archive has holes: stretches with no quote updates while trades keep
printing. In the simulator those make the market maker quote around a frozen
price and get filled on every trade, building positions of up to ~100 BTC.
A day fails if any gap of GAP_SEC or more (including missing data at the
start or end of the day) has trades in it.

Per day: download if needed, check, delete the raw data (except the two
development days). Writes data/data_quality.csv; resumable.

Usage (from src/): python -m scripts.check_data_quality [--workers 2] [--dates D1 D2 ...]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from mmvol.config import KEEP_RAW, RAW, ROOT, SYMBOL
from mmvol.data import GAP_SEC, book_quality, download_and_extract, load_events

OUT_PATH = ROOT / "data" / "data_quality.csv"


def check_day(date: str) -> dict:
    for dt in ("bookTicker", "aggTrades"):
        download_and_extract(SYMBOL, date, dt, RAW)
    book, trades = load_events(date)
    row = {"date": date, **book_quality(book, trades)}
    if date not in KEEP_RAW:
        for dt in ("bookTicker", "aggTrades"):
            (RAW / f"{SYMBOL}-{dt}-{date}.csv").unlink(missing_ok=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    dates = args.dates or pd.read_csv(ROOT / "data" / "study_days.csv")["date"].tolist()
    done = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame(columns=["date"])
    todo = [d for d in dates if d not in set(done["date"])]
    print(f"{len(dates)} days, {len(dates) - len(todo)} already checked, {len(todo)} to check "
          f"(gap threshold {GAP_SEC}s)", flush=True)
    rows = done.to_dict("records")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_day, d): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                r = fut.result()
                rows.append(r)
                pd.DataFrame(rows).sort_values("date").to_csv(OUT_PATH, index=False, float_format="%.4f")
                flag = "ok  " if r["ok"] else "FAIL"
                print(f"[{i}/{len(todo)}] {d} {flag} book {r['book_first_h']:.2f}-{r['book_last_h']:.2f}h, "
                      f"longest gap {r['max_gap_s']:.0f}s at {r['max_gap_at_h']:.2f}h, "
                      f"{r['trades_in_long_gaps']} trades in gaps >= {GAP_SEC}s", flush=True)
            except Exception as exc:
                print(f"[{i}/{len(todo)}] {d} ERROR: {exc!r}", flush=True)
    q = pd.DataFrame(rows)
    print(f"\n{int((~q['ok'].astype(bool)).sum())} of {len(q)} days fail. [csv] {OUT_PATH}")


if __name__ == "__main__":
    main()
