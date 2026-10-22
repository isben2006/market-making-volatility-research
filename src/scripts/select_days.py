"""
Pick the days for the walk-forward study.

Downloads Binance's monthly daily-candle klines for BTCUSDT perp across the
bookTicker window and keeps every weekday in it, except days whose archive
data is incomplete: EXCLUDE, plus days failing the quote-gap check in
data/data_quality.csv. Weekends are excluded: they
have much lower volume, so including them would mix a liquidity regime into
the volatility comparison. Each day is tagged with its daily high-low range
tercile (calm / mid / volatile) for reference.

The first version of the study used a stratified random sample of 41 of
these days (13 per range tercile plus the two development days), kept in
data/study_days_41.csv and reproducible with --per-tercile 13 --seed 7. A
power analysis (scripts/analyze_power.py) showed regime-level comparisons
needed more days, and a day costs well under a minute to run, so the study
now uses all of them.

Using the full window's daily range here is fine: it only decides which days
get sampled. Regime labels used in the analysis come from trailing intraday
volatility, computed from past data only (see mmvol/analysis.py).

Output: data/study_days.csv
Usage (from src/): python -m scripts.select_days [--per-tercile N --seed 7 --out FILE]  (default: all weekdays)
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from mmvol.config import ROOT, SYMBOL

KLINES_URL = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1d/{s}-1d-{m}.zip"
MONTHS = pd.period_range("2023-05", "2024-03", freq="M").astype(str)
# bookTicker exists 2023-05-16..2024-03-30; trim the edges in case the first/last days are partial
WINDOW = ("2023-05-17", "2024-03-29")
ALWAYS_INCLUDE = ["2024-01-10", "2023-07-19"]  # the two days the pipeline was developed on
# Binance's bookTicker archive is incomplete on these days: no quotes for the first hours (checked 2026-10-21).
# Days with quote gaps that have trades inside them are excluded too, from data/data_quality.csv
# (scripts/check_data_quality.py).
QUALITY_PATH = ROOT / "data" / "data_quality.csv"
EXCLUDE = {
    "2023-05-17": "bookTicker starts at 08:46 UTC; also a 6-minute outage around 10:48",
    "2023-09-22": "bookTicker starts at 05:20 UTC",
}
OUT_PATH = ROOT / "data" / "study_days.csv"


def load_daily_klines() -> pd.DataFrame:
    frames = []
    for m in MONTHS:
        with urllib.request.urlopen(KLINES_URL.format(s=SYMBOL, m=m)) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            frames.append(pd.read_csv(zf.open(zf.namelist()[0])))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    return df.drop_duplicates("date").sort_values("date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tercile", type=int, default=None, help="sample this many days per tercile instead of all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    df = load_daily_klines()
    df = df[(df["date"] >= WINDOW[0]) & (df["date"] <= WINDOW[1])].copy()
    df["range_pct"] = (df["high"] - df["low"]) / df["open"] * 100
    df["weekday"] = df["date"].dt.day_name()
    wd = df[~df["weekday"].isin(["Saturday", "Sunday"])].copy()
    wd["tercile"] = pd.qcut(wd["range_pct"], 3, labels=["calm", "mid", "volatile"])
    wd["date_str"] = wd["date"].dt.strftime("%Y-%m-%d")

    if args.per_tercile is None:
        exclude = dict(EXCLUDE)
        if QUALITY_PATH.exists():
            q = pd.read_csv(QUALITY_PATH)
            for r in q[~q["ok"].astype(bool)].itertuples():
                exclude[r.date] = (f"{r.n_long_gaps} quote gap(s) >= 30 s with {r.trades_in_long_gaps} trades inside; "
                                   f"longest {r.max_gap_s:.0f} s at {r.max_gap_at_h:.2f}h")
        else:
            print(f"[warn] {QUALITY_PATH.name} missing: run scripts.check_data_quality, then rerun this")
        for d, why in sorted(exclude.items()):
            print(f"excluded {d}: {why}")
        chosen = wd[~wd["date_str"].isin(exclude)]
    else:
        rng = np.random.default_rng(args.seed)
        picks = []
        for t, grp in wd.groupby("tercile", observed=True):
            pool = grp[~grp["date_str"].isin(ALWAYS_INCLUDE)]
            idx = rng.choice(len(pool), size=min(args.per_tercile, len(pool)), replace=False)
            picks.append(pool.iloc[np.sort(idx)])
        chosen = pd.concat(picks + [wd[wd["date_str"].isin(ALWAYS_INCLUDE)]])
    chosen = chosen.sort_values("date")[["date_str", "weekday", "range_pct", "volume", "tercile"]]
    chosen = chosen.rename(columns={"date_str": "date"})

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    chosen.to_csv(out_path, index=False, float_format="%.4f")
    print(f"{len(wd)} weekdays in window; tercile cutoffs (daily range %): "
          f"{wd['range_pct'].quantile([1/3, 2/3]).round(2).tolist()}")
    print(chosen.groupby("tercile", observed=True).size().to_string())
    print(f"months covered: {chosen['date'].str[:7].nunique()} of {len(MONTHS)}")
    print(f"[csv] {out_path}")


if __name__ == "__main__":
    main()
