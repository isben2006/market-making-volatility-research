"""
Pick the days for the walk-forward study.

Downloads Binance's monthly daily-candle klines for BTCUSDT perp across the
bookTicker window, ranks weekdays by daily high-low range, and draws a
stratified random sample from each range tercile, so the study covers calm,
mid and volatile days. Weekends are excluded: they have much lower volume, so
including them would mix a liquidity regime into the volatility comparison.

Using the full window's daily range here is fine: it only decides which days
get sampled. Regime labels used in the analysis come from trailing intraday
volatility, computed from past data only (see mmvol/analysis.py).

Output: data/study_days.csv
Usage (from src/): python -m scripts.select_days [--per-tercile 13] [--seed 7]
"""
import argparse
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from mmvol.config import ROOT, SYMBOL

KLINES_URL = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1d/{s}-1d-{m}.zip"
MONTHS = pd.period_range("2023-05", "2024-03", freq="M").astype(str)
# bookTicker exists 2023-05-16..2024-03-30; trim the edges in case the first/last days are partial
WINDOW = ("2023-05-17", "2024-03-29")
ALWAYS_INCLUDE = ["2024-01-10", "2023-07-19"]  # the two days the pipeline was developed on
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
    ap.add_argument("--per-tercile", type=int, default=13)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = load_daily_klines()
    df = df[(df["date"] >= WINDOW[0]) & (df["date"] <= WINDOW[1])].copy()
    df["range_pct"] = (df["high"] - df["low"]) / df["open"] * 100
    df["weekday"] = df["date"].dt.day_name()
    wd = df[~df["weekday"].isin(["Saturday", "Sunday"])].copy()
    wd["tercile"] = pd.qcut(wd["range_pct"], 3, labels=["calm", "mid", "volatile"])
    wd["date_str"] = wd["date"].dt.strftime("%Y-%m-%d")

    rng = np.random.default_rng(args.seed)
    picks = []
    for t, grp in wd.groupby("tercile", observed=True):
        pool = grp[~grp["date_str"].isin(ALWAYS_INCLUDE)]
        idx = rng.choice(len(pool), size=min(args.per_tercile, len(pool)), replace=False)
        picks.append(pool.iloc[np.sort(idx)])
    chosen = pd.concat(picks + [wd[wd["date_str"].isin(ALWAYS_INCLUDE)]])
    chosen = chosen.sort_values("date")[["date_str", "weekday", "range_pct", "volume", "tercile"]]
    chosen = chosen.rename(columns={"date_str": "date"})

    OUT_PATH.parent.mkdir(exist_ok=True)
    chosen.to_csv(OUT_PATH, index=False, float_format="%.4f")
    print(f"{len(wd)} weekdays in window; tercile cutoffs (daily range %): "
          f"{wd['range_pct'].quantile([1/3, 2/3]).round(2).tolist()}")
    print(chosen.groupby("tercile", observed=True).size().to_string())
    print(f"months covered: {chosen['date'].str[:7].nunique()} of {len(MONTHS)}")
    print(f"[csv] {OUT_PATH}")


if __name__ == "__main__":
    main()
