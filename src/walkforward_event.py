"""
Walk-forward study on the event-driven simulator (event_sim.py). Replaces
walkforward.py's P&L, which came from the 1-second simulator with the
same-second lookahead (see README).

Per day:
  1. Download bookTicker + aggTrades, load raw events.
  2. Same four estimators as walkforward.py (RV, EWMA; GARCH and HAR refit
     hourly), raw and level-matched.
  3. AS order-flow parameter k calibrated from the data, refit hourly on the
     trailing 3 hours (event_sim.calibrate_k_walkforward). Shared by every
     estimator, so the comparison still varies only sigma.
  4. Simulate every combination of estimator x variant x fill rule
     (conservative / optimistic) x gamma (GAMMAS) in one batch: decisions
     every 100 ms, 50 ms latency, tick rounding, post-only quotes.
     Deterministic, so no seeds.
  5. Score hours 03:00-24:00 separately: P&L, fills, notional (for applying
     fees afterwards), spread, inventory, QLIKE, k in force, and the regime
     features from walkforward.py.
  6. Write output/walkforward_event/blocks_<date>.csv, delete the raw CSVs.

Usage: python walkforward_event.py [--workers 2] [--dates D1 D2 ...] [--force]
"""
import argparse
import json
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from arch.utility.exceptions import ConvergenceWarning

from download_data import download_and_extract
from estimators import (
    ewma_lambda_matching_window, ewma_vol, garch_vol_walkforward, har_vol_walkforward,
    level_match_walkforward, realized_vol,
)
from event_sim import (
    calibrate_k_walkforward, decision_sigma, load_events, mid_1s_from_book, prepare_intervals, run_event_sim,
)
from pipeline import CLIP_SIZE, DAY_SECONDS, OUT, RAW, ROLLING_HORIZON_SEC, ROOT, SYMBOL, calibrate
from walkforward import (
    BLOCK_SEC, ESTIMATORS, FEATURE_SAMPLING_SEC, FIRST_FIT_SEC, KEEP_RAW, REFIT_SEC, REGIME_LOOKBACK_SEC,
    TRAIN_SEC, VARIANTS, _ann_vol_from_r60,
)

DECISION_MS = 100
LATENCY_MS = 50
GAMMAS = [1e-5, 1e-4, 1e-3, 1e-2]
RULES = ["conservative", "optimistic"]
WF_DIR = OUT / "walkforward_event"

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def process_day(date: str, keep_raw: bool) -> dict:
    t_start = time.time()
    for dt in ("bookTicker", "aggTrades"):
        download_and_extract(SYMBOL, date, dt, RAW)

    book, trades = load_events(date)
    index = pd.date_range(f"{date} 00:00:00+00:00", periods=DAY_SECONDS, freq="1s")
    mid = mid_1s_from_book(book, index)
    sigma_guess, _ = calibrate(mid)

    raw = {
        "RV": realized_vol(mid, ROLLING_HORIZON_SEC, sigma_guess),
        "EWMA": ewma_vol(mid, ewma_lambda_matching_window(ROLLING_HORIZON_SEC), sigma_guess),
    }
    raw["GARCH"], g_diag = garch_vol_walkforward(mid, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
    raw["HAR"], h_diag = har_vol_walkforward(mid, ROLLING_HORIZON_SEC, FIRST_FIT_SEC, REFIT_SEC)
    sigmas = {("raw", e): raw[e] for e in ESTIMATORS}
    for e in ESTIMATORS:
        sigmas[("matched", e)] = level_match_walkforward(raw[e], mid, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
    keys = list(sigmas)

    iv = prepare_intervals(book, trades, DECISION_MS, LATENCY_MS)
    k_dec, k_fits = calibrate_k_walkforward(iv, DECISION_MS, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
    sig_dec = {key: decision_sigma(sigmas[key], iv["dec_t"], lagged=(key[1] != "GARCH")) for key in keys}
    del book  # largest object; everything needed from it is in iv / mid

    batch = [(v, e, rule, g) for (v, e) in keys for rule in RULES for g in GAMMAS]
    sim = run_event_sim(iv, trades, mid.values, np.vstack([sig_dec[(v, e)] for (v, e, _, _) in batch]),
                        np.array([rule == "optimistic" for (_, _, rule, _) in batch]),
                        np.array([g for (_, _, _, g) in batch]), k_dec, ROLLING_HORIZON_SEC, CLIP_SIZE,
                        DECISION_MS)
    pnl = sim["pnl"]
    per_sec = sim["per_sec"]

    mid_v = mid.values
    r2 = pd.Series(np.r_[0.0, np.diff(np.log(mid_v)) ** 2])
    y = r2[::-1].rolling(ROLLING_HORIZON_SEC, min_periods=ROLLING_HORIZON_SEC).mean()[::-1].shift(-1).values
    h = {key: (sigmas[key] / mid_v) ** 2 for key in keys}
    r60 = np.diff(np.log(mid_v[::FEATURE_SAMPLING_SEC]))
    k_sec = k_dec[::per_sec]

    rows = []
    for start in range(FIRST_FIT_SEC, DAY_SECONDS, BLOCK_SEC):
        end = start + BLOCK_SEC
        i0, i1 = start // FEATURE_SAMPLING_SEC, end // FEATURE_SAMPLING_SEC
        market = {
            "date": date, "hour": start // 3600,
            "trailing_vol": _ann_vol_from_r60(r60[(start - REGIME_LOOKBACK_SEC) // FEATURE_SAMPLING_SEC:i0],
                                              REGIME_LOOKBACK_SEC),
            "block_vol": _ann_vol_from_r60(r60[i0:i1], BLOCK_SEC),
            "dollar_vol": math.sqrt(np.sum(np.diff(mid_v[start - 1:end]) ** 2)),
            "mid_return": math.log(mid_v[end - 1] / mid_v[start - 1]),
            "k": float(k_sec[start:end].mean()),
        }
        t_eval = np.arange(start, min(end, DAY_SECONDS - ROLLING_HORIZON_SEC))
        fcst = {}
        for key in keys:
            hh, yy = h[key][t_eval], y[t_eval]
            ok = (hh > 0) & (yy > 0)
            ratio = yy[ok] / hh[ok]
            fcst[key] = {
                "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)) if ok.any() else np.nan,
                "fcst_bias": float(hh[ok].mean() / yy[ok].mean()) if ok.any() else np.nan,
            }
        blk_pnl = pnl[end - 1] - pnl[start - 1]
        blk = {name: sim[name][start:end].sum(axis=0) for name in ["fills", "notional", "spread_sum", "inv_abs_sum"]}
        n_dec_blk = BLOCK_SEC * per_sec
        for j, (v, e, rule, g) in enumerate(batch):
            rows.append({**market, "variant": v, "estimator": e, "rule": rule, "gamma": g,
                         "pnl": blk_pnl[j], "fills": int(blk["fills"][j]), "notional": blk["notional"][j],
                         "spread_mean": blk["spread_sum"][j] / n_dec_blk,
                         "inv_abs_mean": blk["inv_abs_sum"][j] / n_dec_blk, **fcst[(v, e)]})

    WF_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WF_DIR / f"blocks_{date}.csv", index=False)
    diag = {"date": date, "seconds": round(time.time() - t_start, 1), "k_median": float(np.median(k_fits[:, 1])),
            "k_min": float(k_fits[:, 1].min()), "k_max": float(k_fits[:, 1].max()),
            "rejected_mean": float(sim["rejected"].mean()), **g_diag, **h_diag}
    (WF_DIR / f"diag_{date}.json").write_text(json.dumps(diag))

    if not keep_raw and date not in KEEP_RAW:
        for dt in ("bookTicker", "aggTrades"):
            (RAW / f"{SYMBOL}-{dt}-{date}.csv").unlink(missing_ok=True)
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun days that already have results")
    args = ap.parse_args()

    dates = args.dates or pd.read_csv(ROOT / "data" / "study_days.csv")["date"].tolist()
    todo = [d for d in dates if args.force or not (WF_DIR / f"blocks_{d}.csv").exists()]
    print(f"{len(dates)} days, {len(dates) - len(todo)} already done, {len(todo)} to run "
          f"({args.workers} workers; {len(ESTIMATORS) * len(VARIANTS) * len(RULES) * len(GAMMAS)} runs/day)",
          flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_day, d, args.keep_raw): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                diag = fut.result()
                print(f"[{i}/{len(todo)}] {d} done in {diag['seconds']}s  k {diag['k_min']:.2f}-{diag['k_max']:.2f} "
                      f"(median {diag['k_median']:.2f})  garch failed {diag['garch_failed']}/{diag['garch_fits']}",
                      flush=True)
            except Exception as exc:
                print(f"[{i}/{len(todo)}] {d} FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
