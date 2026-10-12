"""
Walk-forward study: run every estimator over every study day, score each
hour separately, and save per-hour results for analyze_walkforward.py.

Per day:
  1. Download bookTicker + aggTrades (skipped if already on disk).
  2. Build the four estimators (BV dropped after step 1 -- see README):
       RV, EWMA       no parameters to fit.
       GARCH          refit every hour on the trailing 3 hours.
       HAR            refit every hour on the day so far (expanding window,
                      non-negative coefficients; trailing 3h was unstable --
                      see estimators.har_vol_walkforward).
     Each in two variants: raw, and level-matched (rescaled every hour so its
     average variance over the trailing 3 hours equals realized variance).
     Everything is causal: what's in force from hour T on uses data before T.
  3. Simulate 4 estimators x 2 variants x S seeds in one batch, with common
     random numbers within each seed.
  4. Score hours 03:00-24:00 separately (00:00-03:00 is the first fit
     window). Per hour it records P&L, fills, notional, spread, inventory,
     QLIKE, and market features for regime labeling: realized vol over the
     trailing 3 hours (the regime signal, past data only) and over the hour.
  5. Write output/walkforward/blocks_<date>.csv, delete the raw CSVs.

Resumable: days with a results file are skipped unless --force.
Usage (from src/): python -m legacy.walkforward_1s [--workers 3] [--seeds 10] [--dates D1 D2 ...] [--force]
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

from mmvol.data import download_and_extract
from mmvol.estimators import (
    ewma_lambda_matching_window, ewma_vol, garch_vol_walkforward, har_vol_walkforward,
    level_match_walkforward, realized_vol,
)
from legacy.pipeline_1s import (
    CLIP_SIZE, K, OUT, RAW, ROLLING_HORIZON_SEC, ROOT, SYMBOL, calibrate, load_book_1s,
    load_trade_fill_signals, run_as_sim_batch, solve_fill_params,
)

FIRST_FIT_SEC = 3 * 3600
REFIT_SEC = 3600
TRAIN_SEC = 3 * 3600
BLOCK_SEC = 3600
REGIME_LOOKBACK_SEC = 3 * 3600
FEATURE_SAMPLING_SEC = 60  # regime feature uses 60s returns: the signature plot showed 1s RV's bias varies by regime
SEED_BASE = 1000
ESTIMATORS = ["RV", "EWMA", "GARCH", "HAR"]
VARIANTS = ["raw", "matched"]
KEEP_RAW = {"2024-01-10", "2023-07-19"}  # development days, used by the other scripts
WF_DIR = OUT / "walkforward"
HOURS_PER_YEAR = 365 * 24

# Failed GARCH fits are detected via convergence_flag and counted in the per-day diagnostics.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _ann_vol_from_r60(r60: np.ndarray, seconds: int) -> float:
    return math.sqrt(np.sum(r60 ** 2) * HOURS_PER_YEAR * 3600 / seconds)


def process_day(date: str, n_seeds: int, keep_raw: bool) -> dict:
    t_start = time.time()
    for dt in ("bookTicker", "aggTrades"):
        download_and_extract(SYMBOL, date, dt, RAW)

    mid, best_bid, best_ask = load_book_1s(date)
    sell_min, buy_max = load_trade_fill_signals(mid.index, date)
    sigma_guess, gamma = calibrate(mid)
    fill_A, fill_k = solve_fill_params()
    n = len(mid)

    raw = {
        "RV": realized_vol(mid, ROLLING_HORIZON_SEC, sigma_guess),
        "EWMA": ewma_vol(mid, ewma_lambda_matching_window(ROLLING_HORIZON_SEC), sigma_guess),
    }
    raw["GARCH"], g_diag = garch_vol_walkforward(mid, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
    raw["HAR"], h_diag = har_vol_walkforward(mid, ROLLING_HORIZON_SEC, FIRST_FIT_SEC, REFIT_SEC)
    sigmas = {("raw", e): raw[e] for e in ESTIMATORS}
    for e in ESTIMATORS:
        sigmas[("matched", e)] = level_match_walkforward(raw[e], mid, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
    keys = list(sigmas)  # (variant, estimator)

    # batch: every (variant, estimator) for every seed
    batch = [(v, e, s) for s in range(n_seeds) for (v, e) in keys]
    sigma_b = np.vstack([sigmas[(v, e)] for (v, e, _) in batch])
    seed_idx = np.array([s for (_, _, s) in batch])
    rng = np.random.default_rng([SEED_BASE, int(date.replace("-", ""))])
    uniforms = rng.random((n_seeds, n, 2))
    sim = run_as_sim_batch(mid.values, sell_min, buy_max, best_bid.values, best_ask.values, sigma_b,
                           uniforms, seed_idx, gamma, K, ROLLING_HORIZON_SEC, CLIP_SIZE, fill_A, fill_k)
    pnl, bid, ask, fb, fa = sim["pnl"], sim["bid"], sim["ask"], sim["filled_bid"], sim["filled_ask"]
    inv = CLIP_SIZE * (np.cumsum(fb, axis=0) - np.cumsum(fa, axis=0))

    # forecast scoring inputs
    mid_v = mid.values
    r2 = pd.Series(np.r_[0.0, np.diff(np.log(mid_v)) ** 2])
    y = r2[::-1].rolling(ROLLING_HORIZON_SEC, min_periods=ROLLING_HORIZON_SEC).mean()[::-1].shift(-1).values
    h = {key: (sigmas[key] / mid_v) ** 2 for key in keys}
    r60 = np.diff(np.log(mid_v[::FEATURE_SAMPLING_SEC]))
    trade_secs = (~np.isnan(sell_min)) | (~np.isnan(buy_max))

    rows = []
    for start in range(FIRST_FIT_SEC, n, BLOCK_SEC):
        end = start + BLOCK_SEC
        i0, i1 = start // FEATURE_SAMPLING_SEC, end // FEATURE_SAMPLING_SEC
        market = {
            "date": date, "hour": start // 3600,
            "trailing_vol": _ann_vol_from_r60(r60[(start - REGIME_LOOKBACK_SEC) // FEATURE_SAMPLING_SEC:i0],
                                              REGIME_LOOKBACK_SEC),
            "block_vol": _ann_vol_from_r60(r60[i0:i1], BLOCK_SEC),
            "dollar_vol": math.sqrt(np.sum(np.diff(mid_v[start - 1:end]) ** 2)),
            "mid_return": math.log(mid_v[end - 1] / mid_v[start - 1]),
            "trade_seconds": int(trade_secs[start:end].sum()),
        }
        t_eval = np.arange(start, min(end, n - ROLLING_HORIZON_SEC))
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
        blk_fills = fb[start:end].sum(axis=0) + fa[start:end].sum(axis=0)
        blk_notional = CLIP_SIZE * ((bid[start:end] * fb[start:end]).sum(axis=0)
                                    + (ask[start:end] * fa[start:end]).sum(axis=0))
        blk_spread = (ask[start:end] - bid[start:end]).mean(axis=0)
        blk_inv = np.abs(inv[start:end]).mean(axis=0)
        for j, (v, e, s) in enumerate(batch):
            rows.append({**market, "variant": v, "estimator": e, "seed": s,
                         "pnl": blk_pnl[j], "fills": int(blk_fills[j]), "notional": blk_notional[j],
                         "spread_mean": blk_spread[j], "inv_abs_mean": blk_inv[j], **fcst[(v, e)]})

    WF_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WF_DIR / f"blocks_{date}.csv", index=False)
    diag = {"date": date, "gamma": gamma, "seconds": round(time.time() - t_start, 1), **g_diag, **h_diag}
    (WF_DIR / f"diag_{date}.json").write_text(json.dumps(diag))

    if not keep_raw and date not in KEEP_RAW:
        for dt in ("bookTicker", "aggTrades"):
            (RAW / f"{SYMBOL}-{dt}-{date}.csv").unlink(missing_ok=True)
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun days that already have results")
    args = ap.parse_args()

    dates = args.dates or pd.read_csv(ROOT / "data" / "study_days.csv")["date"].tolist()
    todo = [d for d in dates if args.force or not (WF_DIR / f"blocks_{d}.csv").exists()]
    print(f"{len(dates)} days, {len(dates) - len(todo)} already done, {len(todo)} to run "
          f"({args.workers} workers, {args.seeds} seeds)", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_day, d, args.seeds, args.keep_raw): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                diag = fut.result()
                print(f"[{i}/{len(todo)}] {d} done in {diag['seconds']}s  "
                      f"garch failed fits {diag['garch_failed']}/{diag['garch_fits']}  "
                      f"har floored {diag['har_floored_s']}s", flush=True)
            except Exception as exc:
                print(f"[{i}/{len(todo)}] {d} FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
