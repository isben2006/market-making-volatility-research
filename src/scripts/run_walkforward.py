"""
The 41-day walk-forward study on the event-driven simulator.

Per day:
  1. Download bookTicker + aggTrades, load raw events.
  2. Four estimators (RV, EWMA; GARCH and HAR refit hourly), raw and
     level-matched (mmvol.walkforward.build_sigmas), plus three benchmarks,
     variant "benchmark": Flat (trailing 3h realized vol, updated hourly),
     OracleHour (the current hour's realized vol, held flat) and Oracle (the
     realized vol of the next 5 minutes). The oracles aren't tradable.
  3. AS order-flow parameter k calibrated from the data, refit hourly on the
     trailing 3 hours. Shared by every estimator, so only sigma varies.
  4. Simulate every combination of estimator/benchmark x variant x fill rule
     (conservative / optimistic) x gamma in one batch: decisions every
     100 ms, 50 ms latency, tick rounding, post-only quotes. Deterministic.
  5. Score hours 03:00-24:00 separately: P&L, fills, notional (for fees
     afterwards), spread, inventory, QLIKE, k in force, regime features.
  6. Write output/walkforward_event/blocks_<date>.csv, delete the raw CSVs.

Resumable: days with results are skipped unless --force.
Usage (from src/): python -m scripts.run_walkforward [--workers 3] [--dates D1 D2 ...] [--force]
"""
import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from arch.utility.exceptions import ConvergenceWarning

from mmvol.config import (
    BENCHMARKS, CLIP_SIZE, ESTIMATORS, FIRST_FIT_SEC, KEEP_RAW, OUT, RAW, REFIT_SEC, ROLLING_HORIZON_SEC, ROOT,
    SYMBOL, TRAIN_SEC, VARIANTS,
)
from mmvol.simulator import calibrate_k_walkforward, prepare_intervals, run_event_sim
from mmvol.walkforward import (
    DECISION_MS, GAMMAS, LATENCY_MS, RULES, build_benchmarks, build_sigmas, decision_sigmas, load_day,
    score_blocks,
)

WF_DIR = OUT / "walkforward_event"

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def process_day(date: str, keep_raw: bool) -> dict:
    t_start = time.time()
    book, trades, mid = load_day(date)
    sigmas, fit_diag = build_sigmas(mid)
    sigmas.update(build_benchmarks(mid))

    iv = prepare_intervals(book, trades, DECISION_MS, LATENCY_MS)
    k_dec, k_fits = calibrate_k_walkforward(iv, DECISION_MS, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
    sig_dec = decision_sigmas(sigmas, iv["dec_t"])
    del book  # largest object; everything needed from it is in iv / mid

    batch = [(v, e, rule, g) for (v, e) in sigmas for rule in RULES for g in GAMMAS]
    sim = run_event_sim(iv, trades, mid.values, np.vstack([sig_dec[(v, e)] for (v, e, _, _) in batch]),
                        np.array([rule == "optimistic" for (_, _, rule, _) in batch]),
                        np.array([g for (_, _, _, g) in batch]), k_dec, ROLLING_HORIZON_SEC, CLIP_SIZE,
                        DECISION_MS)
    labels = [{"variant": v, "estimator": e, "rule": rule, "gamma": g} for (v, e, rule, g) in batch]
    rows = score_blocks(date, mid, sigmas, sim, labels, k_dec[::sim["per_sec"]])

    WF_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WF_DIR / f"blocks_{date}.csv", index=False)
    diag = {"date": date, "seconds": round(time.time() - t_start, 1), "k_median": float(np.median(k_fits[:, 1])),
            "k_min": float(k_fits[:, 1].min()), "k_max": float(k_fits[:, 1].max()),
            "rejected_mean": float(sim["rejected"].mean()), **fit_diag}
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
    n_runs = (len(ESTIMATORS) * len(VARIANTS) + len(BENCHMARKS)) * len(RULES) * len(GAMMAS)
    print(f"{len(dates)} days, {len(dates) - len(todo)} already done, {len(todo)} to run "
          f"({args.workers} workers; {n_runs} runs/day)", flush=True)

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
