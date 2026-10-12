"""
Robustness checks for the walk-forward study (scripts/run_walkforward.py).

Same 41 days, regimes, estimators and gammas, level-matched sigma and the
conservative fill rule only (the optimistic bound and raw sigma were shown
to give the same conclusions). Each config changes one thing:

  base       100 ms decisions, 50 ms latency, k refit hourly. Should
             reproduce the main run's conservative / level-matched results.
  k_fixed    k fit once per day on the 3h burn-in, then held fixed.
  k_const    k = K_CONST for every day and hour. K_CONST is the median of
             all hourly fits in the main run (so it uses the whole sample;
             it's a fixed parameter, not something the quotes adapt with).
             Hourly k carries volatility information of its own (it's
             ~0.2/$ in volatile hours, up to ~1.2/$ in calm ones), which
             could mask differences between sigma estimators.
  d250, d1000  decisions every 250 ms / 1 s instead of 100 ms.
  lat10, lat200  10 ms / 200 ms latency instead of 50 ms.

Per day writes output/robustness_event/blocks_<date>.csv (per hour, as in
the main run, plus a `config` column) and days_<date>.csv (per run risk
metrics over the scored part of the day: max drawdown, P&L to mean absolute
inventory, a Sharpe-style ratio of 5-minute P&L changes).

Usage (from src/): python -m scripts.run_robustness [--workers 3] [--dates ...] [--force]
"""
import argparse
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from arch.utility.exceptions import ConvergenceWarning

from mmvol.config import (
    CLIP_SIZE, DAY_SECONDS, ESTIMATORS, FIRST_FIT_SEC, KEEP_RAW, OUT, RAW, REFIT_SEC, ROLLING_HORIZON_SEC, ROOT,
    SYMBOL, TRAIN_SEC,
)
from mmvol.simulator import calibrate_k_walkforward, prepare_intervals, run_event_sim
from mmvol.walkforward import GAMMAS, build_sigmas, decision_sigmas, load_day, score_blocks

K_CONST = 0.46
RB_DIR = OUT / "robustness_event"
PNL_BUCKET_SEC = 300
# config name -> (decision_ms, latency_ms); the k variants share the base pass
TIMING_CONFIGS = {"base": (100, 50), "d250": (250, 50), "d1000": (1000, 50), "lat10": (100, 10), "lat200": (100, 200)}
K_MODES = ["base", "k_fixed", "k_const"]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def day_risk(sim: dict, labels: list) -> list:
    pnl = sim["pnl"][FIRST_FIT_SEC - 1:] - sim["pnl"][FIRST_FIT_SEC - 1]
    n_sec = DAY_SECONDS - FIRST_FIT_SEC
    inv_mean = sim["inv_abs_sum"][FIRST_FIT_SEC:].sum(axis=0) / (n_sec * sim["per_sec"])
    steps = pnl[::PNL_BUCKET_SEC]
    d = np.diff(np.r_[steps, pnl[-1:]], axis=0)
    rows = []
    for j, lab in enumerate(labels):
        p = pnl[:, j]
        sd = d[:, j].std(ddof=1)
        rows.append({**lab, "pnl_post": p[-1], "max_drawdown": float(np.max(np.maximum.accumulate(p) - p)),
                     "inv_abs_mean": inv_mean[j], "pnl_to_map": p[-1] / inv_mean[j] if inv_mean[j] > 0 else np.nan,
                     "sharpe_5m": d[:, j].mean() / sd * math.sqrt(len(d)) if sd > 0 else np.nan,
                     "fills": float(sim["fills"][FIRST_FIT_SEC:, j].sum()),
                     "notional": float(sim["notional"][FIRST_FIT_SEC:, j].sum())})
    return rows


def process_day(date: str, keep_raw: bool) -> dict:
    t_start = time.time()
    book, trades, mid = load_day(date)
    sigmas, _ = build_sigmas(mid)
    sigmas = {key: s for key, s in sigmas.items() if key[0] == "matched"}
    ivs = {name: prepare_intervals(book, trades, d_ms, l_ms) for name, (d_ms, l_ms) in TIMING_CONFIGS.items()}
    del book

    block_rows, day_rows = [], []
    for name, (d_ms, l_ms) in TIMING_CONFIGS.items():
        iv = ivs[name]
        k_hourly, k_fits = calibrate_k_walkforward(iv, d_ms, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
        sig_dec = decision_sigmas(sigmas, iv["dec_t"])
        if name == "base":
            k_table = np.column_stack([k_hourly, np.full(iv["n_dec"], k_fits[0, 1]), np.full(iv["n_dec"], K_CONST)])
            batch = [(cfg, e, g) for cfg in K_MODES for e in ESTIMATORS for g in GAMMAS]
            k_idx = np.array([K_MODES.index(cfg) for (cfg, _, _) in batch])
            k_arg = k_table
        else:
            batch = [(name, e, g) for e in ESTIMATORS for g in GAMMAS]
            k_idx, k_arg = None, k_hourly
        sim = run_event_sim(iv, trades, mid.values, np.vstack([sig_dec[("matched", e)] for (_, e, _) in batch]),
                            np.zeros(len(batch), dtype=bool), np.array([g for (_, _, g) in batch]), k_arg,
                            ROLLING_HORIZON_SEC, CLIP_SIZE, d_ms, k_idx=k_idx)
        labels = [{"config": cfg, "variant": "matched", "estimator": e, "gamma": g} for (cfg, e, g) in batch]
        block_rows += score_blocks(date, mid, sigmas, sim, labels, k_hourly[::sim["per_sec"]])
        day_rows += [{"date": date, **r} for r in day_risk(sim, labels)]
        del sim

    RB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(block_rows).to_csv(RB_DIR / f"blocks_{date}.csv", index=False)
    pd.DataFrame(day_rows).to_csv(RB_DIR / f"days_{date}.csv", index=False)
    if not keep_raw and date not in KEEP_RAW:
        for dt in ("bookTicker", "aggTrades"):
            (RAW / f"{SYMBOL}-{dt}-{date}.csv").unlink(missing_ok=True)
    return {"date": date, "seconds": round(time.time() - t_start, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dates = args.dates or pd.read_csv(ROOT / "data" / "study_days.csv")["date"].tolist()
    todo = [d for d in dates if args.force or not (RB_DIR / f"days_{d}.csv").exists()]
    print(f"{len(dates)} days, {len(dates) - len(todo)} already done, {len(todo)} to run ({args.workers} workers)",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_day, d, args.keep_raw): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                print(f"[{i}/{len(todo)}] {d} done in {fut.result()['seconds']}s", flush=True)
            except Exception as exc:
                print(f"[{i}/{len(todo)}] {d} FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
