"""
Ugly, minimal, end-to-end Avellaneda-Stoikov skeleton on one day of real data.

Goal (per the plan): get the WHOLE pipeline running before touching anything
estimator-related, so pipeline bugs (timestamp alignment, formula sanity at
crypto tick/price scale, fill-rule semantics, event-loop structure) surface
cheap, on one day of data with a fake constant sigma -- before four estimators
and a regime taxonomy get built on top of a skeleton that turns out to be wrong.

Data: BTCUSDT USDS-M futures, 2024-01-10 (see download_data.py for why this
symbol/date/venue -- short version: it's the only combination where Binance's
free archives have both bookTicker (L1) and aggTrades for the same day).

This script runs the sim three times (see pipeline.py for the shared loader /
AS loop / plotting, estimators.py for the realized-vol estimator used in run 3):
  RUN 1 (naive / literal):  T = the whole trading day, treated as one horizon,
      exactly like the classic Avellaneda-Stoikov (2008) textbook formula.
      Expected to misbehave -- see the printed diagnostic. Left in deliberately:
      it's the cheapest possible way to learn that a literal 24h AS horizon is
      unusable at BTC's price level, instead of only shipping the fixed version.
  RUN 2 (rolling horizon):  same guessed CONSTANT sigma, but T resets every
      ROLLING_HORIZON_SEC and gamma is picked (still just arithmetic, not an
      estimator) to target a plausible dollar spread at BTC's price level.
  RUN 3 (causal realized-vol estimator + fill-intensity model): sigma becomes
      a rolling realized-vol estimate computed causally (no lookahead)
      instead of a constant guess -- the first real estimator plugged into
      the harness. Also switches the fill rule from deterministic
      trade-through (runs 1-2, kept as historical baselines) to the
      Poisson fill-intensity model -- see pipeline.run_as_sim's fill_mode
      docstring for why that's a separate, necessary fix.

Estimator comparison (realized vol vs. bipower variation vs. GARCH(1,1)) lives
in compare_estimators.py, built after this skeleton was confirmed to work.

Usage (from src/): python -m legacy.as_skeleton [--date YYYY-MM-DD]  (defaults to 2024-01-10, the
day this was originally built against). Output filenames/titles for any other
date are suffixed with that date so they don't collide with the original run.
"""
import argparse
import math

import numpy as np

from legacy.pipeline_1s import (
    calibrate, load_book_1s, load_trade_fill_signals, plot_run, run_as_sim, solve_fill_params,
    CLIP_SIZE, DAY_SECONDS, DEFAULT_DATE, GAMMA_NAIVE, K, ROLLING_HORIZON_SEC, SIGMA_ANNUAL_GUESS,
)
from mmvol.estimators import realized_vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=DEFAULT_DATE)
    date = ap.parse_args().date
    suffix = "" if date == DEFAULT_DATE else f"_{date}"

    mid_1s, bid_1s, ask_1s = load_book_1s(date)
    sell_min, buy_max = load_trade_fill_signals(mid_1s.index, date)

    sigma, gamma_rolling = calibrate(mid_1s)
    print(f"\n[params] sigma_annual guess={SIGMA_ANNUAL_GUESS:.0%} -> sigma={sigma:.4f} USD/sqrt(s)")

    # Diagnostic BEFORE running anything: what does the inventory-risk term
    # look like at max time-remaining, for each horizon, at BTC's price scale?
    naive_term = GAMMA_NAIVE * sigma**2 * DAY_SECONDS
    print(f"[diagnostic] naive run: gamma*sigma^2*T at t=0 (T=1 day) = ${naive_term:,.0f} "
          f"-- this is the half-spread-ish inventory term alone, i.e. already an unusable quote width")
    rolling_term = gamma_rolling * sigma**2 * ROLLING_HORIZON_SEC
    print(f"[diagnostic] rolling run: gamma back-solved to {gamma_rolling:.6g} so that "
          f"gamma*sigma^2*T at t=0 (T={ROLLING_HORIZON_SEC}s) = ${rolling_term:.2f}")

    # RUN 1: literal textbook AS, T = the whole day. Expected to misbehave.
    run1 = run_as_sim(mid_1s, sell_min, buy_max, gamma=GAMMA_NAIVE, k=K, sigma=sigma,
                       horizon_sec=DAY_SECONDS, clip_size=CLIP_SIZE, label="RUN 1: naive full-day horizon")
    plot_run(run1, "RUN 1: naive full-day horizon (gamma=0.1, textbook)", f"run1_naive{suffix}",
              zoom_start="20:00:00", date=date)

    # RUN 2: rolling 5-minute horizon, gamma back-solved for a sane $ spread at BTC's price level.
    run2 = run_as_sim(mid_1s, sell_min, buy_max, gamma=gamma_rolling, k=K, sigma=sigma,
                       horizon_sec=ROLLING_HORIZON_SEC, clip_size=CLIP_SIZE, label="RUN 2: rolling 5min horizon")
    plot_run(run2, "RUN 2: rolling 5min horizon (gamma back-solved)", f"run2_rolling{suffix}",
              zoom_start="12:00:00", date=date)

    # RUN 3: sigma changes from a constant guess to a causal rolling
    # realized-vol estimate, AND the fill rule changes from deterministic
    # trade-through to the Poisson fill-intensity model.
    sigma_t = realized_vol(mid_1s, window_sec=ROLLING_HORIZON_SEC, fallback_sigma=sigma)
    ann_vol_t = sigma_t / mid_1s.values * math.sqrt(365 * 24 * 3600)
    print(f"\n[params] RUN 3 realized-vol estimator (window={ROLLING_HORIZON_SEC}s, causal, no lookahead): "
          f"annualized-equivalent vol (min/median/max)="
          f"({ann_vol_t.min():.0%}/{np.median(ann_vol_t):.0%}/{ann_vol_t.max():.0%}) "
          f"vs the {SIGMA_ANNUAL_GUESS:.0%} constant guess used in runs 1-2")

    fill_A, fill_k = solve_fill_params()
    print(f"[params] fill-intensity model: A={fill_A:.4f}, k_fill={fill_k:.4f} "
          f"(90% fill chance at the touch, ~10% by $5 of depth)")

    run3 = run_as_sim(mid_1s, sell_min, buy_max, gamma=gamma_rolling, k=K, sigma=sigma_t,
                       horizon_sec=ROLLING_HORIZON_SEC, clip_size=CLIP_SIZE,
                       label="RUN 3: causal realized-vol + fill-intensity model",
                       fill_mode="intensity", best_bid=bid_1s, best_ask=ask_1s, fill_A=fill_A, fill_k=fill_k)
    plot_run(run3, "RUN 3: causal realized-vol + fill-intensity model",
              f"run3_realized_vol{suffix}", zoom_start="12:00:00", date=date)

    print("\nDone. PNGs in output/.")


if __name__ == "__main__":
    main()
