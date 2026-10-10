"""
Compare three volatility estimators through the SAME AS harness, on the same
day, with gamma/k/horizon/clip size all held fixed (from pipeline.calibrate) --
only the sigma_t series driving the quotes differs. Apples to apples.

  RV      realized_vol      rolling sum-of-squared-returns, 300s window.
          (this is run 3 from as_skeleton.py -- the simplest possible
          estimator, and the one shown there to be sensitive to single-tick
          outliers dominating a short window.)
  BV      bipower_vol       rolling |r_t||r_{t-1}| sum, 300s window. Built
          specifically to answer RV's outlier-sensitivity with a jump-robust
          alternative -- same window, same everything else.
  GARCH   garch_vol         exponentially-weighted recursion, parameters
          fit once on a 3h burn-in, run forward causally. Different flavor
          entirely: persistence/memory instead of a flat window.

Fills use the same Poisson fill-intensity model as as_skeleton.py's RUN 3
(not the deterministic trade-through rule from runs 1-2) -- see
pipeline.run_as_sim's fill_mode docstring.

Run this after as_skeleton.py (needs its RUN 1/2 diagnostics to make sense of
why gamma_rolling is calibrated the way it is).

Usage: python compare_estimators.py [--date YYYY-MM-DD] (defaults to 2024-01-10).
"""
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline import (
    calibrate, load_book_1s, load_trade_fill_signals, plot_run, run_as_sim, solve_fill_params,
    CLIP_SIZE, DEFAULT_DATE, K, OUT, PALETTE, ROLLING_HORIZON_SEC, SECONDS_PER_YEAR, SYMBOL,
)
from estimators import realized_vol, bipower_vol, fit_garch_params, garch_vol

GARCH_BURN_IN_SEC = 3 * 3600  # first 3 hours, fit once, offline -- see estimators.py docstring

EST_COLORS = {"RV": PALETTE["mid"], "BV": PALETTE["ask"], "GARCH": PALETTE["bid"]}


def ann_vol(sigma_t, mid_1s):
    return sigma_t / mid_1s.values * math.sqrt(SECONDS_PER_YEAR)


def comparison_plot(results: dict, sigma_guess: float, mid_1s: pd.Series, date: str, out_slug: str):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle(f"Estimator comparison -- RV vs BV vs GARCH(1,1)  ({SYMBOL} {date}, same gamma/k/horizon)",
                 fontsize=11, color=PALETTE["ink"])

    ax = axes[0]
    guess_ann = sigma_guess / mid_1s.values * math.sqrt(SECONDS_PER_YEAR)
    ax.plot(mid_1s.index, guess_ann, color=PALETTE["axis"], lw=1.0, ls="--", label="constant guess (55%)")
    for name, df in results.items():
        ax.plot(df.index, ann_vol(df["sigma"].values, mid_1s), color=EST_COLORS[name], lw=0.9, label=name, alpha=0.9)
    ax.set_ylabel("sigma, annualized-\nequivalent vol")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 4.0)  # clip the view to keep the comparison readable; printed stats below have the true tails
    true_maxes = {name: ann_vol(df["sigma"].values, mid_1s).max() for name, df in results.items()}
    clip_note = "y-axis clipped at 400% -- true max: " + ", ".join(f"{n} {v:.0%}" for n, v in true_maxes.items())
    ax.text(0.99, 0.02, clip_note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=PALETTE["ink2"])
    ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    for name, df in results.items():
        ax.plot(df.index, df["pnl"], color=EST_COLORS[name], lw=1.1, label=name)
    ax.axhline(0, color=PALETTE["axis"], lw=0.8)
    ax.set_ylabel("PnL, mark-to-market (USD)")
    ax.set_xlabel("time (UTC)")
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT / f"{out_slug}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n[plot] {out_path}")


def print_summary_table(results: dict, mid_1s: pd.Series):
    rows = []
    for name, df in results.items():
        av = ann_vol(df["sigma"].values, mid_1s)
        spread = df["ask"] - df["bid"]
        n_fills = int(df["filled_bid"].sum() + df["filled_ask"].sum())
        rows.append({
            "estimator": name,
            "vol_min": f"{av.min():.0%}", "vol_med": f"{np.median(av):.0%}", "vol_max": f"{av.max():.0%}",
            "spread_mean_$": f"{spread.mean():.2f}", "spread_max_$": f"{spread.max():.2f}",
            "fills": n_fills, "final_inv_BTC": f"{df['inventory'].iloc[-1]:.3f}",
            "final_pnl_$": f"{df['pnl'].iloc[-1]:.2f}",
        })
    table = pd.DataFrame(rows).set_index("estimator")
    print("\n=== Estimator comparison summary ===")
    print(table.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=DEFAULT_DATE)
    date = ap.parse_args().date
    suffix = "" if date == DEFAULT_DATE else f"_{date}"

    mid_1s, bid_1s, ask_1s = load_book_1s(date)
    sell_min, buy_max = load_trade_fill_signals(mid_1s.index, date)
    sigma_guess, gamma_rolling = calibrate(mid_1s)
    fill_A, fill_k = solve_fill_params()

    print("\n--- fitting/estimating sigma_t for each method ---")
    sigma_rv = realized_vol(mid_1s, window_sec=ROLLING_HORIZON_SEC, fallback_sigma=sigma_guess)
    sigma_bv = bipower_vol(mid_1s, window_sec=ROLLING_HORIZON_SEC, fallback_sigma=sigma_guess)
    omega, alpha, beta, scale = fit_garch_params(mid_1s, burn_in_sec=GARCH_BURN_IN_SEC)
    sigma_garch = garch_vol(mid_1s, omega, alpha, beta, scale)

    estimators = {"RV": sigma_rv, "BV": sigma_bv, "GARCH": sigma_garch}
    results = {}
    for name, sigma_t in estimators.items():
        df = run_as_sim(mid_1s, sell_min, buy_max, gamma=gamma_rolling, k=K, sigma=sigma_t,
                         horizon_sec=ROLLING_HORIZON_SEC, clip_size=CLIP_SIZE,
                         label=f"{name} (same gamma/horizon, fill-intensity model)",
                         fill_mode="intensity", best_bid=bid_1s, best_ask=ask_1s, fill_A=fill_A, fill_k=fill_k)
        plot_run(df, f"estimator={name}", f"cmp_{name.lower()}{suffix}", zoom_start="12:00:00", date=date)
        results[name] = df

    comparison_plot(results, sigma_guess, mid_1s, date, out_slug=f"estimator_comparison{suffix}")
    print_summary_table(results, mid_1s)
    print("\nDone. PNGs in output/.")


if __name__ == "__main__":
    main()
