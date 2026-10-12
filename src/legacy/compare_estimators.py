"""
Compare volatility estimators through the SAME AS harness, on the same day,
with gamma/k/horizon/clip size all held fixed (from pipeline.calibrate) --
only the sigma_t series driving the quotes differs.

The set follows the literature review in paper/literature_notes.md:

  RV      realized_vol   rolling mean of squared 1s returns, 300s window.
          The literature benchmark (Liu, Patton & Sheppard 2015) and what
          Hummingbot's Avellaneda-Stoikov implementation uses.
  BV      bipower_vol    rolling |r_t||r_{t-1}|, 300s window. Jump-robust
          (Barndorff-Nielsen & Shephard 2004).
  EWMA    ewma_vol       RiskMetrics recursion, lambda chosen so its center
          of mass matches RV's window (RiskMetrics' 0.94 is for daily data).
  GARCH   garch_vol      GARCH(1,1) (Bollerslev 1986), fit once on a 3h
          burn-in, run forward causally. Hansen & Lunde (2005) benchmark.
  HAR     har_vol        HAR-RV (Corsi 2009) with 1/5/30-minute components,
          forecasting the next 300s; fit once on the same burn-in by OLS.

Two variants of every estimator:
  raw      as estimated.
  matched  rescaled by one constant so its average variance over the burn-in
           equals the realized variance over the burn-in (causal: the factor
           uses burn-in data only). An estimator that runs systematically low
           otherwise behaves like a less risk-averse market maker -- tighter
           quotes, more fills -- whether or not it tracks volatility well.
           Matching removes that, so P&L differences reflect how well each
           estimator tracks CHANGES in volatility.

Each (variant, estimator) runs over several seeds of the fill simulation.
Within a seed every estimator gets the same random fill draws at each second
(common random numbers), so paired differences between estimators aren't
just different luck.

Scores, all computed after the burn-in (GARCH/HAR are in-sample before it):
  qlike        QLIKE forecast loss (Patton 2011) of the per-second variance
               forecast against variance realized over the next 300s.
               Deterministic: no fill randomness involved.
  pnl_post     mark-to-market P&L after the burn-in, in USD.
  pnl_per_vol  pnl_post / (clip size * dollar volatility of the mid over the
               same period). P&L in units of the day's price movement, so
               calm and volatile days can be compared.
  sharpe       mean / std of 5-minute P&L changes, times sqrt(#changes).
               Risk-adjusted and dimensionless.
  edge_bps     pnl_post / notional traded, in basis points.

Usage (from src/): python -m legacy.compare_estimators [--date YYYY-MM-DD] [--seeds N]
"""
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from legacy.pipeline_1s import (
    calibrate, load_book_1s, load_trade_fill_signals, plot_run, run_as_sim, solve_fill_params,
    CLIP_SIZE, DEFAULT_DATE, K, OUT, PALETTE, ROLLING_HORIZON_SEC, SECONDS_PER_YEAR, SYMBOL,
)
from mmvol.estimators import (
    bipower_vol, ewma_lambda_matching_window, ewma_vol, fit_garch_params, fit_har_params,
    garch_vol, har_vol, realized_vol,
)

BURN_IN_SEC = 3 * 3600  # GARCH and HAR are fit on this window; scores below are computed after it
SEED_BASE = 1000
PNL_BUCKET_SEC = 300

EST_COLORS = {"RV": PALETTE["mid"], "BV": PALETTE["ask"], "EWMA": "#7b4fd6",
              "GARCH": PALETTE["bid"], "HAR": "#d6336c"}


def ann_vol(sigma_t, mid_1s):
    return sigma_t / mid_1s.values * math.sqrt(SECONDS_PER_YEAR)


def _r2(mid_1s: pd.Series) -> pd.Series:
    return np.log(mid_1s / mid_1s.shift(1)).fillna(0.0) ** 2


def level_match(sigma_t: np.ndarray, mid_1s: pd.Series, burn_in: int):
    ref_var = _r2(mid_1s).values[1:burn_in].mean()
    est_var = ((sigma_t / mid_1s.values) ** 2)[1:burn_in].mean()
    factor = math.sqrt(ref_var / est_var)
    return sigma_t * factor, factor


def forecast_scores(sigma_t: np.ndarray, mid_1s: pd.Series, start: int, horizon: int) -> dict:
    """QLIKE of the per-second variance forecast h_t against the per-second
    variance realized over (t, t+horizon]. Lower is better. Seconds where the
    forecast or the realized proxy is exactly zero are excluded and counted
    (QLIKE is undefined there; on calm days a 300s window can have no mid
    move at all)."""
    h = (sigma_t / mid_1s.values) ** 2
    r2 = _r2(mid_1s)
    y = r2[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-1).values
    t = np.arange(start, len(h) - horizon)
    h, y = h[t], y[t]
    valid = (h > 0) & (y > 0)
    ratio = y[valid] / h[valid]
    return {
        "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
        "fcst_bias": float(h[valid].mean() / y[valid].mean()),  # >1: forecasts too high on average
        "excluded_s": int((~valid).sum()),
    }


def pnl_metrics(df: pd.DataFrame, start: int) -> dict:
    pnl = df["pnl"].values
    mid = df["mid"].values
    pnl_post = pnl[-1] - pnl[start]

    idx = np.unique(np.r_[np.arange(start, len(pnl), PNL_BUCKET_SEC), len(pnl) - 1])
    d = np.diff(pnl[idx])
    sharpe = d.mean() / d.std(ddof=1) * math.sqrt(len(d)) if d.std(ddof=1) > 0 else float("nan")

    dollar_vol = math.sqrt(np.sum(np.diff(mid[start:]) ** 2))

    fb, fa = df["filled_bid"].values[start:], df["filled_ask"].values[start:]
    notional = CLIP_SIZE * (df["bid"].values[start:][fb].sum() + df["ask"].values[start:][fa].sum())

    return {
        "pnl_post": pnl_post,
        "pnl_per_vol": pnl_post / (CLIP_SIZE * dollar_vol),
        "sharpe": sharpe,
        "edge_bps": pnl_post / notional * 1e4 if notional > 0 else float("nan"),
        "fills_post": int(fb.sum() + fa.sum()),
        "spread_mean": float((df["ask"] - df["bid"]).values[start:].mean()),
    }


def comparison_plot(results: dict, sigma_guess: float, mid_1s: pd.Series, date: str, title_note: str,
                     out_slug: str):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle(f"Estimator comparison, {title_note}  ({SYMBOL} {date}, same gamma/k/horizon)",
                 fontsize=11, color=PALETTE["ink"])

    ax = axes[0]
    guess_ann = sigma_guess / mid_1s.values * math.sqrt(SECONDS_PER_YEAR)
    ax.plot(mid_1s.index, guess_ann, color=PALETTE["axis"], lw=1.0, ls="--", label="constant guess (55%)")
    for name, df in results.items():
        ax.plot(df.index, ann_vol(df["sigma"].values, mid_1s), color=EST_COLORS[name], lw=0.8, label=name, alpha=0.85)
    ax.axvline(mid_1s.index[BURN_IN_SEC], color=PALETTE["ink2"], lw=0.8, ls=":")
    ax.set_ylabel("sigma, annualized-\nequivalent vol")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # Scale to the 99.9th percentile so a few spikes don't flatten the rest; note the clip when it bites.
    vols = {name: ann_vol(df["sigma"].values, mid_1s) for name, df in results.items()}
    y_top = 1.1 * max(np.percentile(v, 99.9) for v in vols.values())
    ax.set_ylim(0, y_top)
    if any(v.max() > y_top for v in vols.values()):
        clip_note = f"y-axis clipped at {y_top:.0%} -- true max: " + ", ".join(
            f"{n} {v.max():.0%}" for n, v in vols.items())
        ax.set_title(clip_note, loc="right", fontsize=7.5, color=PALETTE["ink2"])
    ax.legend(loc="upper left", ncol=6, fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    for name, df in results.items():
        ax.plot(df.index, df["pnl"], color=EST_COLORS[name], lw=1.1, label=name)
    ax.axhline(0, color=PALETTE["axis"], lw=0.8)
    ax.axvline(mid_1s.index[BURN_IN_SEC], color=PALETTE["ink2"], lw=0.8, ls=":")
    ax.text(mid_1s.index[BURN_IN_SEC], ax.get_ylim()[1], " end of GARCH/HAR burn-in", va="top",
            fontsize=7.5, color=PALETTE["ink2"])
    ax.set_ylabel("PnL, mark-to-market (USD)\nseed 0")
    ax.set_xlabel("time (UTC)")
    ax.legend(loc="upper left", ncol=5, fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT / f"{out_slug}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] {out_path}")


def seed_spread_plot(runs: pd.DataFrame, date: str, out_slug: str):
    """P&L after burn-in for every seed, per estimator, raw vs matched."""
    variants = ["raw", "matched"]
    names = list(EST_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, variant in zip(axes, variants):
        sub = runs[runs["variant"] == variant]
        for i, name in enumerate(names):
            vals = sub[sub["estimator"] == name]["pnl_post"].values
            jitter = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(i + jitter, vals, s=14, color=EST_COLORS[name], alpha=0.75, zorder=3)
            ax.hlines(vals.mean(), i - 0.28, i + 0.28, color=PALETTE["ink"], lw=1.6, zorder=4)
        ax.axhline(0, color=PALETTE["axis"], lw=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(f"{variant} sigma", fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("P&L after burn-in (USD)\none dot per seed, bar = mean")
    fig.suptitle(f"Fill-simulation noise vs estimator differences ({SYMBOL} {date}, "
                 f"{runs['seed'].nunique()} seeds, common random numbers)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT / f"{out_slug}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] {out_path}")


def print_summary(runs: pd.DataFrame, fcst: pd.DataFrame, factors: dict):
    for variant in ["raw", "matched"]:
        sub = runs[runs["variant"] == variant]
        rv = sub[sub["estimator"] == "RV"].set_index("seed")["pnl_post"]
        rows = []
        for name in EST_COLORS:
            e = sub[sub["estimator"] == name]
            diff = e.set_index("seed")["pnl_post"] - rv
            f = fcst[(fcst["variant"] == variant) & (fcst["estimator"] == name)].iloc[0]
            rows.append({
                "estimator": name,
                "scale": f"{factors[name]:.2f}" if variant == "matched" else "1.00",
                "qlike": f"{f['qlike']:.3f}",
                "fcst_bias": f"{f['fcst_bias']:.2f}x",
                "pnl_post": f"{e['pnl_post'].mean():7.2f} ± {e['pnl_post'].std(ddof=1):5.2f}",
                "vs_RV (paired)": "--" if name == "RV" else
                    f"{diff.mean():+7.2f} ± {diff.std(ddof=1):5.2f} ({(diff > 0).sum()}/{len(diff)} better)",
                "pnl_per_vol": f"{e['pnl_per_vol'].mean():.3f}",
                "sharpe": f"{e['sharpe'].mean():.2f}",
                "edge_bps": f"{e['edge_bps'].mean():.3f}",
                "fills": f"{e['fills_post'].mean():.0f}",
                "spread_$": f"{e['spread_mean'].mean():.2f}",
            })
        print(f"\n=== {variant.upper()} sigma  (scores after burn-in; pnl = mean ± std across seeds; "
              f"QLIKE lower is better) ===")
        print(pd.DataFrame(rows).set_index("estimator").to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=DEFAULT_DATE)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    date = args.date
    suffix = "" if date == DEFAULT_DATE else f"_{date}"

    mid_1s, bid_1s, ask_1s = load_book_1s(date)
    sell_min, buy_max = load_trade_fill_signals(mid_1s.index, date)
    sigma_guess, gamma_rolling = calibrate(mid_1s)
    fill_A, fill_k = solve_fill_params()

    print("\n--- fitting/estimating sigma_t for each method ---")
    lam = ewma_lambda_matching_window(ROLLING_HORIZON_SEC)
    print(f"  [ewma] lambda={lam:.5f} (center of mass matches the {ROLLING_HORIZON_SEC}s RV window)")
    omega, alpha, beta, scale = fit_garch_params(mid_1s, burn_in_sec=BURN_IN_SEC)
    har_coefs, har_floor = fit_har_params(mid_1s, burn_in_sec=BURN_IN_SEC, horizon_sec=ROLLING_HORIZON_SEC)

    raw = {
        "RV": realized_vol(mid_1s, window_sec=ROLLING_HORIZON_SEC, fallback_sigma=sigma_guess),
        "BV": bipower_vol(mid_1s, window_sec=ROLLING_HORIZON_SEC, fallback_sigma=sigma_guess),
        "EWMA": ewma_vol(mid_1s, lam=lam, fallback_sigma=sigma_guess),
        "GARCH": garch_vol(mid_1s, omega, alpha, beta, scale),
        "HAR": har_vol(mid_1s, har_coefs, har_floor),
    }
    matched, factors = {}, {}
    for name, s in raw.items():
        matched[name], factors[name] = level_match(s, mid_1s, BURN_IN_SEC)
    print("  [level-match] scale factors (sigma multiplier so burn-in variance matches realized): "
          + "  ".join(f"{n}={f:.3f}" for n, f in factors.items()))
    variants = {"raw": raw, "matched": matched}

    fcst_rows = []
    for variant, sigmas in variants.items():
        for name, s in sigmas.items():
            fcst_rows.append({"variant": variant, "estimator": name,
                              **forecast_scores(s, mid_1s, start=BURN_IN_SEC, horizon=ROLLING_HORIZON_SEC)})
    fcst = pd.DataFrame(fcst_rows)

    n = len(mid_1s)
    run_rows = []
    seed0 = {}
    print(f"\n--- simulating {len(variants)} variants x {len(raw)} estimators x {args.seeds} seeds ---")
    for seed in range(args.seeds):
        uniforms = np.random.default_rng(SEED_BASE + seed).random((n, 2))
        for variant, sigmas in variants.items():
            for name, s in sigmas.items():
                df = run_as_sim(mid_1s, sell_min, buy_max, gamma=gamma_rolling, k=K, sigma=s,
                                 horizon_sec=ROLLING_HORIZON_SEC, clip_size=CLIP_SIZE, label=name,
                                 fill_mode="intensity", best_bid=bid_1s, best_ask=ask_1s,
                                 fill_A=fill_A, fill_k=fill_k, uniforms=uniforms, verbose=False)
                run_rows.append({"date": date, "variant": variant, "estimator": name, "seed": seed,
                                 **pnl_metrics(df, BURN_IN_SEC)})
                if seed == 0:
                    seed0[(variant, name)] = df
        print(f"  seed {seed + 1}/{args.seeds} done")
    runs = pd.DataFrame(run_rows)

    results_path = OUT / f"results{suffix or '_' + date}.csv"
    runs.merge(fcst, on=["variant", "estimator"]).to_csv(results_path, index=False)
    print(f"[csv] {results_path}")

    for variant, note in [("raw", "raw sigma"), ("matched", "level-matched sigma")]:
        res = {name: seed0[(variant, name)] for name in raw}
        slug = "estimator_comparison" + ("" if variant == "raw" else "_matched") + suffix
        comparison_plot(res, sigma_guess, mid_1s, date, note, slug)
    for name in raw:
        plot_run(seed0[("matched", name)], f"estimator={name}, level-matched, seed 0",
                 f"cmp_{name.lower()}{suffix}", zoom_start="12:00:00", date=date)
    seed_spread_plot(runs, date, f"seed_spread{suffix}")

    print_summary(runs, fcst, factors)
    print("\nDone.")


if __name__ == "__main__":
    main()
