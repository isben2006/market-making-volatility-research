"""
Analyze the walk-forward results (output/walkforward/blocks_*.csv).

Regimes: each scored hour gets a label from its TRAILING 3-hour realized vol
(60s returns, data before the hour only), split into terciles across every
hour in the study: low / mid / high. The terciles are computed over the whole
study, which is fine here: the label only groups results afterwards, and
the market maker never sees it.

Unit of inference: day x regime. Hours within a day share one price path and
carry inventory from one to the next, so they aren't independent; days are
much closer to it. Results are averaged over seeds first (seeds only vary the
fill simulation), then summed within each day x regime:
  pnl_per_vol  P&L / (clip size x dollar vol of the mid over those hours)
  qlike        mean of the hourly QLIKE values

Tests, within each regime, across days:
  - paired difference of each estimator vs RV, t-statistic and 95% CI.
  - Model Confidence Set (Hansen, Lunde & Nason 2011, via arch), at 10%:
    the set of estimators that can't be statistically distinguished from the
    best. Run on QLIKE, and on -pnl_per_vol as a loss.

Usage: python analyze_walkforward.py [--variant matched]
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

from pipeline import CLIP_SIZE, OUT, PALETTE, SYMBOL

WF_DIR = OUT / "walkforward"
REGIMES = ["low", "mid", "high"]
ESTIMATORS = ["RV", "EWMA", "GARCH", "HAR"]
EST_COLORS = {"RV": PALETTE["mid"], "EWMA": "#7b4fd6", "GARCH": PALETTE["bid"], "HAR": "#d6336c"}
MCS_SIZE = 0.10


def load_blocks() -> pd.DataFrame:
    files = sorted(WF_DIR.glob("blocks_*.csv"))
    if not files:
        raise SystemExit(f"no results in {WF_DIR}; run walkforward.py first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def label_regimes(blocks: pd.DataFrame):
    hours = blocks.drop_duplicates(["date", "hour"])[["date", "hour", "trailing_vol"]]
    cuts = hours["trailing_vol"].quantile([1 / 3, 2 / 3]).values
    hours = hours.assign(regime=pd.cut(hours["trailing_vol"], [-np.inf, *cuts, np.inf], labels=REGIMES))
    return blocks.merge(hours[["date", "hour", "regime"]], on=["date", "hour"]), cuts


def day_regime_units(blocks: pd.DataFrame, variant: str) -> pd.DataFrame:
    b = blocks[blocks["variant"] == variant]
    per_hour = (b.groupby(["date", "hour", "regime", "estimator"], observed=True)
                  .agg(pnl=("pnl", "mean"), qlike=("qlike", "first"), fills=("fills", "mean"),
                       spread=("spread_mean", "mean"), dollar_vol=("dollar_vol", "first"),
                       inv_abs=("inv_abs_mean", "mean"))
                  .reset_index())
    per_hour["dv2"] = per_hour["dollar_vol"] ** 2
    units = (per_hour.groupby(["date", "regime", "estimator"], observed=True)
                     .agg(pnl=("pnl", "sum"), dv2=("dv2", "sum"), qlike=("qlike", "mean"),
                          fills=("fills", "sum"), hours=("hour", "count"), spread=("spread", "mean"),
                          inv_abs=("inv_abs", "mean"))
                     .reset_index())
    units["pnl_per_vol"] = units["pnl"] / (CLIP_SIZE * np.sqrt(units["dv2"]))
    return units


def paired_vs_rv(units: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        wide = units[units["regime"] == regime].pivot(index="date", columns="estimator", values=metric)
        for e in ESTIMATORS:
            d = (wide[e] - wide["RV"]).dropna()
            se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
            better = (d < 0) if metric == "qlike" else (d > 0)
            med_lo, med_hi = _bootstrap_median_ci(d.values) if e != "RV" else (0.0, 0.0)
            rows.append({"regime": regime, "estimator": e, "metric": metric, "n_days": len(d),
                         "mean": wide[e].mean(), "median": wide[e].median(), "diff_vs_rv": d.mean(), "se": se,
                         "t": d.mean() / se if se and se > 0 else np.nan, "days_better": int(better.sum()),
                         "median_diff": d.median(), "median_diff_lo": med_lo, "median_diff_hi": med_hi})
    return pd.DataFrame(rows)


def _bootstrap_median_ci(d: np.ndarray, reps: int = 5000, seed: int = 7):
    rng = np.random.default_rng(seed)
    meds = np.median(d[rng.integers(0, len(d), size=(reps, len(d)))], axis=1)
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def mcs_members(units: pd.DataFrame, metric: str, sign: float, seed: int = 7) -> dict:
    """Estimators in the 10% Model Confidence Set per regime (loss = sign * metric)."""
    out = {}
    for regime in REGIMES:
        wide = units[units["regime"] == regime].pivot(index="date", columns="estimator", values=metric)
        losses = (sign * wide[ESTIMATORS]).dropna()
        mcs = MCS(losses, size=MCS_SIZE, reps=10000, block_size=1, method="R", seed=seed)
        mcs.compute()
        out[regime] = set(mcs.included)
    return out


def diff_plot(tests: pd.DataFrame, metric: str, ylabel: str, better: str, ax, use_median: bool):
    others = [e for e in ESTIMATORS if e != "RV"]
    width = 0.22
    for i, e in enumerate(others):
        sub = tests[(tests["metric"] == metric) & (tests["estimator"] == e)].set_index("regime").loc[REGIMES]
        x = np.arange(len(REGIMES)) + (i - 1) * width
        if use_median:
            y = sub["median_diff"]
            yerr = [y - sub["median_diff_lo"], sub["median_diff_hi"] - y]
        else:
            y, yerr = sub["diff_vs_rv"], 1.96 * sub["se"]
        ax.errorbar(x, y, yerr=yerr, fmt="o", ms=6, capsize=3, color=EST_COLORS[e], label=e, lw=1.4)
    ax.axhline(0, color=PALETTE["ink2"], lw=0.9)
    ax.set_xticks(range(len(REGIMES)))
    n_days = tests[tests["metric"] == metric].groupby("regime")["n_days"].first()
    ax.set_xticklabels([f"{r} vol\n({n_days[r]} days)" for r in REGIMES])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{better}", fontsize=9, color=PALETTE["ink2"], loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="matched", choices=["matched", "raw"])
    variant = ap.parse_args().variant

    blocks, cuts = label_regimes(load_blocks())
    n_days = blocks["date"].nunique()
    print(f"{n_days} days, {blocks.drop_duplicates(['date', 'hour']).shape[0]} scored hours, "
          f"{blocks['seed'].nunique()} seeds; variant={variant}")
    print(f"regime cutoffs (trailing 3h vol, annualized): low < {cuts[0]:.0%} <= mid < {cuts[1]:.0%} <= high")

    units = day_regime_units(blocks, variant)
    tests = pd.concat([paired_vs_rv(units, "qlike"), paired_vs_rv(units, "pnl_per_vol"),
                       paired_vs_rv(units, "pnl")])
    mcs_q = mcs_members(units, "qlike", +1.0)
    mcs_p = mcs_members(units, "pnl_per_vol", -1.0)

    rows = []
    for regime in REGIMES:
        for e in ESTIMATORS:
            q = tests[(tests.regime == regime) & (tests.estimator == e) & (tests.metric == "qlike")].iloc[0]
            p = tests[(tests.regime == regime) & (tests.estimator == e) & (tests.metric == "pnl_per_vol")].iloc[0]
            usd = tests[(tests.regime == regime) & (tests.estimator == e) & (tests.metric == "pnl")].iloc[0]
            rows.append({
                "regime": regime, "estimator": e, "days": q["n_days"],
                "qlike_median": q["median"], "qlike_vs_rv": q["diff_vs_rv"], "qlike_t": q["t"],
                "qlike_days_better": q["days_better"], "qlike_in_mcs": e in mcs_q[regime],
                "ppv_mean": p["mean"], "ppv_vs_rv": p["diff_vs_rv"], "ppv_t": p["t"],
                "ppv_days_better": p["days_better"], "ppv_in_mcs": e in mcs_p[regime],
                "pnl_usd_per_day": usd["mean"],
            })
    summary = pd.DataFrame(rows)
    out_csv = WF_DIR / f"summary_{variant}.csv"
    summary.to_csv(out_csv, index=False, float_format="%.4f")

    pd.set_option("display.width", 220)
    fmt = summary.copy()
    for c in ["qlike_median", "ppv_mean"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.3f}")
    for c in ["qlike_vs_rv", "ppv_vs_rv"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:+.3f}")
    for c in ["qlike_t", "ppv_t"]:
        fmt[c] = fmt[c].map(lambda v: "--" if pd.isna(v) else f"{v:+.1f}")
    for c, n in [("qlike_days_better", "days"), ("ppv_days_better", "days")]:
        fmt[c] = [("--" if est == "RV" else f"{b}/{d}") for est, b, d in zip(fmt["estimator"], fmt[c], fmt[n])]
    fmt["pnl_usd_per_day"] = fmt["pnl_usd_per_day"].map(lambda v: f"{v:.1f}")
    print("\nQLIKE: lower is better (median across days shown; means are outlier-prone). "
          "ppv = P&L per unit vol: higher is better.\n*_vs_rv: paired mean difference vs RV across days; "
          "*_t: its t-stat; *_days_better: days it beat RV; *_in_mcs: in the 10% Model Confidence Set.")
    print(fmt.to_string(index=False))
    print(f"[csv] {out_csv}")

    # sanity: how the regime label lines up with hour of day and with the day-level tercile
    hours = blocks.drop_duplicates(["date", "hour"])
    by_hour = pd.crosstab(hours["hour"], hours["regime"], normalize="index").round(2)
    print("\nshare of each regime by hour of day (UTC), to check for time-of-day confounding:")
    print(by_hour.T.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    diff_plot(tests, "qlike", "QLIKE minus RV's QLIKE\n(median across days, bootstrap 95% CI)",
              "Forecast loss vs RV (below 0 = forecasts better)", axes[0], use_median=True)
    diff_plot(tests, "pnl_per_vol", "P&L per unit vol minus RV's\n(mean across days, 95% CI)",
              "Market-making P&L vs RV (above 0 = earns more)", axes[1], use_median=False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Walk-forward: each estimator vs realized vol, by regime ({SYMBOL}, {n_days} days, "
                 f"{variant} sigma, paired differences across days)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = WF_DIR / f"regime_comparison_{variant}.png"
    fig.savefig(out_png, dpi=130)
    print(f"[plot] {out_png}")


if __name__ == "__main__":
    main()
