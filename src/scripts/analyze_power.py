"""
How large an estimator effect can the walk-forward study rule out?

Each estimator vs RV, as paired differences of P&L in $ across days, per
regime x gamma (36 cells, day x regime units as in mmvol.analysis) and over
whole days (12 cells). Per cell:

  ci95         95% confidence interval of the mean difference.
  mde          minimum detectable effect: the smallest true difference the
               study had 80% power to detect, two-sided at 5%, and at the
               Bonferroni level used in the main comparison (5% / 36).
               mde = (t_{1-a/2} + t_{power}) x SE, with t quantiles at n-1 df.
  equiv_bound  the smallest margin D for which a TOST equivalence test at 5%
               passes, i.e. the 90% CI lies inside [-D, +D]:
               D = max(|lo90|, |hi90|). Reporting the bound instead of a
               verdict at one margin avoids choosing the margin after seeing
               the data; compare it to any threshold.

Scale: $ per day x regime unit, and % of RV's spread captured in the same
cell (the market maker's gross revenue; unlike net P&L it's never near zero).
Whole days are also compared to the Oracle benchmark's gain over RV: what
perfect volatility foresight is worth.

Equivalence for all cells at once needs each cell's TOST to pass at 5%; no
multiplicity correction is needed for that joint claim (intersection-union
principle, Berger 1982). Significance claims (the mde at 5% / 36) do need it.

Usage (from src/): python -m scripts.analyze_power [--rule conservative] [--fee-bps 0]
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from mmvol.analysis import REGIMES, label_regimes, study_files, with_fees
from mmvol.config import ESTIMATORS, MAIN_VARIANT, OUT

WF_DIR = OUT / "walkforward_event"
POWER = 0.80
ALPHA = 0.05
N_CELLS = 36  # 3 estimators x 3 regimes x 4 gammas: the main comparison's Bonferroni family


def load(rule: str, fee_bps: float) -> pd.DataFrame:
    files = study_files(WF_DIR)
    blocks = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    blocks = blocks[(blocks["rule"] == rule) & (blocks["variant"].isin([MAIN_VARIANT, "benchmark"]))]
    blocks, _ = label_regimes(blocks)
    blocks = blocks.assign(pnl_net=with_fees(blocks, fee_bps))
    whole = blocks.assign(regime="all")
    both = pd.concat([blocks.astype({"regime": str}), whole], ignore_index=True)
    return (both.groupby(["gamma", "regime", "estimator", "date"])[["pnl_net", "pnl_spread"]].sum()
                .rename(columns={"pnl_net": "pnl", "pnl_spread": "spread"}))


def cell_stats(d: pd.Series, scale: float) -> dict:
    n = len(d)
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t95, t90 = stats.t.ppf(1 - ALPHA / 2, n - 1), stats.t.ppf(1 - ALPHA, n - 1)
    t_pow = stats.t.ppf(POWER, n - 1)
    t_bonf = stats.t.ppf(1 - ALPHA / (2 * N_CELLS), n - 1)
    lo90, hi90 = mean - t90 * se, mean + t90 * se
    row = {"n_days": n, "diff": mean, "se": se, "ci95_lo": mean - t95 * se, "ci95_hi": mean + t95 * se,
           "mde": (t95 + t_pow) * se, "mde_bonf": (t_bonf + t_pow) * se,
           "equiv_bound": max(abs(lo90), abs(hi90)), "rv_spread": scale}
    for key in ["diff", "ci95_lo", "ci95_hi", "mde", "mde_bonf", "equiv_bound"]:
        row[f"{key}_pct"] = 100 * row[key] / scale
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="conservative", choices=["conservative", "optimistic"])
    ap.add_argument("--fee-bps", type=float, default=0.0)
    args = ap.parse_args()

    u = load(args.rule, args.fee_bps)
    rows = []
    for (g, regime), w in u.groupby(level=["gamma", "regime"]):
        w = w.droplevel(["gamma", "regime"])
        rv = w.xs("RV", level="estimator")
        oracle = w.xs("Oracle", level="estimator")["pnl"] - rv["pnl"] if "Oracle" in w.index.unique("estimator") else None
        for e in ESTIMATORS[1:]:
            d = (w.xs(e, level="estimator")["pnl"] - rv["pnl"]).dropna()
            row = {"gamma": g, "regime": regime, "estimator": e, "rv_pnl": rv["pnl"].mean(),
                   **cell_stats(d, rv["spread"].mean())}
            if oracle is not None:
                row["oracle_gain"] = oracle.mean()
            rows.append(row)
    res = pd.DataFrame(rows)
    res["regime"] = pd.Categorical(res["regime"], ["all", *REGIMES], ordered=True)
    res = res.sort_values(["gamma", "regime", "estimator"])
    out_csv = WF_DIR / f"power_{MAIN_VARIANT}_{args.rule}_fee{args.fee_bps:g}.csv"
    res.to_csv(out_csv, index=False, float_format="%.6g")

    pd.set_option("display.width", 220)
    print(f"{u.index.unique('date').size} days; {MAIN_VARIANT} sigma, {args.rule} fills, fee {args.fee_bps:g} bps. "
          "Estimator - RV, P&L in $ per day x regime unit; % = of RV's spread captured in that cell.")
    print(f"mde: 80% power, two-sided 5% / Bonferroni 5%/{N_CELLS}. "
          "equiv: smallest margin passing a 5% TOST (90% CI inside +-margin).\n")
    show = res.copy()
    show["gamma"] = show["gamma"].map(lambda g: f"{g:.0e}")
    show["ci95 $"] = [f"[{lo:+.1f}, {hi:+.1f}]" for lo, hi in zip(show["ci95_lo"], show["ci95_hi"])]
    show["ci95 %"] = [f"[{lo:+.1f}, {hi:+.1f}]" for lo, hi in zip(show["ci95_lo_pct"], show["ci95_hi_pct"])]
    for c in ["mde", "mde_bonf", "equiv_bound"]:
        show[f"{c} $ (%)"] = [f"{v:.1f} ({p:.1f}%)" for v, p in zip(show[c], show[f"{c}_pct"])]
    cols = ["gamma", "regime", "estimator", "n_days", "rv_pnl", "rv_spread", "diff", "ci95 $", "ci95 %",
            "mde $ (%)", "mde_bonf $ (%)", "equiv_bound $ (%)"]
    print(show[cols].round(1).to_string(index=False))

    cells = res[res["regime"] != "all"]
    whole = res[res["regime"] == "all"]
    print(f"\nSummary over the {len(cells)} regime x gamma cells (% of RV's spread captured):")
    for c, label in [("mde_pct", "mde (5%)"), ("mde_bonf_pct", "mde (Bonferroni)"),
                     ("equiv_bound_pct", "equivalence bound")]:
        q = cells[c].quantile([0.5, 0.9, 1.0])
        print(f"  {label:18s} median {q[0.5]:5.1f}%  90th pct {q[0.9]:5.1f}%  max {q[1.0]:5.1f}%")
    for m in [5, 10, 20]:
        print(f"  equivalent within +-{m:>2d}%: {(cells['equiv_bound_pct'] <= m).sum()}/{len(cells)} cells, "
              f"{(whole['equiv_bound_pct'] <= m).sum()}/{len(whole)} whole-day comparisons")

    print("\nWhole days vs what perfect foresight is worth (Oracle - RV, $/day):")
    for g, w in whole.groupby("gamma"):
        print(f"  gamma {g:.0e}: oracle gain {w['oracle_gain'].iloc[0]:+6.1f}; estimators' equivalence bounds "
              + ", ".join(f"{e} {b:.1f}" for e, b in zip(w["estimator"], w["equiv_bound"])))

    n0 = int(whole["n_days"].iloc[0])
    print(f"\nMore days: SE shrinks with sqrt(n). Median whole-day mde ({n0} days) "
          f"{whole['mde_pct'].median():.1f}% of RV's spread captured -> "
          + ", ".join(f"{whole['mde_pct'].median() * np.sqrt(n0 / n):.1f}% at {n} days" for n in [100, 160, 220]))
    print(f"[csv] {out_csv}")


if __name__ == "__main__":
    main()
