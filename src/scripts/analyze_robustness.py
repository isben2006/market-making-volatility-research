"""
Summarize the robustness configs (output/robustness_event/).

Per config:
  - AS economics: RV's P&L per day, fills, edge, days positive.
  - Estimator vs RV on P&L per unit vol, within each regime x gamma
    (36 comparisons): how many reach |t| >= 2, how many survive a
    Bonferroni correction across the 36 (two-sided p < 0.05/36, t-dist
    with n_days - 1 df), how many estimators fall out of the 10% Model
    Confidence Set, and the typical size of differences relative to RV's
    level.
  - Day-level risk (max drawdown, P&L to mean absolute inventory, Sharpe-
    style ratio of 5-minute P&L changes): paired differences vs RV across
    days for each gamma, with the same Bonferroni threshold.

Usage (from src/): python -m scripts.analyze_robustness
"""
import numpy as np
import pandas as pd
from scipy import stats

from mmvol.analysis import REGIMES, label_regimes, mcs_set, paired, study_files, units
from mmvol.config import ESTIMATORS, OUT

RB_DIR = OUT / "robustness_event"
CONFIG_ORDER = ["base", "k_fixed", "k_const", "d250", "d1000", "lat10", "lat200"]
ALPHA = 0.05


def load(prefix: str) -> pd.DataFrame:
    files = study_files(RB_DIR, prefix)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def p_two_sided(t: float, n: int) -> float:
    return 2 * stats.t.sf(abs(t), df=n - 1) if np.isfinite(t) else np.nan


def main():
    blocks, cuts = label_regimes(load("blocks"))
    days = load("days")
    n_days = blocks["date"].nunique()
    print(f"{n_days} days; regime cutoffs {cuts[0]:.0%}, {cuts[1]:.0%}; conservative fills, "
          f"{'/'.join(blocks['variant'].unique())} sigma, no fees")

    econ_rows, cmp_rows, sig_rows, risk_rows = [], [], [], []
    for cfg in CONFIG_ORDER:
        b = blocks[blocks["config"] == cfg]
        if b.empty:
            continue
        rv = b[b["estimator"] == "RV"]
        for g in sorted(b["gamma"].unique()):
            per_day = rv[rv["gamma"] == g].groupby("date").agg(pnl=("pnl", "sum"), fills=("fills", "sum"),
                                                              notional=("notional", "sum"))
            econ_rows.append({"config": cfg, "gamma": f"{g:.0e}", "RV pnl/day": round(per_day["pnl"].mean(), 1),
                              "fills/day": round(per_day["fills"].mean()),
                              "edge_bps": round(per_day["pnl"].sum() / per_day["notional"].sum() * 1e4, 3),
                              "days_positive": f"{(per_day['pnl'] > 0).sum()}/{len(per_day)}"})

        tests = []
        for g in sorted(b["gamma"].unique()):
            u = units(b[b["gamma"] == g], 0.0)
            for regime in REGIMES:
                res, wide = paired(u, "pnl_per_vol", regime)
                inc = mcs_set(wide, -1.0)
                for e in ESTIMATORS[1:]:
                    r = res[e]
                    p = p_two_sided(r["t"], r["n"])
                    tests.append({"config": cfg, "gamma": g, "regime": regime, "estimator": e, "diff": r["diff"],
                                  "rv_level": res["RV"]["mean"], "t": r["t"], "p": p, "in_mcs": e in inc})
        t = pd.DataFrame(tests)
        bonf = ALPHA / len(t)
        cmp_rows.append({"config": cfg, "comparisons": len(t), "|t|>=2": int((t["t"].abs() >= 2).sum()),
                         "Bonferroni-significant": int((t["p"] < bonf).sum()),
                         "out of MCS": int((~t["in_mcs"]).sum()),
                         "median |diff| / |RV level|": f"{(t['diff'].abs() / t['rv_level'].abs()).median():.1%}"})
        sig_rows += t[t["p"] < bonf].to_dict("records")

        d = days[days["config"] == cfg]
        for g in sorted(d["gamma"].unique()):
            dg = d[d["gamma"] == g]
            for metric in ["max_drawdown", "pnl_to_map", "sharpe_5m"]:
                wide = dg.pivot(index="date", columns="estimator", values=metric)
                for e in ESTIMATORS[1:]:
                    diff = (wide[e] - wide["RV"]).dropna()
                    tt = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
                    risk_rows.append({"config": cfg, "gamma": g, "metric": metric, "estimator": e,
                                      "rv_mean": wide["RV"].mean(), "diff": diff.mean(), "t": tt,
                                      "p": p_two_sided(tt, len(diff))})

    pd.set_option("display.width", 200)
    print("\n=== AS economics by config (RV, mean per day over the 21 scored hours) ===")
    econ = pd.DataFrame(econ_rows)
    print(econ.pivot(index="gamma", columns="config", values="RV pnl/day")[
        [c for c in CONFIG_ORDER if c in econ["config"].unique()]].to_string())
    print("\nfills/day:")
    print(econ.pivot(index="gamma", columns="config", values="fills/day")[
        [c for c in CONFIG_ORDER if c in econ["config"].unique()]].to_string())

    print("\n=== Estimator vs RV on P&L per unit vol (36 comparisons per config) ===")
    print(pd.DataFrame(cmp_rows).to_string(index=False))
    if sig_rows:
        print("\nBonferroni-significant comparisons:")
        for r in sig_rows:
            print(f"  {r['config']:8s} gamma {r['gamma']:.0e} {r['regime']:4s} {r['estimator']:5s} "
                  f"diff {r['diff']:+.2f} (RV level {r['rv_level']:+.2f}) t={r['t']:+.1f} p={r['p']:.2g}")

    risk = pd.DataFrame(risk_rows)
    n_risk = risk.groupby("config").size().iloc[0]
    print(f"\n=== Day-level risk metrics vs RV ({n_risk} comparisons per config: 3 metrics x 3 estimators x 4 gammas; "
          f"Bonferroni within config) ===")
    rsum = pd.DataFrame([{"config": cfg, "|t|>=2": int((x["t"].abs() >= 2).sum()),
                          "Bonferroni-significant": int((x["p"] < ALPHA / len(x)).sum())}
                         for cfg, x in risk.groupby("config")]).set_index("config").reindex(
        [c for c in CONFIG_ORDER if c in risk["config"].unique()])
    print(rsum.to_string())
    sig = risk[risk.apply(lambda r: r["p"] < ALPHA / n_risk, axis=1)]
    for _, r in sig.iterrows():
        print(f"  {r['config']:8s} gamma {r['gamma']:.0e} {r['metric']:12s} {r['estimator']:5s} "
              f"diff {r['diff']:+.2f} (RV {r['rv_mean']:+.2f}) t={r['t']:+.1f}")

    econ.to_csv(RB_DIR / "summary_economics.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(RB_DIR / "summary_comparisons.csv", index=False)
    risk.to_csv(RB_DIR / "summary_risk.csv", index=False, float_format="%.6g")
    print(f"\n[csv] {RB_DIR}/summary_*.csv")


if __name__ == "__main__":
    main()
