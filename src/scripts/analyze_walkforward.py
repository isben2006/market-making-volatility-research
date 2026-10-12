"""
Analyze the walk-forward results (output/walkforward_event/blocks_*.csv).

Regimes and the day x regime inference unit are described in mmvol.analysis.
Fees are applied here from the recorded notional (fee_bps per side traded).

Prints:
  1. AS economics: what an AS market maker earns at each gamma, gross and
     net of fees, under both fill bounds (RV, level-matched).
  2. Forecast accuracy (QLIKE) by regime.
  3. The estimator comparison: P&L per unit vol, paired differences vs RV
     across days within each regime, Model Confidence Set, for each gamma
     under the chosen fill rule and fee.

Usage (from src/): python -m scripts.analyze_walkforward [--variant matched] [--rule conservative] [--fee-bps 0]
"""
import argparse

import numpy as np
import pandas as pd

from mmvol.analysis import REGIMES, label_regimes, mcs_set, paired, units
from mmvol.config import ESTIMATORS, OUT, SYMBOL
from mmvol.plotting import EST_COLORS, PALETTE, plt

WF_DIR = OUT / "walkforward_event"
FEE_SCENARIOS_BPS = [0.0, 2.0]   # gross, and ~Binance's standard maker fee (check the 2023-24 schedule)


def load_blocks() -> pd.DataFrame:
    files = sorted(WF_DIR.glob("blocks_*.csv"))
    if not files:
        raise SystemExit(f"no results in {WF_DIR}; run scripts.run_walkforward first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def economics_table(blocks: pd.DataFrame):
    rv = blocks[(blocks["variant"] == "matched") & (blocks["estimator"] == "RV")]
    n_days = rv["date"].nunique()
    print(f"\n=== 1. AS economics (RV, level-matched; per day = sum over the 21 scored hours, mean of {n_days} days) ===")
    rows = []
    for (g, rule), grp in rv.groupby(["gamma", "rule"]):
        per_day = grp.groupby("date").agg(pnl=("pnl", "sum"), notional=("notional", "sum"), fills=("fills", "sum"),
                                         spread=("spread_mean", "mean"))
        row = {"gamma": f"{g:.0e}", "rule": rule, "fills/day": f"{per_day['fills'].mean():.0f}",
               "spread_$": f"{per_day['spread'].mean():.2f}",
               "edge_bps": f"{per_day['pnl'].sum() / per_day['notional'].sum() * 1e4:+.3f}",
               "days_positive": f"{(per_day['pnl'] > 0).sum()}/{len(per_day)}"}
        for fee in FEE_SCENARIOS_BPS:
            net = per_day["pnl"] - fee * 1e-4 * per_day["notional"]
            row[f"pnl/day fee {fee:g}bps"] = f"{net.mean():+.1f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="matched", choices=["matched", "raw"])
    ap.add_argument("--rule", default="conservative", choices=["conservative", "optimistic"])
    ap.add_argument("--fee-bps", type=float, default=0.0)
    args = ap.parse_args()

    blocks, cuts = label_regimes(load_blocks())
    print(f"{blocks['date'].nunique()} days, {blocks.drop_duplicates(['date', 'hour']).shape[0]} scored hours; "
          f"regime cutoffs (trailing 3h vol): {cuts[0]:.0%}, {cuts[1]:.0%}")
    economics_table(blocks)

    sel = blocks[(blocks["variant"] == args.variant) & (blocks["rule"] == args.rule)]

    print(f"\n=== 2. Forecast accuracy (QLIKE, lower is better; {args.variant} sigma) ===")
    u0 = units(sel[sel["gamma"] == sel["gamma"].iloc[0]], 0.0)
    rows = []
    for regime in REGIMES:
        res, wide = paired(u0, "qlike", regime)
        inc = mcs_set(wide, +1.0)
        for e in ESTIMATORS:
            r = res[e]
            rows.append({"regime": regime, "estimator": e, "median": f"{r['median']:.3f}",
                         "beats_RV": "--" if e == "RV" else f"{r['better']}/{r['n']}", "in_MCS": e in inc})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n=== 3. Estimator comparison: P&L per unit vol vs RV ({args.variant} sigma, {args.rule} fills, "
          f"fee {args.fee_bps:g} bps) ===")
    all_rows, plot_rows = [], []
    for g in sorted(sel["gamma"].unique()):
        u = units(sel[sel["gamma"] == g], args.fee_bps)
        for regime in REGIMES:
            res, wide = paired(u, "pnl_per_vol", regime)
            inc = mcs_set(wide, -1.0)
            for e in ESTIMATORS:
                r = res[e]
                all_rows.append({"gamma": g, "regime": regime, "estimator": e, "days": r["n"],
                                 "ppv_mean": r["mean"], "diff_vs_rv": r["diff"], "se": r["se"], "t": r["t"],
                                 "days_better": r["better"], "in_mcs": e in inc})
    summary = pd.DataFrame(all_rows)
    out_csv = WF_DIR / f"summary_{args.variant}_{args.rule}_fee{args.fee_bps:g}.csv"
    summary.to_csv(out_csv, index=False, float_format="%.6g")  # %.6g keeps gamma=1e-05 intact

    for g in sorted(summary["gamma"].unique()):
        s = summary[summary["gamma"] == g]
        print(f"\ngamma = {g:.0e}")
        tab = {}
        for regime in REGIMES:
            col = []
            for e in ESTIMATORS:
                r = s[(s["regime"] == regime) & (s["estimator"] == e)].iloc[0]
                mark = "" if r["in_mcs"] else " (out of MCS)"
                col.append(f"{r['ppv_mean']:+.2f}" if e == "RV" else
                           f"{r['diff_vs_rv']:+.2f} t={r['t']:+.1f} {int(r['days_better'])}/{int(r['days'])}{mark}")
            tab[f"{regime} vol"] = col
        print(pd.DataFrame(tab, index=["RV (level)"] + [f"{e} vs RV" for e in ESTIMATORS[1:]]).to_string())
    print(f"[csv] {out_csv}")

    gammas = sorted(summary["gamma"].unique())
    fig, axes = plt.subplots(1, len(gammas), figsize=(3.4 * len(gammas), 4.2), sharey=False)
    for ax, g in zip(np.atleast_1d(axes), gammas):
        s = summary[summary["gamma"] == g]
        for i, e in enumerate([e for e in ESTIMATORS if e != "RV"]):
            sub = s[s["estimator"] == e].set_index("regime").loc[REGIMES]
            x = np.arange(len(REGIMES)) + (i - 1) * 0.22
            ax.errorbar(x, sub["diff_vs_rv"], yerr=1.96 * sub["se"], fmt="o", ms=5, capsize=3,
                        color=EST_COLORS[e], label=e, lw=1.3)
        ax.axhline(0, color=PALETTE["ink2"], lw=0.9)
        ax.set_xticks(range(len(REGIMES)))
        ax.set_xticklabels([f"{r}\nvol" for r in REGIMES])
        ax.set_title(f"gamma = {g:.0e}", fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("P&L per unit vol minus RV's\n(mean across days, 95% CI)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Event-driven walk-forward: estimator vs RV by regime ({SYMBOL}, {args.variant} sigma, "
                 f"{args.rule} fills, fee {args.fee_bps:g} bps; above 0 = earns more / loses less)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = WF_DIR / f"regime_comparison_{args.variant}_{args.rule}_fee{args.fee_bps:g}.png"
    fig.savefig(out_png, dpi=130)
    print(f"[plot] {out_png}")


if __name__ == "__main__":
    main()
