"""
Analyze the walk-forward results (output/walkforward_event/blocks_*.csv).

Regimes and the day x regime inference unit are described in mmvol.analysis.
Fees are applied here from the recorded notional (fee_bps per side traded).

Prints:
  1. AS economics: what an AS market maker earns at each gamma, gross and
     net of fees, under both fill bounds (RV, chosen variant).
  2. Forecast accuracy (QLIKE) by regime.
  3. The estimator comparison: P&L per unit vol, paired differences vs RV
     across days within each regime, Model Confidence Set, for each gamma
     under the chosen fill rule and fee.
  4. Benchmarks: the estimators between Flat (no intra-hour dynamics) and
     Oracle (perfect foresight of the next 5 minutes' realized vol), with
     OracleHour (the hour's realized vol, held flat) splitting the gap into
     knowing the level vs knowing the timing. Paired differences vs RV in USD
     per day. Oracle - RV bounds what any better sigma forecast could add.
  5. P&L decomposition (mmvol.simulator.run_event_sim): spread captured,
     adverse selection (the mid's move against each fill over --markout
     seconds), inventory P&L and fees, for RV, then each estimator and
     benchmark vs RV component by component.

Usage (from src/): python -m scripts.analyze_walkforward [--variant raw] [--rule conservative] [--fee-bps 0] [--markout 5]
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from mmvol.analysis import REGIMES, label_regimes, mcs_set, paired, study_files, units
from mmvol.config import CLIP_SIZE, ESTIMATORS, MAIN_VARIANT, OUT, SYMBOL
from mmvol.plotting import EST_COLORS, PALETTE, plt

WF_DIR = OUT / "walkforward_event"
FEE_SCENARIOS_BPS = [0.0, 2.0]   # gross, and ~Binance's standard maker fee (check the 2023-24 schedule)


def load_blocks() -> pd.DataFrame:
    files = study_files(WF_DIR)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def economics_table(blocks: pd.DataFrame, variant: str):
    rv = blocks[(blocks["variant"] == variant) & (blocks["estimator"] == "RV")]
    n_days = rv["date"].nunique()
    print(f"\n=== 1. AS economics (RV, {variant} sigma; per day = sum over the 21 scored hours, mean of {n_days} days) ===")
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
    ap.add_argument("--variant", default=MAIN_VARIANT, choices=["raw", "matched"])
    ap.add_argument("--rule", default="conservative", choices=["conservative", "optimistic"])
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--markout", type=int, default=5, choices=[5, 60], help="adverse-selection horizon, seconds")
    args = ap.parse_args()

    blocks, cuts = label_regimes(load_blocks())
    print(f"{blocks['date'].nunique()} days, {blocks.drop_duplicates(['date', 'hour']).shape[0]} scored hours; "
          f"regime cutoffs (trailing 3h vol): {cuts[0]:.0%}, {cuts[1]:.0%}")
    economics_table(blocks, args.variant)

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

    if (blocks["variant"] == "benchmark").any():
        benchmark_table(blocks, args.variant, args.rule, args.fee_bps)
    if "pnl_spread" in blocks.columns:
        decomposition_table(blocks, args.variant, args.rule, args.fee_bps, args.markout)


def benchmark_table(blocks: pd.DataFrame, variant: str, rule: str, fee_bps: float):
    names = ["Flat", *ESTIMATORS, "OracleHour", "Oracle"]
    b = blocks[blocks["variant"].isin([variant, "benchmark"]) & (blocks["rule"] == rule)
               & blocks["estimator"].isin(names)]
    full = b.groupby("date")["estimator"].nunique() == len(names)
    b = b[b["date"].isin(full.index[full])]  # same days for every column
    b = pd.concat([b, b.assign(regime="all")], ignore_index=True)  # "all": every scored hour of the day
    print(f"\n=== 4. Benchmarks: P&L in USD per day vs RV ({variant} sigma, {rule} fills, fee {fee_bps:g} bps) ===")
    print("Flat: trailing 3h realized vol, updated hourly. OracleHour: this hour's realized vol, held flat. "
          "Oracle: realized vol of the next 5 minutes. Oracles aren't tradable.")
    print(f"{b['date'].nunique()} days. Cells: mean paired difference vs RV, t-stat, days better. "
          "Headroom (whole days only) = (Oracle - RV) / |RV|.")
    rows, plot_rows = [], []
    for g in sorted(b["gamma"].unique()):
        u = units(b[b["gamma"] == g], fee_bps)
        for regime in ["all", *REGIMES]:
            res, _ = paired(u, "pnl", regime, names)
            rv = res["RV"]["mean"]
            for e in names:
                r = res[e]
                rows.append({"gamma": g, "regime": regime, "estimator": e, "days": r["n"], "pnl_mean": r["mean"],
                             "diff_vs_rv": r["diff"], "se": r["se"], "t": r["t"], "days_better": r["better"]})
            print(f"gamma {g:.0e}, {regime:>4}: RV {rv:+8.1f}/day | "
                  + " | ".join(f"{e} {res[e]['diff']:+6.1f} (t={res[e]['t']:+.1f}, {res[e]['better']}/{res[e]['n']})"
                               for e in names if e != "RV")
                  + (f" | headroom {res['Oracle']['diff'] / abs(rv):+.0%}" if regime == "all" else ""))
    out = pd.DataFrame(rows)
    out_csv = WF_DIR / f"benchmarks_{variant}_{rule}_fee{fee_bps:g}.csv"
    out.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"[csv] {out_csv}")

    gammas = sorted(out["gamma"].unique())
    fig, axes = plt.subplots(1, len(gammas), figsize=(3.4 * len(gammas), 4.2))
    others = [e for e in names if e != "RV"]
    for ax, g in zip(np.atleast_1d(axes), gammas):
        s = out[(out["gamma"] == g) & (out["regime"] == "all")].set_index("estimator").loc[others]
        x = np.arange(len(others))
        for i, e in enumerate(others):
            ax.errorbar(x[i], s.loc[e, "diff_vs_rv"], yerr=1.96 * s.loc[e, "se"], fmt="o", ms=5, capsize=3,
                        color=EST_COLORS[e], lw=1.3)
        ax.axhline(0, color=PALETTE["ink2"], lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(others, rotation=45, ha="right")
        ax.set_title(f"gamma = {g:.0e}", fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("P&L per day minus RV's, USD\n(mean across days, 95% CI)")
    fig.suptitle(f"Estimators between the benchmarks: Flat (no dynamics) and Oracle (perfect foresight) "
                 f"({SYMBOL}, {rule} fills, fee {fee_bps:g} bps; above 0 = loses less than RV)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = WF_DIR / f"benchmarks_{variant}_{rule}_fee{fee_bps:g}.png"
    fig.savefig(out_png, dpi=130)
    print(f"[plot] {out_png}")


def _paired_t(d: pd.Series):
    d = d.dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() / se if se > 0 else np.nan


def decomposition_table(blocks: pd.DataFrame, variant: str, rule: str, fee_bps: float, h: int):
    names = ["Flat", *ESTIMATORS, "OracleHour", "Oracle"]
    b = blocks[blocks["variant"].isin([variant, "benchmark"]) & (blocks["rule"] == rule)
               & blocks["estimator"].isin(names) & blocks["pnl_spread"].notna()]
    b = b.assign(fees=-fee_bps * 1e-4 * b["notional"])
    comps = {"spread": "pnl_spread", "adverse_sel": f"pnl_as_{h}", "inventory": f"pnl_inv_{h}", "fees": "fees"}
    day = (b.groupby(["gamma", "estimator", "date"])[[*comps.values(), "pnl", "fills"]].sum()
            .rename(columns={v: k for k, v in comps.items()}))
    day["total"] = day["pnl"] + day["fees"]
    full = day.groupby(["gamma", "date"]).size() == len(names)  # same days for every row
    day = day[day.index.droplevel("estimator").isin(full.index[full])]
    per_fill = 1.0 / (day["fills"] * CLIP_SIZE)  # $ per BTC traded: comparable to a price distance
    day["spread_pf"], day["as_pf"] = day["spread"] * per_fill, day["adverse_sel"] * per_fill
    n_days = day.index.get_level_values("date").nunique()

    print(f"\n=== 5. P&L decomposition ({variant} sigma, {rule} fills, fee {fee_bps:g} bps, "
          f"adverse selection over {h} s; {n_days} days) ===")
    print("Per day: spread captured + adverse selection + inventory + fees = total. "
          "Per fill: $ per BTC traded (spread = distance from the decision mid; adverse = mid move against the fill).")
    rv = day.xs("RV", level="estimator").groupby("gamma").mean()
    tab = rv[[*comps, "total", "fills", "spread_pf", "as_pf"]].copy()
    tab.index = [f"{g:.0e}" for g in tab.index]
    print("RV:")
    print(tab.round(2).to_string())

    rows = []
    print("\nVs RV, mean paired difference per day (t-stat); per-fill adverse selection difference in $ per BTC:")
    for g in sorted(day.index.get_level_values("gamma").unique()):
        w = day.xs(g, level="gamma")
        base = w.xs("RV", level="estimator")
        line = []
        for e in [e for e in names if e != "RV"]:
            x = w.xs(e, level="estimator")
            row = {"gamma": g, "estimator": e}
            for c in [*comps, "total", "fills", "as_pf"]:
                row[f"{c}_diff"], row[f"{c}_t"] = _paired_t(x[c] - base[c])
            rows.append(row)
            line.append(f"{e:>10s}: spread {row['spread_diff']:+6.1f} ({row['spread_t']:+.1f})  "
                        f"adverse {row['adverse_sel_diff']:+6.1f} ({row['adverse_sel_t']:+.1f})  "
                        f"inventory {row['inventory_diff']:+6.1f} ({row['inventory_t']:+.1f})  "
                        f"fees {row['fees_diff']:+5.1f}  total {row['total_diff']:+6.1f}  "
                        f"| adverse/fill {row['as_pf_diff']:+.3f} ({row['as_pf_t']:+.1f})")
        print(f"gamma {g:.0e}:\n  " + "\n  ".join(line))
    out_csv = WF_DIR / f"decomposition_{variant}_{rule}_fee{fee_bps:g}_h{h}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6g")
    print(f"[csv] {out_csv}")

    # by regime: estimators vs RV per component, day x regime units, Bonferroni over the 36 cells
    reg = (b[b["estimator"].isin(ESTIMATORS)].groupby(["gamma", "regime", "estimator", "date"], observed=True)
            [[*comps.values(), "pnl", "fills"]].sum().rename(columns={v: k for k, v in comps.items()}))
    reg["total"] = reg["pnl"] + reg["fees"]
    reg["as_pf"] = reg["adverse_sel"] / (reg["fills"] * CLIP_SIZE)
    cells = []
    for (g, regime), w in reg.groupby(level=["gamma", "regime"], observed=True):
        w = w.droplevel(["gamma", "regime"])
        base = w.xs("RV", level="estimator")
        for e in ESTIMATORS[1:]:
            x = w.xs(e, level="estimator")
            for c in ["total", "spread", "adverse_sel", "inventory", "fills", "as_pf"]:
                d = (x[c] - base[c]).dropna()
                diff, t = _paired_t(d)
                cells.append({"gamma": g, "regime": regime, "estimator": e, "component": c, "diff": diff, "t": t,
                              "bonferroni": 2 * stats.t.sf(abs(t), len(d) - 1) < 0.05 / 36})
    cells = pd.DataFrame(cells)
    n_cells = cells.groupby("component").size().iloc[0]
    print(f"\nBy regime: estimators vs RV, {n_cells} cells per component (3 estimators x 3 regimes x 4 gammas):")
    print(cells.groupby("component", sort=False).agg(bonferroni=("bonferroni", "sum"),
                                                    abs_t_ge_2=("t", lambda t: int((t.abs() >= 2).sum())))
          .to_string())
    sig = cells[(cells["component"] == "as_pf") & cells["bonferroni"]]
    if len(sig):
        print("Bonferroni-significant per-fill adverse selection ($ per BTC, + = less damage than RV):")
        print(sig[["gamma", "regime", "estimator", "diff", "t"]].round(3).to_string(index=False))
    out_csv = WF_DIR / f"decomposition_regimes_{variant}_{rule}_fee{fee_bps:g}_h{h}.csv"
    cells.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"[csv] {out_csv}")


if __name__ == "__main__":
    main()
