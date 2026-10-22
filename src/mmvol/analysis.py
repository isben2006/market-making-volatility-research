"""
Regime labels and the statistics used to compare estimators.

Regimes: each scored hour is labeled by its TRAILING 3-hour realized vol
(data before the hour only), split into terciles across every hour in the
study. The terciles use the whole study; that's fine because the label only
groups results afterwards and the market maker never sees it.

Unit of inference: day x regime. Hours within a day share one price path and
carry inventory from one to the next, so they aren't independent; days are
much closer to it.
"""
import math

import numpy as np
import pandas as pd
from arch.bootstrap import MCS

from mmvol.config import CLIP_SIZE, ESTIMATORS, ROOT

REGIMES = ["low", "mid", "high"]
STUDY_DAYS = ROOT / "data" / "study_days.csv"


def study_files(directory, prefix: str = "blocks") -> list:
    """Result files for the days in data/study_days.csv only, so days dropped
    from the study (e.g. by the data-quality check) never leak into results."""
    dates = pd.read_csv(STUDY_DAYS)["date"]
    files = [directory / f"{prefix}_{d}.csv" for d in dates]
    present = [f for f in files if f.exists()]
    if len(present) < len(files):
        print(f"[note] {directory.name}: {len(files) - len(present)} of {len(files)} study days have no "
              f"{prefix} file")
    if not present:
        raise SystemExit(f"no {prefix} files for the study days in {directory}")
    return present
MCS_SIZE = 0.10


def label_regimes(blocks: pd.DataFrame):
    hours = blocks.drop_duplicates(["date", "hour"])[["date", "hour", "trailing_vol"]]
    cuts = hours["trailing_vol"].quantile([1 / 3, 2 / 3]).values
    hours = hours.assign(regime=pd.cut(hours["trailing_vol"], [-np.inf, *cuts, np.inf], labels=REGIMES))
    return blocks.merge(hours[["date", "hour", "regime"]], on=["date", "hour"]), cuts


def with_fees(b: pd.DataFrame, fee_bps: float) -> pd.Series:
    return b["pnl"] - fee_bps * 1e-4 * b["notional"]


def units(b: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    """day x regime x estimator: summed P&L (net of fees), P&L per unit vol
    (P&L / (clip size x dollar vol of the mid over those hours)), mean QLIKE."""
    b = b.assign(pnl_net=with_fees(b, fee_bps), dv2=b["dollar_vol"] ** 2)
    u = (b.groupby(["date", "regime", "estimator"], observed=True)
          .agg(pnl=("pnl_net", "sum"), dv2=("dv2", "sum"), qlike=("qlike", "mean"), notional=("notional", "sum"),
               inv_abs=("inv_abs_mean", "mean"))
          .reset_index())
    u["pnl_per_vol"] = u["pnl"] / (CLIP_SIZE * np.sqrt(u["dv2"]))
    return u


def paired(u: pd.DataFrame, metric: str, regime: str, names: list = ESTIMATORS):
    """Paired difference of each of `names` vs RV across days, within one regime."""
    wide = u[u["regime"] == regime].pivot(index="date", columns="estimator", values=metric)
    res = {}
    for e in names:
        d = (wide[e] - wide["RV"]).dropna()
        se = d.std(ddof=1) / math.sqrt(len(d))
        better = (d < 0) if metric == "qlike" else (d > 0)
        res[e] = {"mean": wide[e].mean(), "median": wide[e].median(), "diff": d.mean(), "se": se,
                  "t": d.mean() / se if e != "RV" and se > 0 else np.nan, "better": int(better.sum()), "n": len(d)}
    return res, wide


def mcs_set(wide: pd.DataFrame, sign: float) -> set:
    """Estimators in the 10% Model Confidence Set (Hansen, Lunde & Nason 2011), loss = sign * metric."""
    losses = (sign * wide[ESTIMATORS]).dropna()
    m = MCS(losses, size=MCS_SIZE, reps=10000, block_size=1, method="R", seed=7)
    m.compute()
    return set(m.included)
