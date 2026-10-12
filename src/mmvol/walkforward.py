"""
Building blocks for the walk-forward study (used by scripts/run_walkforward.py
and scripts/run_robustness.py): load a day, build every estimator, turn
per-second sigmas into per-decision sigmas, and score each hour.
"""
import math

import numpy as np
import pandas as pd

from mmvol.config import (
    BLOCK_SEC, DAY_SECONDS, ESTIMATORS, FEATURE_SAMPLING_SEC, FIRST_FIT_SEC, RAW, REFIT_SEC, REGIME_LOOKBACK_SEC,
    ROLLING_HORIZON_SEC, SYMBOL, TRAIN_SEC,
)
from mmvol.data import download_and_extract, load_events, mid_1s_from_book
from mmvol.estimators import (
    ewma_lambda_matching_window, ewma_vol, garch_vol_walkforward, har_vol_walkforward, level_match_walkforward,
    realized_vol, sigma_guess,
)
from mmvol.simulator import decision_sigma

DECISION_MS = 100
LATENCY_MS = 50
GAMMAS = [1e-5, 1e-4, 1e-3, 1e-2]
RULES = ["conservative", "optimistic"]
HOURS_PER_YEAR = 365 * 24


def ann_vol_from_r60(r60: np.ndarray, seconds: int) -> float:
    """Annualized vol from 60s log returns spanning `seconds`."""
    return math.sqrt(np.sum(r60 ** 2) * HOURS_PER_YEAR * 3600 / seconds)


def load_day(date: str):
    """Download if needed; return raw book events, trades and the per-second mid."""
    for dt in ("bookTicker", "aggTrades"):
        download_and_extract(SYMBOL, date, dt, RAW)
    book, trades = load_events(date)
    index = pd.date_range(f"{date} 00:00:00+00:00", periods=DAY_SECONDS, freq="1s")
    mid = mid_1s_from_book(book, index)
    return book, trades, mid


def build_sigmas(mid: pd.Series):
    """Per-second sigma for every (variant, estimator), plus fit diagnostics."""
    fallback = sigma_guess(mid)
    raw = {
        "RV": realized_vol(mid, ROLLING_HORIZON_SEC, fallback),
        "EWMA": ewma_vol(mid, ewma_lambda_matching_window(ROLLING_HORIZON_SEC), fallback),
    }
    raw["GARCH"], g_diag = garch_vol_walkforward(mid, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
    raw["HAR"], h_diag = har_vol_walkforward(mid, ROLLING_HORIZON_SEC, FIRST_FIT_SEC, REFIT_SEC)
    sigmas = {("raw", e): raw[e] for e in ESTIMATORS}
    for e in ESTIMATORS:
        sigmas[("matched", e)] = level_match_walkforward(raw[e], mid, FIRST_FIT_SEC, REFIT_SEC, TRAIN_SEC)
    return sigmas, {**g_diag, **h_diag}


def decision_sigmas(sigmas: dict, dec_t: np.ndarray) -> dict:
    # GARCH's value at second s already uses only returns up to s-1; the others include second s
    return {key: decision_sigma(sigmas[key], dec_t, lagged=(key[1] != "GARCH")) for key in sigmas}


def score_blocks(date: str, mid: pd.Series, sigmas: dict, sim: dict, labels: list, k_sec: np.ndarray) -> list:
    """Per scored hour and run: P&L, fills, notional, spread, inventory, plus
    the run's QLIKE and the hour's market features. labels[j] is a dict of
    fields for run j and must include 'variant' and 'estimator'."""
    pnl, per_sec = sim["pnl"], sim["per_sec"]
    mid_v = mid.values
    r2 = pd.Series(np.r_[0.0, np.diff(np.log(mid_v)) ** 2])
    y = r2[::-1].rolling(ROLLING_HORIZON_SEC, min_periods=ROLLING_HORIZON_SEC).mean()[::-1].shift(-1).values
    keys = sorted({(lab["variant"], lab["estimator"]) for lab in labels})
    h = {key: (sigmas[key] / mid_v) ** 2 for key in keys}
    r60 = np.diff(np.log(mid_v[::FEATURE_SAMPLING_SEC]))

    rows = []
    for start in range(FIRST_FIT_SEC, DAY_SECONDS, BLOCK_SEC):
        end = start + BLOCK_SEC
        i0, i1 = start // FEATURE_SAMPLING_SEC, end // FEATURE_SAMPLING_SEC
        market = {
            "date": date, "hour": start // 3600,
            "trailing_vol": ann_vol_from_r60(r60[(start - REGIME_LOOKBACK_SEC) // FEATURE_SAMPLING_SEC:i0],
                                             REGIME_LOOKBACK_SEC),
            "block_vol": ann_vol_from_r60(r60[i0:i1], BLOCK_SEC),
            "dollar_vol": math.sqrt(np.sum(np.diff(mid_v[start - 1:end]) ** 2)),
            "mid_return": math.log(mid_v[end - 1] / mid_v[start - 1]),
            "k": float(k_sec[start:end].mean()),
        }
        t_eval = np.arange(start, min(end, DAY_SECONDS - ROLLING_HORIZON_SEC))
        fcst = {}
        for key in keys:
            hh, yy = h[key][t_eval], y[t_eval]
            ok = (hh > 0) & (yy > 0)
            ratio = yy[ok] / hh[ok]
            fcst[key] = {
                "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)) if ok.any() else np.nan,
                "fcst_bias": float(hh[ok].mean() / yy[ok].mean()) if ok.any() else np.nan,
            }
        blk_pnl = pnl[end - 1] - pnl[start - 1]
        blk = {name: sim[name][start:end].sum(axis=0) for name in ["fills", "notional", "spread_sum", "inv_abs_sum"]}
        n_dec_blk = BLOCK_SEC * per_sec
        for j, lab in enumerate(labels):
            rows.append({**market, **lab,
                         "pnl": blk_pnl[j], "fills": int(blk["fills"][j]), "notional": blk["notional"][j],
                         "spread_mean": blk["spread_sum"][j] / n_dec_blk,
                         "inv_abs_mean": blk["inv_abs_sum"][j] / n_dec_blk,
                         **fcst[(lab["variant"], lab["estimator"])]})
    return rows
