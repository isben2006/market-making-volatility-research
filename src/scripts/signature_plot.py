"""
Volatility signature plot: daily realized variance as a function of the
sampling interval. If microstructure noise matters, RV computed from very
short returns is biased upward and the curve rises as the interval shrinks;
where it flattens is a reasonable sampling interval for the RV-based
estimators (Bandi & Russell 2008; Hansen & Lunde 2006).

Each interval uses the subsampled average over all grid offsets, so coarse
intervals don't throw away data.

Usage (from src/): python -m scripts.signature_plot --dates 2024-01-10 2023-07-19
"""
import argparse
import math

import numpy as np

from mmvol.config import OUT
from mmvol.data import load_book_1s
from mmvol.plotting import PALETTE, TEXT_WIDTH_IN, plt, save

INTERVALS_SEC = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300]
DAYS_PER_YEAR = 365


def subsampled_rv(log_mid: np.ndarray, step: int) -> float:
    rvs = []
    for offset in range(step):
        grid = log_mid[offset::step]
        rvs.append(np.sum(np.diff(grid) ** 2))
    return float(np.mean(rvs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["2024-01-10", "2023-07-19"])
    dates = ap.parse_args().dates

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.0))
    colors = [PALETTE["mid"], PALETTE["ask"], PALETTE["bid"], PALETTE["ink2"]]
    markers = ["o", "s", "^", "D"]

    for date, color, marker in zip(dates, colors, markers):
        mid_1s, _, _ = load_book_1s(date)
        log_mid = np.log(mid_1s.values)
        ann_vol = [math.sqrt(subsampled_rv(log_mid, s) * DAYS_PER_YEAR) for s in INTERVALS_SEC]

        print(f"\n[{date}] annualized vol by sampling interval (subsampled RV):")
        ref = ann_vol[INTERVALS_SEC.index(300)]
        for s, v in zip(INTERVALS_SEC, ann_vol):
            print(f"  {s:>4}s  {v:6.1%}   ({v / ref:5.2f}x the 300s value)")

        # Normalized so days with very different vol levels share one axis; the shape is what matters.
        ax.plot(INTERVALS_SEC, [v / ref for v in ann_vol], marker=marker, ms=4, lw=1.2, color=color,
                label=f"{date} ({ref:.0%} annualized at 300 s)")

    ax.axhline(1.0, color=PALETTE["ink2"], lw=0.8, zorder=1.5)
    ax.set_xscale("log")
    ax.set_xticks(INTERVALS_SEC)
    ax.set_xticklabels([str(s) for s in INTERVALS_SEC])
    ax.minorticks_off()
    ax.set_xlabel("Sampling interval (seconds, log scale)")
    ax.set_ylabel("Realized vol / realized vol at 300 s")
    ax.legend()
    fig.tight_layout()
    save(fig, OUT / "signature_plot.png")


if __name__ == "__main__":
    main()
