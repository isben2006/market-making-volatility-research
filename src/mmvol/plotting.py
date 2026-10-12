"""Shared chart style. Importing this module applies it."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = dict(mid="#2a78d6", bid="#1baf7a", ask="#eb6834", series="#2a78d6",
               grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e")
EST_COLORS = {"RV": PALETTE["mid"], "BV": PALETTE["ask"], "EWMA": "#7b4fd6",
              "GARCH": PALETTE["bid"], "HAR": "#d6336c", "Flat": "#8a8984", "Oracle": "#0b0b0b"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": PALETTE["axis"], "axes.grid": True,
    "grid.color": PALETTE["grid"], "grid.linewidth": 0.7,
    "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink2"],
    "xtick.color": PALETTE["ink2"], "ytick.color": PALETTE["ink2"],
    "axes.titlecolor": PALETTE["ink"], "font.size": 10,
})
