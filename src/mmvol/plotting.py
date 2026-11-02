"""Shared chart style, sized for a paper page (6.5 in text width). Importing this module applies it."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT_WIDTH_IN = 6.5

PALETTE = dict(mid="#2a78d6", bid="#1baf7a", ask="#eb6834", series="#2a78d6",
               grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e", muted="#8a8984")
# The three compared estimators take the first three categorical slots (validated colorblind-safe
# as a set); benchmarks are neutral, since their x-axis labels already name them.
EST_COLORS = {"RV": PALETTE["ink2"], "BV": PALETTE["muted"], "EWMA": "#2a78d6", "GARCH": "#eb6834",
              "HAR": "#1baf7a", "Flat": PALETTE["muted"], "OracleHour": PALETTE["ink2"], "Oracle": PALETTE["ink"]}
# Shapes carry identity too, so the charts survive grayscale print.
EST_MARKERS = {"EWMA": "o", "GARCH": "s", "HAR": "^", "Flat": "D", "OracleHour": "D", "Oracle": "D"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": PALETTE["axis"], "axes.linewidth": 0.8,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": PALETTE["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink2"],
    "xtick.color": PALETTE["ink2"], "ytick.color": PALETTE["ink2"],
    "axes.titlecolor": PALETTE["ink"], "axes.titlesize": 9, "font.size": 8,
    "legend.frameon": False, "mathtext.default": "regular",
    "pdf.fonttype": 42,  # embed TrueType so the PDF text stays selectable
})


def gamma_label(g: float) -> str:
    """1e-05 -> 'γ = 10⁻⁵' in mathtext."""
    return rf"$\gamma = 10^{{{round(math.log10(g))}}}$"


def save(fig, path_png) -> None:
    """Write the PNG (README) and a vector PDF next to it (paper)."""
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] {path_png} (+ .pdf)")
