"""Paths, instrument and settings shared across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/mmvol/config.py -> repo root
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

SYMBOL = "BTCUSDT"
DEFAULT_DATE = "2024-01-10"  # first development day
DAY_SECONDS = 86_400
SECONDS_PER_YEAR = 365 * 24 * 3600

# ---- market maker -----------------------------------------------------------
CLIP_SIZE = 0.005            # BTC per order
ROLLING_HORIZON_SEC = 300    # AS horizon resets every 5 minutes (a literal full-day horizon is unusable)
SIGMA_ANNUAL_GUESS = 0.55    # annualized vol used only until an estimator has data (and by the legacy runs)

# ---- walk-forward -----------------------------------------------------------
FIRST_FIT_SEC = 3 * 3600     # burn-in: first fits use 00:00-03:00; scoring starts at 03:00 UTC
REFIT_SEC = 3600             # GARCH, HAR, level-matching and k are refit every hour
TRAIN_SEC = 3 * 3600         # trailing training window (GARCH, level-matching, k)
BLOCK_SEC = 3600             # results are scored per hour
REGIME_LOOKBACK_SEC = 3 * 3600   # regime signal: realized vol over the trailing 3 hours
FEATURE_SAMPLING_SEC = 60        # ...from 60s returns (1s RV's bias varies by regime; see the signature plot)
ESTIMATORS = ["RV", "EWMA", "GARCH", "HAR"]
VARIANTS = ["raw", "matched"]
KEEP_RAW = {"2024-01-10", "2023-07-19"}  # development days: raw data kept on disk for the single-day scripts
