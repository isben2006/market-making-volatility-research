"""
Volatility estimators. Every function returns sigma_t in USD/sqrt(second),
computed CAUSALLY (only information up to and including t) so it can be
dropped straight into pipeline.run_as_sim's sigma= argument.

dS ~= S * d(log S), so per-second dollar-vol ~= mid_t * sqrt(per-second
log-return variance) -- every estimator below follows that same last step,
so they're comparable apples-to-apples; only how "per-second variance" gets
estimated differs.
"""
import numpy as np
import pandas as pd


def _log_returns(mid_1s: pd.Series) -> pd.Series:
    return np.log(mid_1s / mid_1s.shift(1)).fillna(0.0)  # first bar: no prior price, no info


# ---------------------------------------------------------------------------
# A. Realized vol: rolling sum of squared returns. The simplest possible
#    estimator -- and, as run 3 showed, the one most exposed to a single
#    outlier tick dominating the window.
# ---------------------------------------------------------------------------
def realized_vol(mid_1s: pd.Series, window_sec: int, fallback_sigma: float) -> np.ndarray:
    log_ret = _log_returns(mid_1s)
    sq = log_ret**2
    window_sum = sq.rolling(window=window_sec, min_periods=1).sum()
    window_n = sq.rolling(window=window_sec, min_periods=1).count().clip(lower=1)
    per_sec_var = window_sum / window_n
    sigma_t = mid_1s.values * np.sqrt(per_sec_var.values)
    sigma_t[0] = fallback_sigma  # zero return history at t=0
    return sigma_t


# ---------------------------------------------------------------------------
# B. Bipower variation: rolling sum of |r_t||r_{t-1}|, scaled by pi/2
#    (Barndorff-Nielsen & Shephard). Converges to the same integrated-variance
#    quantity as realized vol under a pure diffusion, but a single isolated
#    jump only enters as (big * normal) instead of (big * big) -- so one wild
#    tick contributes roughly linearly instead of quadratically. Directly
#    targets the outlier-sensitivity run 3 surfaced.
# ---------------------------------------------------------------------------
def bipower_vol(mid_1s: pd.Series, window_sec: int, fallback_sigma: float) -> np.ndarray:
    log_ret = _log_returns(mid_1s)
    abs_ret = log_ret.abs()
    prod = abs_ret * abs_ret.shift(1)
    prod.iloc[0] = 0.0  # no t-1 for the first bar
    window_sum = prod.rolling(window=window_sec, min_periods=1).sum()
    window_n = prod.rolling(window=window_sec, min_periods=1).count().clip(lower=1)
    per_sec_var = (np.pi / 2.0) * window_sum / window_n
    sigma_t = mid_1s.values * np.sqrt(per_sec_var.values)
    sigma_t[:2] = fallback_sigma  # first two bars have no |r_t||r_{t-1}| pair yet
    return sigma_t


# ---------------------------------------------------------------------------
# C. GARCH(1,1): captures vol clustering/persistence via an exponentially-
#    weighted recursion, sigma^2_t = omega + alpha*r_{t-1}^2 + beta*sigma^2_{t-1},
#    instead of a flat rolling window.
#
#    Parameters are fit ONCE by MLE on a burn-in window (first `burn_in_sec`
#    seconds only), then the recursion runs forward causally for the rest of
#    the day using only realized returns and the recursive state -- no
#    lookahead there. The fit step itself is a batch/offline step over the
#    burn-in sample (same as any real GARCH deployment: refit periodically,
#    run the recursion online in between) -- flagged here rather than glossed
#    over, since it's a real simplification vs. a proper walk-forward refit.
# ---------------------------------------------------------------------------
def fit_garch_params(mid_1s: pd.Series, burn_in_sec: int):
    from arch import arch_model

    raw_ret = _log_returns(mid_1s).values[:burn_in_sec]
    # 1-second returns are ~1e-4-1e-5 -- far below arch's recommended 1-1000 std
    # range for the optimizer. Scale to roughly unit variance instead of a fixed
    # x100 (which is enough for daily-bar returns but not tick-by-tick ones).
    scale = 1.0 / raw_ret.std()
    am = arch_model(raw_ret * scale, mean="Zero", vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")
    omega, alpha, beta = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    print(f"  [garch fit] burn-in={burn_in_sec}s  omega={omega:.6g} alpha={alpha:.4f} beta={beta:.4f} "
          f"(alpha+beta={alpha + beta:.4f}, {'stationary' if alpha + beta < 1 else 'NON-stationary!'})")
    return omega, alpha, beta, scale


def garch_vol(mid_1s: pd.Series, omega: float, alpha: float, beta: float, scale: float) -> np.ndarray:
    scaled_ret = _log_returns(mid_1s).values * scale
    n = len(scaled_ret)
    var_scaled = np.empty(n)  # conditional variance of the scaled return series
    # Seed with the empirical variance (==1.0 by construction: `scale` was chosen
    # to make the burn-in returns unit-variance), NOT the textbook long-run
    # formula omega/(1-alpha-beta) -- 1-second BTC returns fit close to a
    # unit-root process (alpha+beta ~ 1), so that formula divides by ~0 and
    # explodes the very first second. This IS realistic GARCH-on-HFT-data
    # behavior, not a data problem; the empirical seed just avoids the
    # division-by-epsilon rather than pretending stationarity that isn't there.
    var_scaled[0] = 1.0
    for t in range(1, n):
        var_scaled[t] = omega + alpha * scaled_ret[t - 1] ** 2 + beta * var_scaled[t - 1]
    per_sec_var = np.clip(var_scaled, 0, None) / scale**2  # back to raw log-return variance
    return mid_1s.values * np.sqrt(per_sec_var)
