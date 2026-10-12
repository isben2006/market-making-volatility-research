"""
LEGACY: the superseded 1-second-bar simulator. Its quotes use each second's
last mid and are filled by trades from earlier in the same second (a
lookahead that hides adverse selection), so its P&L results are invalid.
Kept so the project history can be reproduced; the current simulator is
mmvol/simulator.py.

Contains the 1-second AS quote loop and fill rules, the per-second trade
loader, the guessed calibration constants, and per-run plotting.
"""
import math

import numpy as np
import pandas as pd

from mmvol.config import (  # noqa: F401  (re-exported for the other legacy scripts)
    CLIP_SIZE, DAY_SECONDS, DEFAULT_DATE, OUT, RAW, ROLLING_HORIZON_SEC, ROOT, SECONDS_PER_YEAR,
    SIGMA_ANNUAL_GUESS, SYMBOL,
)
from mmvol.data import load_book_1s  # noqa: F401
from mmvol.plotting import PALETTE, plt

# ---- guessed AS parameters, before k was calibrated from data ----------------
GAMMA_NAIVE = 0.1                  # risk aversion, straight from the AS(2008) worked example
K = 1.5                            # order-flow decay param, also straight from the AS(2008) example
TARGET_MAX_SPREAD_USD = 8.0        # gamma is back-solved to hit roughly this $ spread at max time-remaining

# ---- fill-intensity model (replaces deterministic trade-through) -----------
# AS assumes fills arrive as a Poisson process with intensity lambda(delta) =
# A * exp(-k*delta), delta = distance from the touch. The textbook K=1.5
# above is calibrated only for the SPREAD formula (where it enters as a
# gamma/k ratio inside a log -- forgiving of scale). Reusing it directly here,
# where k multiplies a raw dollar delta, is a different story: at BTC's
# typical AS-quoted depth (a few dollars behind a ~$0.10 real market spread),
# K=1.5 pushes fill probability to ~0.001% by $8.5 of depth -- the sim would
# almost never fill. So k_fill is back-solved separately, same pattern as
# gamma_rolling: pin two interpretable targets and solve for the constants.
TARGET_TOUCH_FILL_PROB = 0.90      # guessed P(fill this second) if resting exactly at the real touch
TARGET_FILL_DECAY_USD = 5.0        # guessed $ depth at which that probability has decayed to ~10%
FILL_RNG_SEED = 7                  # fixed seed -- fills are now stochastic, so results must be reproducible


# ---------------------------------------------------------------------------
# 1. Load data (per-second book: mmvol.data.load_book_1s)
# ---------------------------------------------------------------------------
def load_trade_fill_signals(index: pd.DatetimeIndex, date: str):
    """Per-second 'most aggressive print' in each direction, used by the
    trade-through fill rule. NaN where nothing traded that second."""
    path = RAW / f"{SYMBOL}-aggTrades-{date}.csv"
    print(f"[load] {path.name}")
    df = pd.read_csv(
        path,
        usecols=["price", "transact_time", "is_buyer_maker"],
        dtype={"price": "float64", "transact_time": "int64"},
    )
    # CSV stores lowercase true/false; normalize defensively rather than assume dtype inference.
    if df["is_buyer_maker"].dtype != bool:
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.strip().str.lower().eq("true")

    ts = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    sec = ts.dt.floor("1s")

    # is_buyer_maker == True  -> the buyer was resting, i.e. a SELLER hit the bid.
    # is_buyer_maker == False -> the seller was resting, i.e. a BUYER lifted the ask.
    sell_hits = df.loc[df["is_buyer_maker"].values, "price"]
    sell_hits.index = sec[df["is_buyer_maker"].values]
    buy_lifts = df.loc[~df["is_buyer_maker"].values, "price"]
    buy_lifts.index = sec[~df["is_buyer_maker"].values]

    sell_min_by_sec = sell_hits.groupby(level=0).min().reindex(index)   # lowest print that hit the bid this second
    buy_max_by_sec = buy_lifts.groupby(level=0).max().reindex(index)    # highest print that lifted the ask this second
    print(f"  -> {len(df):,} trades ({df['is_buyer_maker'].sum():,} sell-initiated, "
          f"{(~df['is_buyer_maker']).sum():,} buy-initiated)")
    return sell_min_by_sec.values, buy_max_by_sec.values


def calibrate(mid_1s: pd.Series):
    """The one constant-sigma guess, and the gamma back-solved from it so the
    rolling-horizon inventory term targets a plausible $ spread at BTC's price
    level. Shared by every run except run 1 (which deliberately uses the
    textbook gamma to show why that back-solve step is necessary)."""
    s0 = mid_1s.iloc[0]
    sigma_guess = SIGMA_ANNUAL_GUESS * s0 / math.sqrt(SECONDS_PER_YEAR)  # USD per sqrt(second)
    gamma_rolling = TARGET_MAX_SPREAD_USD / (sigma_guess**2 * ROLLING_HORIZON_SEC)
    return sigma_guess, gamma_rolling


def solve_fill_params():
    """A, k_fill for lambda(delta) = A*exp(-k_fill*delta), back-solved so
    P(fill this second) = TARGET_TOUCH_FILL_PROB at delta=0 and decays to
    ~10% of that by TARGET_FILL_DECAY_USD. See the constants above for why
    this is a separate k from the spread-formula K."""
    A = -math.log(1.0 - TARGET_TOUCH_FILL_PROB)
    k_fill = -math.log(0.10) / TARGET_FILL_DECAY_USD
    return A, k_fill


# ---------------------------------------------------------------------------
# 2. AS quote loop + fill rule
# ---------------------------------------------------------------------------
def run_as_sim(mid_1s, sell_min, buy_max, gamma, k, sigma, horizon_sec, clip_size, label,
                fill_mode="trade_through", best_bid=None, best_ask=None, fill_A=None, fill_k=None, rng=None,
                uniforms=None, verbose=True):
    """fill_mode:
      "trade_through" (original, runs 1-2): a resting quote fills for certain
          the instant any print crosses it. No latency, no queue, no
          competition -- deterministic and, as run 2/3 showed, fills on
          ~48% of all seconds. Kept as-is for the historical baselines.
      "intensity" (run 3 onward): a print crossing the quote is necessary
          but not sufficient. Whether it actually fills is a Poisson draw
          with rate lambda(delta) = fill_A*exp(-fill_k*delta), delta = our
          quote's distance BEHIND the real market touch (best_bid/best_ask,
          not mid) -- the AS model's own fill-intensity assumption, applied
          against real trade-through events instead of a separate simulated
          arrival process, so real order flow still gates every fill.

    uniforms: optional (n, 2) array of pre-drawn U(0,1) values for the bid
      and ask fill draws at each second. Passing the same array to every
      estimator gives them common random numbers: identical fill luck at
      each second, so P&L differences between estimators aren't just
      different draws. Without it, draws come from `rng` in the order fills
      are attempted, which differs between estimators.
    """
    n = len(mid_1s)
    mid_vals = mid_1s.values
    sigma_arr = np.full(n, sigma, dtype="float64") if np.isscalar(sigma) else np.asarray(sigma, dtype="float64")
    if fill_mode == "intensity":
        best_bid_vals, best_ask_vals = best_bid.values, best_ask.values
        rng = rng or np.random.default_rng(FILL_RNG_SEED)

    bid = np.empty(n)
    ask = np.empty(n)
    reservation = np.empty(n)
    inventory = np.zeros(n)
    cash = np.zeros(n)
    pnl_mtm = np.zeros(n)
    filled_bid = np.zeros(n, dtype=bool)
    filled_ask = np.zeros(n, dtype=bool)

    inv = 0.0
    csh = 0.0
    degenerate_quotes = 0  # bid <= 0 or bid >= ask, i.e. the formula produced a nonsense quote
    opportunities_bid = opportunities_ask = 0  # intensity mode only: trade-throughs that didn't convert to a fill

    for t in range(n):
        time_remaining = horizon_sec - (t % horizon_sec)
        m = mid_vals[t]
        sig = sigma_arr[t]

        r = m - inv * gamma * sig**2 * time_remaining
        spread = gamma * sig**2 * time_remaining + (2.0 / gamma) * math.log(1.0 + gamma / k)
        b = r - spread / 2.0
        a = r + spread / 2.0

        if b <= 0 or b >= a:
            degenerate_quotes += 1

        sm, bx = sell_min[t], buy_max[t]
        trade_through_bid = not np.isnan(sm) and sm <= b
        trade_through_ask = not np.isnan(bx) and bx >= a

        if fill_mode == "trade_through":
            do_fill_bid, do_fill_ask = trade_through_bid, trade_through_ask
        else:
            do_fill_bid = do_fill_ask = False
            if trade_through_bid:
                delta_bid = max(0.0, best_bid_vals[t] - b)
                p_bid = 1.0 - math.exp(-fill_A * math.exp(-fill_k * delta_bid))
                u = uniforms[t, 0] if uniforms is not None else rng.random()
                do_fill_bid = u < p_bid
                opportunities_bid += 1
            if trade_through_ask:
                delta_ask = max(0.0, a - best_ask_vals[t])
                p_ask = 1.0 - math.exp(-fill_A * math.exp(-fill_k * delta_ask))
                u = uniforms[t, 1] if uniforms is not None else rng.random()
                do_fill_ask = u < p_ask
                opportunities_ask += 1

        if do_fill_bid:
            inv += clip_size
            csh -= clip_size * b
            filled_bid[t] = True
        if do_fill_ask:
            inv -= clip_size
            csh += clip_size * a
            filled_ask[t] = True

        bid[t], ask[t], reservation[t] = b, a, r
        inventory[t] = inv
        cash[t] = csh
        pnl_mtm[t] = csh + inv * m

    out = pd.DataFrame({
        "mid": mid_vals, "bid": bid, "ask": ask, "reservation": reservation,
        "sigma": sigma_arr, "inventory": inventory, "cash": cash, "pnl": pnl_mtm,
        "filled_bid": filled_bid, "filled_ask": filled_ask,
    }, index=mid_1s.index)

    if not verbose:
        return out

    n_bid_fills, n_ask_fills = filled_bid.sum(), filled_ask.sum()
    ann_vol_equiv = sigma_arr / mid_vals * math.sqrt(SECONDS_PER_YEAR)  # for interpretability only
    print(f"\n[{label}] gamma={gamma:.6g}  horizon={horizon_sec}s  fill_mode={fill_mode}  "
          f"spread(min/mean/max)=({(ask-bid).min():.2f}/{(ask-bid).mean():.2f}/{(ask-bid).max():.2f}) USD")
    print(f"  sigma as annualized-equivalent vol (min/mean/max)="
          f"({ann_vol_equiv.min():.0%}/{ann_vol_equiv.mean():.0%}/{ann_vol_equiv.max():.0%})")
    if fill_mode == "intensity":
        print(f"  trade-through opportunities: {opportunities_bid} bid, {opportunities_ask} ask  "
              f"-> converted to fills: {n_bid_fills} ({n_bid_fills / max(1, opportunities_bid):.1%}), "
              f"{n_ask_fills} ({n_ask_fills / max(1, opportunities_ask):.1%})")
    print(f"  fills: {n_bid_fills} buys, {n_ask_fills} sells  "
          f"| final inventory {inv:.3f} BTC | final PnL(mtm) ${pnl_mtm[-1]:,.2f} "
          f"| degenerate quotes: {degenerate_quotes}/{n} seconds")
    return out


def run_as_sim_batch(mid, sell_min, buy_max, best_bid, best_ask, sigma, uniforms, seed_idx,
                      gamma, k, horizon_sec, clip_size, fill_A, fill_k, quote_mid=None):
    """Same quote loop and fill-intensity model as run_as_sim(fill_mode=
    "intensity", uniforms=...), but for B runs at once: one per row of
    `sigma` (B, n). Row j uses uniforms[seed_idx[j]] (uniforms: (S, n, 2)),
    so runs sharing a seed get common random numbers. Looping over time
    once, with numpy over the batch, is ~100x faster than B scalar runs.

    quote_mid: price the quotes are built around; defaults to `mid` (the last
    mid of the same second, which is a lookahead: see causal_inputs). P&L is
    always marked at `mid`.

    Returns arrays laid out (n, B): pnl, bid, ask, filled_bid, filled_ask.
    """
    B, n = sigma.shape
    quote_mid = mid if quote_mid is None else quote_mid
    sig2 = np.ascontiguousarray((sigma ** 2).T)                    # (n, B)
    u_bid = np.ascontiguousarray(uniforms[seed_idx, :, 0].T)       # (n, B)
    u_ask = np.ascontiguousarray(uniforms[seed_idx, :, 1].T)
    const = (2.0 / gamma) * math.log(1.0 + gamma / k)

    pnl = np.empty((n, B))
    bid = np.empty((n, B))
    ask = np.empty((n, B))
    filled_bid = np.zeros((n, B), dtype=bool)
    filled_ask = np.zeros((n, B), dtype=bool)
    inv = np.zeros(B)
    cash = np.zeros(B)

    for t in range(n):
        tr = horizon_sec - (t % horizon_sec)
        m = mid[t]
        risk = gamma * sig2[t] * tr
        r = quote_mid[t] - inv * risk
        half = (risk + const) / 2.0
        b = r - half
        a = r + half

        sm = sell_min[t]
        if not np.isnan(sm):
            delta = np.maximum(0.0, best_bid[t] - b)
            fb = (sm <= b) & (u_bid[t] < 1.0 - np.exp(-fill_A * np.exp(-fill_k * delta)))
            if fb.any():
                inv += clip_size * fb
                cash -= clip_size * b * fb
                filled_bid[t] = fb
        bx = buy_max[t]
        if not np.isnan(bx):
            delta = np.maximum(0.0, a - best_ask[t])
            fa = (bx >= a) & (u_ask[t] < 1.0 - np.exp(-fill_A * np.exp(-fill_k * delta)))
            if fa.any():
                inv -= clip_size * fa
                cash += clip_size * a * fa
                filled_ask[t] = fa

        bid[t], ask[t] = b, a
        pnl[t] = cash + inv * m

    return {"pnl": pnl, "bid": bid, "ask": ask, "filled_bid": filled_bid, "filled_ask": filled_ask}


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------
def plot_run(df: pd.DataFrame, label: str, slug: str, zoom_start: str, date: str, zoom_hours: float = 2.0):
    tight_start = pd.Timestamp(f"{date} {zoom_start}+00:00") + pd.Timedelta(hours=zoom_hours / 2)
    panels = [
        ("full_day", df),
        (f"zoom_{zoom_start.replace(':', '')}", _zoom(df, zoom_start, zoom_hours, date)),
        (f"tight_{tight_start.strftime('%H%M%S')}", df.loc[tight_start:tight_start + pd.Timedelta(minutes=12)]),
    ]
    for suffix, sub in panels:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1, 1, 1]})
        fig.suptitle(f"AS skeleton -- {label}  ({SYMBOL} {date}, {suffix})", fontsize=11, color=PALETTE["ink"])

        ax = axes[0]
        ax.plot(sub.index, sub["mid"], color=PALETTE["mid"], lw=0.9, label="mid")
        ax.plot(sub.index, sub["bid"], color=PALETTE["bid"], lw=0.7, label="AS bid", alpha=0.9)
        ax.plot(sub.index, sub["ask"], color=PALETTE["ask"], lw=0.7, label="AS ask", alpha=0.9)
        fb = sub[sub["filled_bid"]]
        fa = sub[sub["filled_ask"]]
        ax.scatter(fb.index, fb["bid"], color=PALETTE["bid"], s=10, zorder=5, label="buy fill")
        ax.scatter(fa.index, fa["ask"], color=PALETTE["ask"], s=10, zorder=5, label="sell fill")
        ax.set_ylabel("USD")
        ax.legend(loc="upper left", ncol=5, fontsize=8, frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax = axes[1]
        ax.plot(sub.index, sub["inventory"], color=PALETTE["series"], lw=1.1)
        ax.axhline(0, color=PALETTE["axis"], lw=0.8)
        ax.set_ylabel("inventory (BTC)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax = axes[2]
        ax.plot(sub.index, sub["pnl"], color=PALETTE["series"], lw=1.1)
        ax.axhline(0, color=PALETTE["axis"], lw=0.8)
        ax.set_ylabel("PnL, mark-to-market (USD)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax = axes[3]
        ann_vol = sub["sigma"] / sub["mid"] * math.sqrt(SECONDS_PER_YEAR)
        ax.plot(sub.index, ann_vol, color=PALETTE["series"], lw=1.1)
        ax.set_ylabel("sigma, annualized-\nequivalent vol")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("time (UTC)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = OUT / f"{slug}_{suffix}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  [plot] {out_path}")


def _zoom(df, start, hours, date):
    start_ts = pd.Timestamp(f"{date} {start}+00:00")
    end_ts = start_ts + pd.Timedelta(hours=hours)
    return df.loc[start_ts:end_ts]
