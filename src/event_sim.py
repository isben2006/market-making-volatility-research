"""
Event-driven Avellaneda-Stoikov simulator on raw Binance L1 + trade events.

Replaces the 1-second-bar simulator in pipeline.py, which had two problems:
  1. Lookahead: each second's quote used that second's LAST mid, then got
     filled by trades from earlier in the same second. That hid adverse
     selection: fills were followed by favorable moves (+$2.57 per fill 10s
     later on 2024-01-10), which can't happen in a real market.
  2. Quotes sat still for a full second, while a typical 1-second move on a
     volatile day ($4.30) is a large fraction of the half-spread. Real market
     makers requote on every tick.

How this one works:
  - Decisions every DECISION_MS (100 ms by default). Each uses only
    information available before it: the latest L1 mid before the decision
    time, the volatility estimate from the last completed second, and the
    current inventory.
  - Quotes are rounded to the $0.10 tick (bid down, ask up) and reach the
    exchange LATENCY_MS later. They're post-only: a quote that would cross
    the book on arrival is rejected for that interval.
  - Each quote rests from its arrival until the next quote arrives, and can
    be filled by events in that window. We only have L1 + trades, so queue
    position behind the best price is unknown; results are bracketed by two
    deterministic rules:
      conservative  filled only if a trade prints strictly through our price
                    (the whole level, including us, was swept) or the
                    opposite best price moves to or through our price.
      optimistic    also filled by trades exactly at our price, up to their
                    quantity, as if we were first in the queue.
    A fill through our level fills our whole remaining quantity; fills are
    at our quote price.
  - Fees aren't charged here (AS doesn't price them); notional traded is
    recorded so any maker fee can be applied afterwards.

No randomness: the same inputs always give the same fills.
"""
import math

import numpy as np
import pandas as pd

from pipeline import DAY_SECONDS, RAW, SYMBOL

TICK = 0.1
EPS = 1e-6


def load_events(date: str):
    bt = pd.read_csv(RAW / f"{SYMBOL}-bookTicker-{date}.csv",
                     usecols=["best_bid_price", "best_ask_price", "transaction_time"],
                     dtype={"best_bid_price": "float64", "best_ask_price": "float64", "transaction_time": "int64"})
    tr = pd.read_csv(RAW / f"{SYMBOL}-aggTrades-{date}.csv", usecols=["price", "quantity", "transact_time",
                     "is_buyer_maker"], dtype={"price": "float64", "quantity": "float64", "transact_time": "int64"})
    if tr["is_buyer_maker"].dtype != bool:
        tr["is_buyer_maker"] = tr["is_buyer_maker"].astype(str).str.strip().str.lower().eq("true")

    day0 = int(pd.Timestamp(f"{date} 00:00:00", tz="UTC").timestamp() * 1000)
    bt_ts = bt["transaction_time"].values - day0
    order = np.argsort(bt_ts, kind="stable")
    book = {"ts": bt_ts[order], "bid": bt["best_bid_price"].values[order], "ask": bt["best_ask_price"].values[order]}

    tr_ts = tr["transact_time"].values - day0
    order = np.argsort(tr_ts, kind="stable")
    tr = tr.iloc[order]
    sell = tr["is_buyer_maker"].values  # buyer was resting => a seller hit the bid
    trades = {
        "sell_ts": tr_ts[order][sell], "sell_px": tr["price"].values[sell], "sell_qty": tr["quantity"].values[sell],
        "buy_ts": tr_ts[order][~sell], "buy_px": tr["price"].values[~sell], "buy_qty": tr["quantity"].values[~sell],
    }
    return book, trades


def mid_1s_from_book(book: dict, index: pd.DatetimeIndex) -> pd.Series:
    """Last mid of each second (forward-filled, back-filled before the first
    update): the same series pipeline.load_book_1s builds, without reading
    the CSV a second time."""
    ts = book["ts"]
    last = np.clip(np.searchsorted(ts, (np.arange(DAY_SECONDS) + 1) * 1000, side="left") - 1, 0, len(ts) - 1)
    return pd.Series((book["bid"][last] + book["ask"][last]) / 2.0, index=index)


def _window_reduce(ts, values, edges, fn, empty):
    """fn-reduce `values` over events with ts in [edges[k], edges[k+1]) for each k; `empty` if none."""
    lo = np.searchsorted(ts, edges[:-1], side="left")
    hi = np.searchsorted(ts, edges[1:], side="left")
    out = np.full(len(lo), empty)
    has = hi > lo
    if has.any():
        # windows are contiguous (hi[k] == lo[k+1]), so reduceat over the lo's reduces exactly [lo, hi);
        # the extra end index (and the neutral sentinel it may point at) closes the last window at hi[-1].
        red = fn.reduceat(np.r_[values, empty], np.r_[lo, hi[-1]])[:-1]
        out[has] = red[has]
    return out, lo, hi


def prepare_intervals(book, trades, decision_ms: int, latency_ms: int):
    """Everything the decision loop needs per decision interval k: the mid at
    decision time, the book when the quote arrives, and what happens while
    it rests, i.e. during [k*D + L, (k+1)*D + L)."""
    n_dec = DAY_SECONDS * 1000 // decision_ms
    dec_t = np.arange(n_dec) * decision_ms
    arrive = dec_t + latency_ms
    edges = np.r_[arrive, arrive[-1] + decision_ms]

    ts, bid, ask = book["ts"], book["bid"], book["ask"]
    i_dec = np.clip(np.searchsorted(ts, dec_t, side="left") - 1, 0, len(ts) - 1)   # last update before decision
    i_arr = np.clip(np.searchsorted(ts, arrive, side="left") - 1, 0, len(ts) - 1)  # book when quote arrives
    ask_min_in, _, _ = _window_reduce(ts, ask, edges, np.minimum, np.inf)
    bid_max_in, _, _ = _window_reduce(ts, bid, edges, np.maximum, -np.inf)
    sell_min, s_lo, s_hi = _window_reduce(trades["sell_ts"], trades["sell_px"], edges, np.minimum, np.inf)
    buy_max, b_lo, b_hi = _window_reduce(trades["buy_ts"], trades["buy_px"], edges, np.maximum, -np.inf)
    return {
        "n_dec": n_dec, "dec_t": dec_t,
        "quote_mid": (bid[i_dec] + ask[i_dec]) / 2.0,
        "bid_arrive": bid[i_arr], "ask_arrive": ask[i_arr],
        "ask_min_in": ask_min_in, "bid_max_in": bid_max_in,
        "sell_min": sell_min, "s_lo": s_lo, "s_hi": s_hi,
        "buy_max": buy_max, "b_lo": b_lo, "b_hi": b_hi,
    }


def reach_distances(iv: dict):
    """Per decision window, how far from the decision-time mid the market
    reached on each side, through either trades or the opposite best price.
    A quote at distance delta would have been filled (conservative rule)
    when the reach is at least delta."""
    qm = iv["quote_mid"]
    reach_bid = qm - np.minimum(iv["sell_min"], iv["ask_min_in"])
    reach_ask = np.maximum(iv["buy_max"], iv["bid_max_in"]) - qm
    return reach_bid, reach_ask


INTENSITY_DELTAS = np.round(np.arange(0.1, 12.01, 0.1), 2)


def fit_intensity(reach_bid: np.ndarray, reach_ask: np.ndarray, window_sec: float, deltas=INTENSITY_DELTAS):
    """Fit lambda(delta) = A exp(-k delta), the rate (per second) at which a
    quote at distance delta from mid gets filled, averaged over both sides.
    This is Avellaneda-Stoikov's own calibration idea, applied to our fill
    rule. Returns (A, k)."""
    secs = len(reach_bid) * window_sec
    lam = []
    for reach in (reach_bid, reach_ask):
        srt = np.sort(reach)
        lam.append((len(srt) - np.searchsorted(srt, deltas - EPS, side="left")) / secs)
    lam = (lam[0] + lam[1]) / 2.0
    ok = lam > 0
    slope, icpt = np.polyfit(deltas[ok], np.log(lam[ok]), 1)
    return math.exp(icpt), -slope


def calibrate_k_walkforward(iv: dict, decision_ms: int, first_fit_sec: int, refit_sec: int, train_sec: int):
    """k per decision, refit every refit_sec on the trailing train_sec (data
    before the refit time only). The first fit also covers the burn-in."""
    reach_bid, reach_ask = reach_distances(iv)
    dec_s = iv["dec_t"] / 1000.0
    fit_times = np.arange(first_fit_sec, DAY_SECONDS, refit_sec)
    fits = []
    for T in fit_times:
        m = (dec_s >= T - train_sec) & (dec_s < T)
        fits.append(fit_intensity(reach_bid[m], reach_ask[m], decision_ms / 1000.0))
    fits = np.array(fits)
    seg = np.maximum(np.searchsorted(fit_times, dec_s, side="right") - 1, 0)
    return fits[seg, 1], fits


def decision_sigma(sigma_sec: np.ndarray, dec_t: np.ndarray, lagged: bool) -> np.ndarray:
    """Per-second sigma -> per-decision sigma, using only completed seconds.
    lagged=True for estimators whose value at second s includes that second's
    return (RV, EWMA, HAR, level-matched): use second s-1. GARCH's value at s
    is already built from returns up to s-1, so it's used as is."""
    sec = dec_t // 1000
    idx = np.maximum(sec - 1, 0) if lagged else sec
    return sigma_sec[idx]


def run_event_sim(iv: dict, trades: dict, mark_mid_1s: np.ndarray, sigma_dec: np.ndarray, optimistic: np.ndarray,
                  gamma: float, k: float, horizon_sec: int, clip_size: float, decision_ms: int,
                  trace_run: int = None, k_idx: np.ndarray = None):
    """B runs at once: sigma_dec (B, n_dec); optimistic (B,) bool picks each run's fill rule.
    gamma: scalar or (B,) per run. k: scalar, (n_dec,) per decision (e.g.
    from calibrate_k_walkforward), or (n_dec, M) with k_idx (B,) choosing
    column k_idx[j] for run j -- lets runs with different k series share one
    pass over the data.

    Returns per-second arrays (DAY_SECONDS, B): pnl (marked to the mid at the
    end of each second), fills, notional, spread_sum, inv_abs_sum, plus
    per-run counts of post-only rejections. With trace_run=j, also returns
    every fill of run j as (decision index, +1 buy / -1 sell, price, qty)."""
    trace = []
    B, n_dec = sigma_dec.shape
    per_sec = 1000 // decision_ms
    gam = np.broadcast_to(np.asarray(gamma, dtype=float), (B,)).copy()
    k_arr = np.asarray(k, dtype=float)
    if k_arr.ndim == 2:
        k_dec = np.ascontiguousarray(k_arr)
        k_cols = np.asarray(k_idx)
    else:
        k_dec = np.broadcast_to(k_arr, (n_dec,))
        k_cols = None
    sig2 = np.ascontiguousarray((sigma_dec ** 2).T)
    qmid, bid_arr, ask_arr = iv["quote_mid"], iv["bid_arrive"], iv["ask_arrive"]
    ask_min_in, bid_max_in, sell_min, buy_max = iv["ask_min_in"], iv["bid_max_in"], iv["sell_min"], iv["buy_max"]
    s_lo, s_hi, b_lo, b_hi = iv["s_lo"], iv["s_hi"], iv["b_lo"], iv["b_hi"]
    s_px, s_q, b_px, b_q = trades["sell_px"], trades["sell_qty"], trades["buy_px"], trades["buy_qty"]

    out = {key: np.zeros((DAY_SECONDS, B)) for key in ["pnl", "fills", "notional", "spread_sum", "inv_abs_sum"]}
    inv, cash = np.zeros(B), np.zeros(B)
    acc_fills, acc_notional, acc_spread, acc_inv = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
    rejected = np.zeros(B)
    clip = np.full(B, clip_size)

    for kk in range(n_dec):
        tr = horizon_sec - ((kk * decision_ms / 1000.0) % horizon_sec)
        risk = gam * sig2[kk] * tr
        r = qmid[kk] - inv * risk
        k_now = k_dec[kk] if k_cols is None else k_dec[kk][k_cols]
        half = (risk + (2.0 / gam) * np.log1p(gam / k_now)) / 2.0
        b = np.floor((r - half) / TICK + EPS) * TICK
        a = np.ceil((r + half) / TICK - EPS) * TICK
        bid_ok = b < ask_arr[kk] - EPS  # post-only: a bid at or above the ask would cross
        ask_ok = a > bid_arr[kk] + EPS
        rejected += ~bid_ok
        rejected += ~ask_ok

        # bid side: sells hitting bids, or the ask moving down to our bid
        fb = np.zeros(B)
        through = (sell_min[kk] < b - EPS) | (ask_min_in[kk] <= b + EPS)
        fb[through] = clip_size
        lo, hi = s_lo[kk], s_hi[kk]
        if hi > lo and optimistic.any():
            at = np.abs(s_px[lo:hi][None, :] - b[:, None]) < EPS
            qty_at = (at * s_q[lo:hi][None, :]).sum(axis=1)
            fb = np.where(~through & optimistic, np.minimum(clip, qty_at), fb)
        fb *= bid_ok

        # ask side: buys lifting asks, or the bid moving up to our ask
        fa = np.zeros(B)
        through = (buy_max[kk] > a + EPS) | (bid_max_in[kk] >= a - EPS)
        fa[through] = clip_size
        lo, hi = b_lo[kk], b_hi[kk]
        if hi > lo and optimistic.any():
            at = np.abs(b_px[lo:hi][None, :] - a[:, None]) < EPS
            qty_at = (at * b_q[lo:hi][None, :]).sum(axis=1)
            fa = np.where(~through & optimistic, np.minimum(clip, qty_at), fa)
        fa *= ask_ok

        if trace_run is not None:
            if fb[trace_run] > 0:
                trace.append((kk, 1, b[trace_run], fb[trace_run]))
            if fa[trace_run] > 0:
                trace.append((kk, -1, a[trace_run], fa[trace_run]))

        inv += fb - fa
        cash += a * fa - b * fb
        acc_fills += (fb > 0).astype(float) + (fa > 0).astype(float)
        acc_notional += a * fa + b * fb
        acc_spread += a - b
        acc_inv += np.abs(inv)

        if (kk + 1) % per_sec == 0:
            s = kk // per_sec
            out["pnl"][s] = cash + inv * mark_mid_1s[s]
            out["fills"][s] = acc_fills
            out["notional"][s] = acc_notional
            out["spread_sum"][s] = acc_spread
            out["inv_abs_sum"][s] = acc_inv
            acc_fills[:], acc_notional[:], acc_spread[:], acc_inv[:] = 0, 0, 0, 0

    out["rejected"] = rejected
    out["per_sec"] = per_sec
    if trace_run is not None:
        out["trace"] = np.array(trace, dtype=float).reshape(-1, 4)
    return out
