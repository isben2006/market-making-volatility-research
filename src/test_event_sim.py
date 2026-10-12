"""
Hand-built checks of event_sim's fill rules. With sigma = 0, gamma = k = 1,
the quotes are fixed around a 100.0 mid: bid 99.3, ask 100.7. Each decision
interval gets one scripted event and we check what each fill rule does.

Run: python test_event_sim.py
"""
import numpy as np

from event_sim import run_event_sim

CLIP = 0.005
INF = np.inf

# one second of 100ms decisions; each row: (event, expected bid fill, expected ask fill) for (conservative, optimistic)
SCENARIOS = [
    ("nothing happens",                         (0, 0), (0, 0)),
    ("sell 0.002 exactly at our bid 99.3",      (0, 0.002), (0, 0)),
    ("sell strictly through our bid (99.2)",    (CLIP, CLIP), (0, 0)),
    ("best ask drops to our bid (book cross)",  (CLIP, CLIP), (0, 0)),
    ("bid rejected post-only, sell through",    (0, 0), (0, 0)),
    ("buy strictly through our ask (100.8)",    (0, 0), (CLIP, CLIP)),
    ("buy 0.01 exactly at our ask 100.7",       (0, 0), (0, CLIP)),
    ("best bid rises to our ask (book cross)",  (0, 0), (CLIP, CLIP)),
    ("ask rejected post-only, buy through",     (0, 0), (0, 0)),
    ("nothing happens",                         (0, 0), (0, 0)),
]


def build():
    n = len(SCENARIOS)
    iv = {"n_dec": n, "dec_t": np.arange(n) * 100,
          "quote_mid": np.full(n, 100.0), "bid_arrive": np.full(n, 99.9), "ask_arrive": np.full(n, 100.1),
          "ask_min_in": np.full(n, INF), "bid_max_in": np.full(n, -INF),
          "sell_min": np.full(n, INF), "buy_max": np.full(n, -INF)}
    sells, buys = [], []  # (interval, price, qty)
    sells.append((1, 99.3, 0.002))
    sells.append((2, 99.2, 1.0))
    iv["ask_min_in"][3] = 99.3
    iv["ask_arrive"][4] = 99.3          # our 99.3 bid would cross on arrival -> rejected
    sells.append((4, 99.0, 1.0))
    buys.append((5, 100.8, 1.0))
    buys.append((6, 100.7, 0.01))
    iv["bid_max_in"][7] = 100.7
    iv["bid_arrive"][8] = 100.7
    buys.append((8, 101.0, 1.0))

    trades = {}
    for side, lst, key_min, fn in [("sell", sells, "sell_min", min), ("buy", buys, "buy_max", max)]:
        lst.sort()
        trades[f"{side}_px"] = np.array([p for _, p, _ in lst])
        trades[f"{side}_qty"] = np.array([q for _, _, q in lst])
        k_of = np.array([k for k, _, _ in lst])
        lo = np.searchsorted(k_of, np.arange(n), side="left")
        hi = np.searchsorted(k_of, np.arange(n), side="right")
        iv["s_lo" if side == "sell" else "b_lo"], iv["s_hi" if side == "sell" else "b_hi"] = lo, hi
        for k, p, _ in lst:
            iv[key_min][k] = fn(iv[key_min][k], p)
    return iv, trades


def main():
    iv, trades = build()
    n = iv["n_dec"]
    optimistic = np.array([False, True])
    failures = 0
    # run one interval at a time so each scenario's fills can be read separately
    for kk, (name, exp_bid, exp_ask) in enumerate(SCENARIOS):
        sub = {key: (v[kk:kk + 1] if isinstance(v, np.ndarray) and len(v) == n else v) for key, v in iv.items()}
        sub["n_dec"], sub["dec_t"] = 1, np.array([0])
        sub["s_lo"], sub["s_hi"], sub["b_lo"], sub["b_hi"] = (iv["s_lo"][kk:kk + 1], iv["s_hi"][kk:kk + 1],
                                                              iv["b_lo"][kk:kk + 1], iv["b_hi"][kk:kk + 1])
        out = run_event_sim(sub, trades, mark_mid_1s=np.full(86400, 100.0), sigma_dec=np.zeros((2, 1)),
                            optimistic=optimistic, gamma=1.0, k=1.0, horizon_sec=300, clip_size=CLIP,
                            decision_ms=1000)
        notional = out["notional"][0]
        # recover filled quantities from cash: cash = 100.7*ask_q - 99.3*bid_q, inv = bid_q - ask_q, pnl = cash + 100*inv
        # => pnl = 0.7*(bid_q + ask_q); notional = 99.3*bid_q + 100.7*ask_q
        tot = out["pnl"][0] / 0.7
        ask_q = (notional - 99.3 * tot) / (100.7 - 99.3)
        bid_q = tot - ask_q
        for j, rule in enumerate(["conservative", "optimistic"]):
            ok = abs(bid_q[j] - exp_bid[j]) < 1e-9 and abs(ask_q[j] - exp_ask[j]) < 1e-9
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name:42s} {rule:12s} bid {bid_q[j]:.4f} (exp {exp_bid[j]})  "
                  f"ask {ask_q[j]:.4f} (exp {exp_ask[j]})")
    print(f"\n{failures} failures")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
