# Market Making Volatility Research

Empirical research into how different volatility estimation methods affect an
Avellaneda-Stoikov (2008) market maker's behavior and P&L, on real BTC data,
across market regimes.

## Research question

Does the choice of volatility estimator (realized vol, bipower variation,
GARCH, ...) meaningfully change a market maker's quotes and P&L -- and does
the answer depend on the market regime? The plan is to build a trustworthy
end-to-end pipeline first, on almost no data, and only then scale up to a
proper multi-day / multi-regime study.

## Data

- **Source:** [Binance's free public archive](https://data.binance.vision),
  BTCUSDT USDS-M perpetual futures.
- **Feeds used:** `bookTicker` (L1 best bid/ask, tick-by-tick) and
  `aggTrades` (executed trades, buyer/seller-tagged).
- **Why futures, not spot:** spot BTCUSDT was never published with
  `bookTicker` on this archive at all; futures `bookTicker` exists but only
  for 2023-05-16 through 2024-03-30 (~10.5 months, then discontinued). This
  window bounds how far the "different regimes" phase of this research can go
  without paying for data or capturing a live feed.
- **On L2 depth:** true tick-level order book data isn't free from this
  source. A coarser free alternative exists (`bookDepth`: cumulative depth in
  +/-1%...+/-5% bands from mid, sampled every 30s) but isn't wired in yet --
  see Status below.
- Is Avellaneda-Stoikov valid on a perpetual future? Yes -- the model is
  asset-agnostic. It doesn't price funding-rate carry or margin/leverage, but
  both are second-order at the position sizes and horizons used here.

## Repo layout

```
src/pipeline.py            shared harness: data loading, calibration,
                            the AS quote loop + fill models, plotting
src/estimators.py          realized vol, bipower variation, GARCH(1,1)
                            -- all causal (no lookahead)
src/as_skeleton.py         the original build-up: runs 1-3
src/compare_estimators.py  RV vs BV vs GARCH through the identical harness
src/download_data.py       pulls one day of bookTicker + aggTrades
output/                    per-run charts + the estimator comparison chart
requirements.txt
```

## Approach so far (one variable at a time)

1. **Skeleton (RUN 1/2).** Loaded one day of data, wrote the AS formula with
   a hardcoded constant sigma. RUN 1 (literal textbook full-day horizon) is
   left in deliberately -- it misbehaves (spreads to $176k), proving a 24h AS
   horizon is unusable at BTC's price scale. RUN 2 fixes this with a rolling
   5-minute horizon and a back-solved gamma.
2. **First real estimator (RUN 3).** Swapped the constant sigma for a causal
   rolling realized-vol estimate. Worked cleanly and tracked real regime
   shifts in the data.
3. **Estimator comparison.** Added bipower variation (jump-robust) and
   GARCH(1,1) (fit on a burn-in window, then run forward as an online
   recursion). Surfaced a real bug: GARCH's fitted alpha+beta ~= 1
   (near unit-root at 1-second granularity) breaks the textbook long-run-
   variance seeding formula (divides by ~0) -- fixed by seeding empirically.
4. **Fill-rule realism.** Replaced the deterministic "any trade-through
   fills you" rule with a Poisson fill-intensity model,
   `lambda(delta) = A * exp(-k*delta)`, where delta is the real distance from
   the quote to the actual market touch (not mid). This closed a gap where
   the AS `k` parameter shaped the spread but never touched fills. Reusing
   the spread-formula `k` directly here was checked and rejected -- it would
   make fills almost impossible beyond ~$2.50 of depth -- so a separate
   `k_fill` is back-solved from interpretable targets instead (90% fill
   chance at the touch, ~10% by $5 of depth).

5. **Second day, chosen deliberately.** `DATE` was a hardcoded module
   constant everywhere -- a real generalization risk. Refactored every
   function in `pipeline.py` to take `date` as a parameter, and both scripts
   now take `--date`. Rather than pick a second day arbitrarily, pulled
   monthly `klines` (daily candles) across the whole bookTicker window to
   find a genuine contrast: 2024-01-10 (the original day) turned out to be an
   elevated-vol day (7.4% daily range, top quartile -- it's also the day the
   SEC approved spot Bitcoin ETFs), so the second day, **2023-07-19**, was
   chosen to be calm (1.45% daily range) while matching 2024-01-10's daily
   *volume* closely (298k vs. 299k median) -- isolating the volatility
   contrast without also changing the liquidity regime.

## Key findings so far

- A literal 24h AS horizon is unusable at BTC's price scale (see RUN 1).
- Naive realized vol from raw 1-second returns is noisy: a single outlier
  tick can swing the estimate 10x within a 300s window.
- GARCH(1,1) fit on 1-second BTC futures returns comes out essentially
  non-mean-reverting (alpha+beta ~= 1) -- a genuine property of the data at
  this frequency, not an artifact.
- **The fill rule mattered more than any estimator choice.** Under the
  original (unrealistic) deterministic fill rule, all three estimators
  produced nearly identical day-end P&L (within 2% of each other) on
  2024-01-10. Once fills were made realistic, P&L dropped ~68% overall and
  the estimators clearly differentiated: bipower variation ($268) > realized
  vol ($221) > GARCH ($136). Bipower's jump-robustness keeps quotes tighter
  and closer to the real touch, which converts directly into more fills
  under a distance-decaying fill probability.
- **The estimator ranking flips by regime.** On the calm day (2023-07-19),
  GARCH wins clearly ($29.00) while realized vol and bipower variation are
  statistically indistinguishable ($18.20 vs. $18.49 -- within what a
  different random seed could produce). This is the first real evidence for
  the project's core hypothesis, but it's a two-day, single-seed-per-run
  result: not yet a claim, a reason to build the walk-forward.
- Raw $ P&L is **not** a fair way to compare across regimes -- a calmer day
  mechanically offers less to capture regardless of estimator quality. A real
  cross-regime comparison needs a normalized metric (e.g. P&L per unit of
  realized volatility), which doesn't exist yet.

## Status / what's next

Everything above is still **two days** of data -- enough to show the
pipeline generalizes and to get one genuinely interesting (ranking-flips)
result, not enough to claim one estimator is "better" in any regime. Not yet
started:

- A normalized cross-regime P&L metric (see above) -- needed before more
  days are added, or every new day just adds another incomparable number.
- Calibrating the AS `k` parameter from real book-depth data (`bookDepth`)
  instead of guessing it.
- The actual walk-forward phase: many days across the ~10.5-month window,
  with causal (no-lookahead) regime labeling, so an estimator ranking could
  be statistically tested instead of eyeballed from two charts.

## Running it

```bash
pip install -r requirements.txt
python src/download_data.py --date 2024-01-10
python src/as_skeleton.py               # add --date YYYY-MM-DD for any other day
python src/compare_estimators.py        # add --date YYYY-MM-DD for any other day
```
