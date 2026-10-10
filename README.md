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
src/estimators.py          realized vol, bipower variation, EWMA, GARCH(1,1),
                            HAR-RV -- all causal (no lookahead)
src/as_skeleton.py         the original build-up: runs 1-3
src/compare_estimators.py  all five estimators through the identical harness,
                            scored on P&L and on QLIKE forecast loss
src/signature_plot.py      realized variance vs sampling interval
src/download_data.py       pulls one day of bookTicker + aggTrades
paper/references.bib       references for the write-up
paper/literature_notes.md  what each reference is used for + verification status
output/                    per-run charts, comparison charts, signature plot
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
6. **Estimator set grounded in the literature.** The first three
   estimators were picked by intuition. A literature review
   (`paper/literature_notes.md`) added the two most widely used ones that
   were missing: **EWMA** (RiskMetrics, the industry default) and **HAR-RV**
   (Corsi 2009, the standard academic forecasting benchmark, which beats
   GARCH on Bitcoin in two recent studies). Range-based estimators
   (Parkinson, Garman-Klass) were left out: they're built for daily OHLC bars.
   Adapting the daily-data versions to 1-second data needed two choices:
   EWMA's lambda is set so its average data age matches RV's 300s window
   (RiskMetrics' 0.94 is for daily returns), and HAR uses 1/5/30-minute
   components to forecast the next 5-minute horizon. Each estimator is now
   also scored on **QLIKE** forecast loss (Patton 2011), the metric the
   forecasting literature uses, so the results can be compared with it.
   P&L is reported after the 3-hour burn-in too, since GARCH and HAR are
   fit in-sample on it.
7. **Signature plot.** Checked whether 1-second sampling biases RV, as the
   microstructure-noise literature warns.
8. **A fair P&L comparison.** Single-run P&L turned out to be confounded
   (see findings), so `compare_estimators.py` now:
   - runs every estimator twice: **raw**, and **level-matched**, rescaled by
     one constant so its average variance over the burn-in equals the
     realized variance there. The factor uses burn-in data only, so it's
     causal. This stops an estimator that runs low from "winning" just by
     quoting tighter.
   - runs **20 fill-simulation seeds** with **common random numbers**:
     within a seed, every estimator gets the same random fill draw at each
     second, so estimators are compared on the same luck.
   - adds cross-regime metrics: **P&L per unit of volatility** (P&L divided
     by clip size times the dollar volatility of the mid), a **Sharpe-style
     ratio** of 5-minute P&L changes, and **edge** per dollar traded.
   - writes every run to `output/results_<date>.csv`, ready for aggregation
     across days.

## Key findings so far

- A literal 24h AS horizon is unusable at BTC's price scale (see RUN 1).
- Naive realized vol from raw 1-second returns is noisy: a single outlier
  tick can swing the estimate 10x within a 300s window.
- GARCH(1,1) fit on 1-second BTC futures returns comes out essentially
  non-mean-reverting (alpha+beta ~= 1) -- a genuine property of the data at
  this frequency, not an artifact.
- **The fill rule mattered more than any estimator choice.** Under the
  original (unrealistic) deterministic fill rule, the first three estimators
  produced nearly identical day-end P&L (within 2% of each other) on
  2024-01-10. Once fills were made realistic, P&L dropped ~68% overall and
  the estimators differentiated.
- **1-second RV isn't inflated by noise the textbook way.** The usual
  warning is that bid-ask bounce inflates RV at short intervals. On this
  mid-price data the opposite happens on the calm day: 79% of 1-second
  returns are exactly zero and returns are positively autocorrelated at
  1-second lags (+0.13), so 1-second RV is ~17% *below* its peak at
  15-20s sampling. On the volatile day it's ~6% below the peak at 5s. So
  the bias depends on the regime, and noise-robust estimators designed for
  bounce noise (realized kernels, pre-averaging) don't address it. See
  `output/signature_plot.png`.
- **Forecast accuracy (raw QLIKE, after the burn-in, lower is better)
  matches the literature:**

  | Estimator | Volatile day (2024-01-10) | Calm day (2023-07-19) |
  |---|---|---|
  | GARCH | **0.509** | 0.385 |
  | HAR | 0.638 | **0.273** |
  | EWMA | 0.560 | 0.444 |
  | RV | 0.595 | 0.545 |
  | BV | 1.586 | 8.992 |

  Fitted models (GARCH, HAR) forecast best, as Hansen & Lunde (2005) and
  the Bitcoin HAR studies would predict. GARCH beats EWMA on both days, so
  fitting the parameters adds something over a fixed lambda.
- **The earlier "BV wins on the volatile day" result was an artifact.**
  Bipower variation runs low on this data (0.79x the realized variance on
  the volatile day, 0.47x on the calm day): with so many zero 1-second
  returns, most `|r_t||r_{t-1}|` products are zero. Lower sigma means
  tighter quotes and more fills. With raw sigma BV beats RV by $22 on the
  volatile day (paired t = +2.3 over 20 seeds); level-matched, it trails
  RV by $18 (t = -1.7). Even level-matched, BV is the worst forecaster on
  both days, so **BV is dropped as a standalone estimator** from here on.
  The literature mostly uses it as a jump component inside HAR, which can
  be revisited.
- **The earlier "ranking flips by regime" result doesn't survive a fair
  comparison either, but a different regime effect does.** Level-matched
  P&L after the burn-in, mean of 20 seeds, with the paired difference vs
  RV and its t-statistic:

  | Estimator | Volatile day | vs RV | Calm day | vs RV |
  |---|---|---|---|---|
  | RV | $205 | -- | $26 | -- |
  | EWMA | $187 | -$18 (t=-2.6) | $29 | +$2.6 (t=+3.0) |
  | GARCH | $190 | -$15 (t=-1.5) | $36 | +$9.2 (t=+7.0) |
  | HAR | $189 | -$15 (t=-1.4) | $35 | +$8.3 (t=+7.1) |
  | BV | $187 | -$18 (t=-1.7) | $19 | -$7.0 (t=-6.6) |

  - **Calm day:** clear, significant differences, and the P&L ranking
    matches the forecast-accuracy ranking (GARCH ≈ HAR > EWMA > RV > BV).
    Better forecasts make more money.
  - **Volatile day:** nothing significantly beats RV. Fill and inventory
    noise is large (paired std ~$30–48), so estimator differences mostly
    disappear inside it.
  - Likely reason GARCH and HAR don't help on the volatile day: they're fit
    once on 00:00–03:00 UTC, which was calm; the day turned volatile after
    12:00. Both then under-forecast (HAR at 0.62x realized variance), a
    classic fit-once failure. Periodic refitting should help.
- **Volatility-normalized P&L works as a cross-regime metric.** Raw P&L
  differs 8x between the days (RV: $205 vs $26), but P&L per unit of
  volatility is close (16.3 vs 15.1). So the normalization takes out most
  of the "calm days offer less" effect.
- Caveat on the t-statistics: seeds only vary the fill simulation. Each
  day's price path is fixed, so these say whether a difference is real *on
  that day*, not whether it holds across days. That needs more days.

## Status / what's next

Still **two days** of data. The comparison is now fair: level-matched,
multi-seed, with regime-comparable metrics. It suggests better forecasters
make more money in calm conditions, while in volatile conditions the
differences drown in fill and inventory noise. Two days can't establish
that. Next:

1. The walk-forward: many days across the ~10.5-month window, with regime
   labels computed from past data only, and estimators compared within
   each regime across days (Diebold-Mariano or Model Confidence Set for
   QLIKE, paired tests for P&L). BV is dropped from these runs.
2. Refit GARCH and HAR periodically (e.g. hourly on a trailing window)
   instead of once per day; the volatile-day results suggest fit-once is a
   real handicap.
3. Robustness checks on the judgment-call parameters: the 300s window,
   gamma, the fill-model targets, and the burn-in length.
4. Calibrating `k` from `bookDepth` instead of guessing it.

## Running it

```bash
pip install -r requirements.txt
python src/download_data.py --date 2024-01-10
python src/download_data.py --date 2023-07-19
python src/as_skeleton.py               # add --date YYYY-MM-DD for any other day
python src/compare_estimators.py --seeds 20     # add --date YYYY-MM-DD for any other day (~5 min/day)
python src/signature_plot.py --dates 2024-01-10 2023-07-19
```
