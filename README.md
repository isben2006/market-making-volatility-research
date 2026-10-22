# Market Making Volatility Research

Does the choice of volatility estimator matter for an Avellaneda-Stoikov
(2008) market maker? An empirical study on BTCUSDT perpetual futures, across
volatility regimes, with an event-driven backtest on tick data.

## Research question

Avellaneda-Stoikov (AS) quotes depend on volatility, but the model takes
sigma as given. The forecasting literature has a clear view of which
estimators forecast volatility best. This project asks whether that ranking
carries over to market making: **does a better volatility forecast make an
AS market maker earn more, or lose less, and does the answer depend on the
regime?**

## Summary of results

221 trading days, four estimators (RV, EWMA, GARCH, HAR), three volatility
regimes, four risk-aversion levels.

1. **The estimators differ clearly as forecasters.** HAR has the lowest
   forecast loss in every regime; realized volatility (RV) is the worst and
   falls out of the Model Confidence Set in all three.
2. **But they don't change market-making P&L.** None of the 36 estimator
   comparisons (3 estimators × 3 regimes × 4 risk aversions) survives a
   multiple-testing correction, here or in any of 6 robustness settings
   (k calibration, decision speed, latency; run on a 41-day sample).
3. **The null is precise.** Over whole days, every estimator is
   statistically equivalent to RV within $3–12 a day (1–5% of the loss for
   γ ≤ 1e-3). By regime, 27 of 36 comparisons are equivalent within ±10% of
   gross spread revenue. Effects of up to ~5% can't be excluded in every
   regime; detecting them would need years of data that doesn't exist.
4. **Why: adverse selection dominates.** Calibrated AS loses money at every
   risk aversion, with both fill bounds, before and after fees. Within 5
   seconds of each fill the price moves against the market maker by about
   twice the spread it captured. The estimators differ mainly in how much
   they trade, and trading less captures less spread but also takes less
   adverse selection, so totals don't move.
5. **Could any sigma help?** Yes, but not these. Perfect foresight of
   volatility is worth $18–29 a day; most of that comes from knowing the
   coming hour's volatility *level*, which none of the four short-horizon
   estimators targets. Even the perfect forecast only reaches break-even
   before fees, at the highest risk aversion, and loses about $95 a day
   after them.

This matches Falces Marín et al. (2022), whose tuned AS has a negative
Sharpe ratio on 28 of 30 BTC days.

## Data

- **Source:** [Binance's free public archive](https://data.binance.vision),
  BTCUSDT USDⓈ-M perpetual futures.
- **Feeds:** `bookTicker` (every change to the best bid/ask, millisecond
  timestamps) and `aggTrades` (every trade, with size and aggressor side).
- **Window:** futures `bookTicker` is only published from 2023-05-16 to
  2024-03-30; spot never had it. That bounds the study period.
- **Days:** every weekday in the window, 228, minus 7 with broken data:
  **221 days** (`data/study_days.csv`). Weekends are excluded: their much
  lower volume would mix a liquidity regime into the volatility comparison.
- **Data quality** (`src/scripts/check_data_quality.py`,
  `data/data_quality.csv`): Binance's archive has holes, stretches with no
  quote updates while trades keep printing (14 minutes and 268,529 trades
  on 2024-01-03). A simulated market maker would quote around a frozen price
  and get filled by every trade, building positions of up to 100 BTC. Every
  day is checked; a day is excluded if any quote gap of 30 s or more has
  trades in it, or if quotes are missing at the start or end of the day.
  7 days fail. Among the rest, 99% have no quote gap over 6 s. One kept day
  (2023-09-12) has a 20-minute gap with no trades at all, i.e. an exchange
  pause, which the simulator handles correctly (no fills).
- **No order book depth (L2):** the free archive has no depth behind the
  best price, so queue position is unknown. The simulator brackets it
  instead (see Method).

An earlier version of the study used a stratified sample of 41 of these
days (`data/study_days_41.csv`). The power analysis (Results, section 4)
showed regime-level comparisons needed more days, so the study now uses
all of them. None of the 41 days fail the data check, and their
conclusions match.

## Method

### Volatility estimators (`src/mmvol/estimators.py`)

All causal: the value in force at any moment uses only earlier data.

| Estimator | Specification | Why |
|---|---|---|
| **RV** | rolling mean of squared 1-second returns, 300 s window | Literature benchmark (Liu, Patton & Sheppard 2015); what Hummingbot's AS uses |
| **EWMA** | RiskMetrics recursion, λ set so its average data age matches RV's window (RiskMetrics' 0.94 is for daily data) | Industry default |
| **GARCH(1,1)** | refit hourly on the trailing 3 hours | Standard conditional-variance benchmark (Hansen & Lunde 2005) |
| **HAR-RV** | 1/5/30-minute components forecasting the next 5 minutes; refit hourly on the day so far, non-negative coefficients | Standard realized-vol forecasting benchmark (Corsi 2009); beats GARCH on Bitcoin in two recent studies |

Each estimator's sigma goes straight into AS, unadjusted, so any level
bias counts as part of that estimator's quality. (A variant rescaled every
hour to the trailing 3-hour realized variance is kept as a robustness check:
its factors are ~1, since all four estimators come from the same 1-second
returns, and it gives the same conclusions.) The first 3 hours of each day
are burn-in; scoring starts at 03:00 UTC.

### Benchmarks

Not estimators: they bracket the four, run through the same simulator, and
are kept out of the estimator comparisons and the Model Confidence Set.

| Benchmark | Sigma | Tradable? |
|---|---|---|
| **Flat** | trailing 3-hour realized vol, updated hourly, constant within the hour | yes |
| **OracleHour** | the current hour's actual realized vol, constant within the hour | no |
| **Oracle** | the actual realized vol of the next 5 minutes (QLIKE's target) | no |

Flat → OracleHour is the value of knowing the hour's volatility level;
OracleHour → Oracle is the value of also knowing its timing within the hour.

### Market maker

- AS quotes with a rolling 5-minute horizon (a literal full-day horizon gives
  absurd spreads at BTC's price level).
- **k**, AS's order-flow parameter, is calibrated from the data every hour:
  fit λ(δ) = A·e^(−kδ), the rate at which a quote δ dollars from the mid
  gets filled, on the trailing 3 hours. The fits have R² ≈ 0.98–0.995; k
  ranges from about 0.2 per dollar in volatile hours to about 1.2 in calm
  ones.
- **Gamma** (risk aversion) has no data-driven value, so results are
  reported on a grid: 1e-5, 1e-4, 1e-3, 1e-2.
- Fixed clip of 0.005 BTC per order.

### Event-driven simulator (`src/mmvol/simulator.py`)

- **Decisions:** every 100 ms, using only information available before the
  decision: latest mid, sigma from the last completed second, current
  inventory.
- **Orders:** reach the exchange after 50 ms latency, are rounded to the
  $0.10 tick, and are post-only (a quote that would cross on arrival is
  rejected). Each quote rests until the next one arrives.
- **Fills:** queue position behind the best price is unknown, so results
  are bracketed by two deterministic rules:
  - **Conservative:** filled only if a trade prints strictly through our
    price, or the opposite best price reaches it. Either means our whole
    price level was taken.
  - **Optimistic:** also filled by trades exactly at our price, up to their
    size, as if we were first in the queue.

  Any true queue model lands between the two. In practice they're a few
  percent apart, because AS quotes sit a few dollars from a $0.10-wide
  market, where fills come from sweeps rather than queueing.
- **Fees:** applied afterwards from traded notional (0 and 2 bps, roughly
  Binance's standard maker fee).
- **P&L decomposition:** each second's P&L is split exactly into
  - **spread captured:** each fill's distance from the mid its quote was
    based on;
  - **adverse selection:** the mid's move against each fill over the next
    H seconds (H = 5, and 60 as a check);
  - **inventory:** everything else, i.e. price moves on older positions.

  This mirrors the effective / realized spread split of the
  microstructure literature (Huang & Stoll 1996).
- Deterministic, and tested on hand-built fill and decomposition scenarios
  (`src/tests/test_simulator.py`).

### Evaluation

- **Regimes:** each scored hour is labeled low, mid or high volatility by
  its trailing 3-hour realized vol (60 s returns, data before the hour
  only), split into terciles across all 4,641 scored hours. Cutoffs: 29% and
  46% annualized. Low, mid and high hours occur on 170, 209 and 189 of the
  221 days.
- **Metrics:**
  - **P&L per unit volatility:** P&L divided by clip size times the dollar
    volatility of the mid. Comparable across regimes.
  - **QLIKE forecast loss** (Patton 2011), the forecasting literature's
    standard.
  - **Day-level risk:** max drawdown, P&L relative to mean absolute
    inventory, and a Sharpe-style ratio of 5-minute P&L changes.
- **Inference:** the unit is day × regime, since hours within a day share
  one price path and carry inventory over. Each estimator is compared to RV
  with paired tests across days, a Bonferroni correction over the 36
  comparisons, and the 10% Model Confidence Set (Hansen, Lunde & Nason
  2011).
- **Precision of the null** (`src/scripts/analyze_power.py`): 95% confidence
  intervals; the minimum detectable effect (the smallest true effect the
  study had 80% power to detect); and two one-sided tests (TOST) for
  equivalence, reported as the smallest margin at which equivalence holds,
  so no margin has to be chosen after seeing the data.

## Results

All results: 221 days, raw sigma, conservative fills, no fees, unless
stated. Robustness (section 7) uses the earlier 41-day sample.

### 1. AS economics (baseline)

RV, mean per day over the 21 scored hours:

| γ | avg spread | fills/day | edge (bps of notional) | P&L/day, no fees | P&L/day, 2 bps fee | days positive |
|---|---|---|---|---|---|---|
| 1e-5 | $4.88 | 18,653 | −0.83 | −$350 | −$1,193 | 58/221 |
| 1e-4 | $5.09 | 16,959 | −0.74 | −$281 | −$1,035 | 29/221 |
| 1e-3 | $7.15 | 9,985 | −0.62 | −$128 | −$540 | 18/221 |
| 1e-2 | $27.79 | 2,679 | −0.38 | −$19 | −$119 | 77/221 |

Higher γ loses less mainly by trading less: going from 1e-5 to 1e-2 cuts
fills by 86% and the loss per fill by about 60%, but never makes it
positive. The optimistic bound is within 1–2% of these figures.

### 2. Forecast accuracy

Median QLIKE across days (lower is better), with the number of days each
estimator beats RV, and membership in the 10% Model Confidence Set:

| Regime (days) | RV | EWMA | GARCH | HAR |
|---|---|---|---|---|
| low (170) | 0.724 ✗ | 0.583, 165/170 ✗ | 0.496, 139/170 ✗ | **0.416**, 155/170 ✓ |
| mid (209) | 0.414 ✗ | 0.377, 201/209 ✓ | 0.361, 147/209 ✓ | **0.355**, 162/209 ✓ |
| high (189) | 0.275 ✗ | 0.250, 177/189 ✗ | 0.279, 112/189 ✓ | **0.237**, 147/189 ✓ |

(✓ in the Model Confidence Set, ✗ excluded.) HAR has the lowest loss and
stays in the confidence set in every regime; RV is excluded in all three.
EWMA beats RV most consistently, on 94–97% of days.

### 3. Estimator vs market-making P&L

36 comparisons: 3 estimators vs RV × 3 regimes × 4 gammas.

- **No significant differences.** 6 of 36 reach |t| ≥ 2, about what chance
  alone gives (1.8 expected); none survive Bonferroni. The median
  difference is 2% of RV's level. The same holds under the optimistic bound
  (6 / 0) and with level-matched sigma (4 / 0).
- **One weak pattern:** EWMA does slightly better than RV in low volatility
  at every γ ≥ 1e-4 (t = 2.1, 2.6, 3.0), just short of the Bonferroni
  threshold. It may be a small real effect (section 4).
- **With a 2 bps fee,** 10 of 36 survive Bonferroni, because fees charge for
  volume: estimators that trade less (mostly EWMA) pay less in fees. The
  directions match the fill-count differences in section 5, so this is a
  volume effect, not better forecasting.

Chart: `output/walkforward_event/regime_comparison_raw_conservative_fee0.png`.

### 4. How precise is the null? Power and equivalence

A non-significant result can mean "no effect" or "too little data to see
one". These numbers separate the two.

Whole days, estimator − RV in $/day:

| γ | RV P&L | EWMA: 95% CI / equivalent within | GARCH | HAR | Oracle − RV |
|---|---|---|---|---|---|
| 1e-5 | −$350 | [−0.9, +5.2] / ±$4.7 | [−7.1, +7.0] / ±$5.9 | [−10.5, +7.7] / ±$9.1 | +$18.4 |
| 1e-4 | −$281 | [−2.3, +4.6] / ±$4.0 | [−8.5, +4.6] / ±$7.5 | [−13.5, +1.2] / ±$12.3 | +$28.5 |
| 1e-3 | −$128 | [−0.6, +5.5] / ±$5.0 | [−2.4, +6.3] / ±$5.6 | [−7.3, +4.3] / ±$6.3 | +$26.9 |
| 1e-2 | −$19 | [−3.7, +1.2] / ±$3.3 | [−4.3, +3.6] / ±$3.6 | [−0.2, +7.2] / ±$6.6 | +$21.6 |

"Equivalent within" is the smallest margin at which a 5% TOST passes.

- **Whole days:** every estimator is within $3–12 a day of RV, 1–5% of the
  loss for γ ≤ 1e-3, and well under what perfect volatility foresight is
  worth. The minimum detectable effect is 1.5–9% of RV's spread captured
  (median 3.6%).
- **By regime** (36 cells, as % of RV's spread captured in that cell):
  equivalence holds within ±5% in 14 cells, ±10% in 27 and ±20% in 35. The
  median minimum detectable effect is 7.5% (10.9% at the Bonferroni level).
  The loosest cells are in low volatility and at γ = 1e-2, where revenue is
  small.
- **Sample size:** for effects of 10% of spread revenue the sample is
  adequate (the typical comparison needs 29–240 days). Detecting 5% effects
  in individual regimes would need roughly 500–4,000 days: years of data
  that doesn't exist in the free archive. Those effects are bounded by the
  equivalence tests instead. With the original 41 days the typical
  whole-day detectable effect was 8.4%, and only 12 of 36 regime cells were
  equivalent within ±10%.

So the claim is not "zero effect" but: estimator effects are within a few
percent over whole days and within ±10–20% by regime, too small to change
whether AS loses money.

Table: `output/walkforward_event/power_raw_conservative_fee0.csv`.

### 5. Why: P&L decomposition

RV, mean per day; adverse selection over 5 s:

| γ | spread captured | adverse selection | inventory | total | per BTC traded: spread / adverse selection |
|---|---|---|---|---|---|
| 1e-5 | +$281 | −$600 | −$31 | −$350 | $2.49 / −$5.28 |
| 1e-4 | +$282 | −$560 | −$3 | −$281 | $2.69 / −$5.44 |
| 1e-3 | +$195 | −$337 | +$13 | −$128 | $3.41 / −$6.01 |
| 1e-2 | +$60 | −$92 | +$12 | −$19 | $4.53 / −$6.92 |

**The loss is adverse selection:** about twice the spread captured, at
every γ, at the 60 s horizon too and under the optimistic bound. Inventory
P&L is small.

Estimators vs RV, per component, by regime (36 cells each, Bonferroni
across the 36):

| Component | Bonferroni-significant | \|t\| ≥ 2 |
|---|---|---|
| total P&L | 0 | 2 |
| inventory | 0 | 2 |
| spread captured | 18 | 28 |
| adverse selection | 15 | 21 |
| fills | 21 | 28 |
| adverse selection per fill | 14 | 22 |

- **The estimators differ mainly in volume.** Trading less captures less
  spread and takes less adverse selection, by similar amounts, so totals
  don't move. Example, γ = 1e-2, whole days: EWMA makes 119 fewer fills a
  day than RV (t = −35), captures $2.86 less spread (t = −18), takes $4.08
  less adverse selection (t = +18), and differs by −$1.20 in total.
- **Per-fill toxicity differs, but only by a few percent, in both
  directions.** EWMA and GARCH take $0.01–0.09 more adverse selection per
  BTC than RV (under 1.5%); HAR takes $0.015–0.07 less at γ ≤ 1e-3 but
  $0.17–0.21 more at γ = 1e-2 (about 3%). These offset against the volume
  effects.
- **Inventory**, the channel through which sigma would still act if adverse
  selection were removed, shows no estimator differences (0 of 36).

Tables: `output/walkforward_event/decomposition_*.csv`.

### 6. Could any sigma help? Benchmarks

Whole-day P&L vs RV, $/day (t-stat):

| γ | RV level | Flat | EWMA | GARCH | HAR | OracleHour | Oracle |
|---|---|---|---|---|---|---|---|
| 1e-5 | −$350 | +12 (0.9) | +2 | 0 | −1 | +13 (1.8) | +18 (2.5) |
| 1e-4 | −$281 | −19 (−2.3) | +1 | −2 | −6 | +19 (4.4) | +29 (6.6) |
| 1e-3 | −$128 | −28 (−5.1) | +2 | +2 | −2 | +10 (3.3) | +27 (6.8) |
| 1e-2 | −$19 | −15 (−3.7) | −1 | 0 | +3 | +16 (6.1) | +22 (8.9) |

Splitting the oracle's value (paired across days):

| γ | level: OracleHour − Flat | timing: Oracle − OracleHour |
|---|---|---|
| 1e-5 | +1 (t=0.1) | +6 (t=1.7) |
| 1e-4 | +38 (t=4.3) | +10 (t=2.8) |
| 1e-3 | +38 (t=7.3) | +17 (t=5.0) |
| 1e-2 | +31 (t=6.9) | +6 (t=2.6) |

- **At γ = 1e-5 sigma barely enters the quotes.** The volatility term
  γσ²τ is cents, so the spread is set by k; even perfect foresight adds
  only ~5% of the loss.
- **At higher γ a better sigma is worth $22–29 a day**, more than the whole
  gross loss at γ = 1e-2. The four estimators capture almost none of it
  (section 4 bounds them at a fraction of the oracle's gain).
- **Most of that value is the hourly level** ($31–38 a day), with timing
  within the hour adding $6–17. Some dynamics beat none: Flat, which only
  knows the trailing 3-hour level, is clearly worse than RV at γ ≥ 1e-4.
- **How the oracles gain** (decomposition): both mainly through inventory
  (OracleHour +$23 a day at γ = 1e-4, t = 5.1; Oracle +$28, t = 6.5). Only
  the Oracle also reduces adverse selection per fill, by $0.39 per BTC at
  γ = 1e-2 (t = 13; $0.96 at the 60 s horizon): it widens its quotes just
  before sweeps.
- **Sigma doesn't rescue AS.** At γ = 1e-2 the Oracle just breaks even
  before fees (+$2.50 a day, positive on 121 of 221 days) and loses about
  $95 a day after 2 bps fees; at lower γ it loses $100–330 a day before
  fees.
- The same pattern holds under the optimistic bound. Caveat: knowing an
  hour's realized vol also tells you a spike is coming sometime that hour,
  so some timing information leaks into OracleHour.

Chart: `output/walkforward_event/benchmarks_raw_conservative_fee0.png`.

### 7. Robustness (`src/scripts/run_robustness.py`, `src/scripts/analyze_robustness.py`)

The 41-day sample rerun with one change each (conservative fills, no fees):

| Setting | RV P&L/day at γ=1e-3 | Estimator vs RV: \|t\| ≥ 2 / Bonferroni-significant (of 36) | Typical difference vs RV |
|---|---|---|---|
| base (100 ms, 50 ms latency, k refit hourly) | −$151 | 5 / 0 | 4.5% |
| k fit once per day | −$171 | 3 / 0 | 4.4% |
| k constant (0.46/$) | −$189 | 4 / 0 | 3.5% |
| 250 ms decisions | −$133 | 1 / 0 | 2.9% |
| 1 s decisions | −$117 | 1 / 0 | 4.9% |
| 10 ms latency | −$138 | 2 / 0 | 4.5% |
| 200 ms latency | −$164 | 1 / 0 | 3.8% |

- **AS loses money in every setting, at every gamma.** Slower requoting
  loses less only because it trades less.
- **Holding k constant doesn't reveal hidden estimator differences.** Hourly
  k carries volatility information of its own, so it could have masked
  them. With k constant, differences get smaller, not larger.
- **Day-level risk metrics:** nothing survives the correction. One weak
  tendency: every nominally significant GARCH result on risk-adjusted P&L
  (7 of 7) points to GARCH doing 4–14% worse than RV. All settings reuse
  the same days, so that's a tendency, not a finding.

### 8. Microstructure note: the signature plot

The usual warning is that bid-ask bounce inflates realized volatility at
very short sampling intervals. On Binance perp mid-quotes the opposite
happens on calm days: 79% of 1-second returns are exactly zero and
returns are positively autocorrelated at 1-second lags (+0.13), so
1-second RV sits ~17% *below* its peak at 15–20 s sampling. On a volatile
day it's ~6% below. So the bias depends on the regime, and noise-robust
estimators designed for bounce noise don't address it.
`output/signature_plot.png`

## Pitfalls found along the way

Each of these changed the results at some point. The full step-by-step
history is in the git log.

- **Same-second lookahead.** An early simulator built each second's quote
  from that second's *last* mid, then filled it with trades from *earlier*
  in the same second. That hid adverse selection completely: fills looked
  favorable (+$2.57 markout after 10 s) and the strategy looked profitable.
  With causal quotes, the same strategy went from +$204 to −$458 on the
  volatile development day. Every P&L result before the fix was discarded.
- **Holes in the archive.** Extending the sample from 41 to all weekdays
  first multiplied the average loss fivefold: on 5 days the quote data
  stops for minutes while trades continue, and the simulator built
  positions of up to 100 BTC. Every day is now screened (Data).
- **Fill rules.** The first fill rule (any trade through the quote fills
  it) and a later one with guessed fill probabilities both overstated
  fills. They were replaced by the two data-driven bounds.
- **Single days mislead.** On two development days the estimator ranking
  appeared to flip between calm and volatile regimes, with large t-stats.
  Over many days the effect disappeared: the t-stats only reflected
  fill-simulation noise on one fixed price path.
- **An estimator's level could masquerade as skill.** One that runs low
  quotes tighter, trades more, and can look better for that reason alone.
  A level-matched variant was built to control for it; the levels turned
  out to agree (matching factors ~1), so the main results use the raw sigma
  and level-matching is a robustness check.
- **GARCH on 1-second data** fits as nearly integrated (α+β ≈ 1), which
  breaks the textbook long-run variance used to start the recursion
  (division by ~0). It's started from the empirical variance instead.
- **HAR on short windows** is unstable: refit on 3 hours, it produced
  negative coefficients and negative variance forecasts. Fixed with an
  expanding window and non-negative coefficients.
- **A literal 24-hour AS horizon** produces spreads of over $100,000 at
  BTC's price level. Hence the rolling 5-minute horizon.

## Limitations

- **Simulated fills.** Queue position is bracketed, not modeled. Free
  first-of-month L2 data from [Tardis.dev](https://tardis.dev) could place
  real fills within the bracket, but can't move results outside it.
- **The strategy is plain AS.** It has no adverse-selection signals
  (order-book imbalance, cross-venue leads), no queue priority, no fee
  rebates and no hedging, which is how real market makers stay profitable.
  The question here is about the volatility input, not about building a
  profitable strategy.
- **One asset, one exchange, 221 days.** The free archive caps the period;
  small effects (~5%) in individual regimes can't be excluded (Results,
  section 4). Within the sample, the regime label partly tracks time of
  day: high-volatility hours cluster in the US session.
- **Robustness checks use the 41-day sample.**
- **Some settings were chosen on the two development days**, which are part
  of the sample: refit windows and the decision/latency baseline. The
  constant k for the robustness check uses the whole 41-day sample.
- **The oracles are upper bounds,** not achievable targets.
- **Fee levels are approximate:** 2 bps is Binance's standard maker fee;
  check the 2023–24 schedule.
- **Inventory updates instantly** on a fill, so the market maker learns of
  its fills slightly earlier than it would in reality.

## Future work

- **Forecasting the hourly volatility level.** OracleHour suggests the
  valuable input is the coming hour's volatility level, which none of the
  four short-horizon estimators targets. An hourly HAR with time-of-day
  seasonality (volatility rises reliably around the US open), used as the
  level anchor under the existing estimators, would test how much of the
  ~$30–38 a day is achievable.
- **Volatility forecasts under adverse-selection protection.** Re-run the
  comparison with a strategy that defends against toxic fills (e.g.
  microprice or order-book-imbalance skew, Stoikov 2018), to test whether
  estimator choice matters once adverse selection no longer dominates. The
  decomposition suggests it may not: estimators already don't differ in
  the inventory component.
- **Queue-aware fills** on L2 data, to place results within the
  conservative / optimistic bracket.
- **Other assets and venues.**

## Running it

Scripts run as modules from the `src/` folder. A day takes about 45 s to
simulate plus its download (two workers in parallel).

```bash
pip install -r requirements.txt
cd src

python -m scripts.check_data_quality --workers 2   # writes data/data_quality.csv (downloads every day)
python -m scripts.select_days                      # writes data/study_days.csv (all weekdays that pass)
python -m scripts.run_walkforward --workers 2      # the study; deletes raw data after each day
python -m scripts.analyze_walkforward              # options: --rule optimistic | --fee-bps 2 | --variant matched | --markout 60
python -m scripts.analyze_power                    # confidence intervals, detectable effects, equivalence
python -m scripts.run_robustness --workers 3 --dates $(tail -n +2 ../data/study_days_41.csv | cut -d, -f1)
python -m scripts.analyze_robustness
python -m tests.test_simulator                     # fill-rule and decomposition checks

# signature plot (needs the two development days on disk)
python -m scripts.download_data --date 2024-01-10
python -m scripts.download_data --date 2023-07-19
python -m scripts.signature_plot --dates 2024-01-10 2023-07-19
```

## Repo layout

```
src/
  mmvol/                      core library
    config.py                 paths, instrument, shared settings
    data.py                   Binance archive download, loaders, quote-gap check
    estimators.py             volatility estimators (causal) and benchmarks
    simulator.py              event-driven simulator, P&L decomposition, k calibration
    walkforward.py            per-day pipeline: build estimators, score hours
    analysis.py               regime labels, paired tests, Model Confidence Set
    plotting.py               chart style
  scripts/                    entry points
    check_data_quality.py     quote-gap screen for every day
    select_days.py            study days: all weekdays that pass the screen
    run_walkforward.py        the study
    analyze_walkforward.py    main results, benchmarks, P&L decomposition, charts
    analyze_power.py          confidence intervals, detectable effects, equivalence
    run_robustness.py         k / decision-interval / latency variants
    analyze_robustness.py     robustness summary
    signature_plot.py         RV vs sampling interval
    download_data.py          download one day
  tests/
    test_simulator.py         hand-built fill-rule and decomposition checks
  legacy/                     superseded 1-second-bar pipeline (invalid P&L;
                              kept so the history can be reproduced)
data/study_days.csv           the 221 study days
data/study_days_41.csv        the earlier 41-day sample (used by the robustness checks)
data/data_quality.csv         per-day quote coverage and gaps
paper/references.bib          references for the write-up
paper/literature_notes.md     what each reference is used for
output/walkforward_event/     main results: per-day files, summaries, charts
output/robustness_event/      robustness results
output/signature_plot.png
output/ (other files)         outputs from the legacy pipeline
```
