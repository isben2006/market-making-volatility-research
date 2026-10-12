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

1. **With realistic fills, calibrated AS loses money on BTC perps** at every
   risk-aversion setting tested, before and after fees. The cause is adverse
   selection: quotes get filled when the price runs through them. This
   matches Falces Marín et al. (2022), whose tuned AS has a negative Sharpe
   ratio on 28 of 30 BTC days.
2. **The estimators differ clearly as forecasters.** EWMA beats realized
   volatility on 30/32, 37/39 and 34/35 days in low, mid and high volatility.
3. **But the estimator barely changes market-making P&L.** Across 252
   estimator comparisons (36 per setting, 7 settings), none survive a
   multiple-testing correction, and typical differences are 3–6% of the
   loss. The forecasting literature's ranking doesn't carry over to market
   making; adverse selection dominates.

## Data

- **Source:** [Binance's free public archive](https://data.binance.vision),
  BTCUSDT USDⓈ-M perpetual futures.
- **Feeds:** `bookTicker` (every change to the best bid/ask, millisecond
  timestamps) and `aggTrades` (every trade, with size and aggressor side).
- **Window:** futures `bookTicker` is only published from 2023-05-16 to
  2024-03-30; spot never had it. That bounds the study period.
- **Days:** 41 weekdays (`data/study_days.csv`), a stratified random sample
  across calm, mid and volatile daily-range terciles, covering all 11 months.
  Weekends are excluded: their much lower volume would mix a liquidity
  regime into the volatility comparison.
- **No order book depth (L2):** the free archive has no depth behind the
  best price, so queue position is unknown. The simulator brackets it
  instead (see Method).

## Method

### Volatility estimators (`src/mmvol/estimators.py`)

All causal: the value in force at any moment uses only earlier data.

| Estimator | Specification | Why |
|---|---|---|
| **RV** | rolling mean of squared 1-second returns, 300 s window | Literature benchmark (Liu, Patton & Sheppard 2015); what Hummingbot's AS uses |
| **EWMA** | RiskMetrics recursion, λ set so its average data age matches RV's window (RiskMetrics' 0.94 is for daily data) | Industry default |
| **GARCH(1,1)** | refit hourly on the trailing 3 hours | Standard conditional-variance benchmark (Hansen & Lunde 2005) |
| **HAR-RV** | 1/5/30-minute components forecasting the next 5 minutes; refit hourly on the day so far, non-negative coefficients | Standard realized-vol forecasting benchmark (Corsi 2009); beats GARCH on Bitcoin in two recent studies |

Each estimator also runs **level-matched**: rescaled every hour so that its
average variance over the trailing 3 hours equals realized variance. An
estimator that runs systematically low would otherwise quote tighter and
"win" on volume rather than forecasting. The first 3 hours of each day are
burn-in; scoring starts at 03:00 UTC.

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

  Any true queue model lands between the two. In practice they're 3–4%
  apart, because AS quotes sit a few dollars from a $0.10-wide market,
  where fills come from sweeps rather than queueing.
- **Fees:** applied afterwards from traded notional (0 and 2 bps, roughly
  Binance's standard maker fee).
- Deterministic, and tested on hand-built fill scenarios
  (`src/tests/test_simulator.py`).

### Evaluation (`src/scripts/run_walkforward.py`, `src/scripts/analyze_walkforward.py`)

- **Regimes:** each scored hour is labeled low, mid or high volatility by
  its trailing 3-hour realized vol (60 s returns, data before the hour
  only), split into terciles across all 861 hours. Cutoffs: 29% and 46%
  annualized.
- **Metrics:**
  - **P&L per unit volatility:** P&L divided by clip size times the dollar
    volatility of the mid. Comparable across regimes.
  - **QLIKE forecast loss** (Patton 2011), the forecasting literature's
    standard.
  - **Day-level risk:** max drawdown, P&L relative to mean absolute
    inventory, and a Sharpe-style ratio of 5-minute P&L changes.
- **Inference:** the unit is day × regime, since hours within a day share
  one price path and carry inventory over. Each estimator is compared to RV
  with paired tests across days, a Bonferroni correction, and the 10% Model
  Confidence Set (Hansen, Lunde & Nason 2011).

## Results

### 1. AS economics

RV, level-matched, mean per day over the 21 scored hours, 41 days:

| γ | avg spread | fills/day | edge (bps of notional) | P&L/day, no fees | P&L/day, 2 bps fee | days positive |
|---|---|---|---|---|---|---|
| 1e-5 | $5.00 | 20,644 | −0.73 | −$348 | −$1,304 | 11/41 |
| 1e-4 | $5.28 | 18,333 | −0.82 | −$342 | −$1,174 | 4/41 |
| 1e-3 | $8.12 | 10,545 | −0.68 | −$151 | −$594 | 2/41 |
| 1e-2 | $36.50 | 2,827 | −0.47 | −$25 | −$134 | 14/41 |

These are conservative-bound figures; the optimistic bound is within a few
percent. The mechanism shows in fill markouts (how far the mid moves after a
fill), measured on the two development days before k was calibrated. On the
calm day the market maker captures $0.96 of spread per fill, but
the mid has moved $1.00 against it within 0.1 s and $2.60 within 5 s. On the
volatile day: $3.41 captured, −$2.17 and −$3.26.

### 2. Forecast accuracy

Median QLIKE across days (lower is better), level-matched, with the number
of days each estimator beats RV:

| Regime (days) | RV | EWMA | GARCH | HAR |
|---|---|---|---|---|
| low (32) | 0.624 | 0.508, 30/32 | 0.503, 21/32 | **0.413**, 25/32 |
| mid (39) | 0.296 | 0.268, 37/39 | 0.343, 24/39 | **0.261**, 28/39 |
| high (35) | 0.245 | **0.224**, 34/35 | 0.248, 18/35 | 0.227, 21/35 |

### 3. Estimator vs market-making P&L

36 comparisons per setting: 3 estimators vs RV × 3 regimes × 4 gammas.

- **Without fees:**
  - The median difference is ~5% of the loss level.
  - Only 2–5 comparisons reach |t| ≥ 2, about what chance alone gives.
  - None survive a Bonferroni correction.
  - Signs flip between gammas.
  - The same holds under the optimistic bound and with raw sigma.
- **With a 2 bps fee** more differences become significant, because fees
  charge for volume. An estimator that over-forecasts quotes wider, trades
  less and pays less in fees. That's a volume effect, not better
  forecasting.

Chart: `output/walkforward_event/regime_comparison_matched_conservative_fee0.png`.

### 4. Robustness (`src/scripts/run_robustness.py`, `src/scripts/analyze_robustness.py`)

The 41 days rerun with one change each (conservative fills, level-matched,
no fees):

| Setting | RV P&L/day at γ=1e-3 | Estimator vs RV: \|t\| ≥ 2 / Bonferroni-significant (of 36) | Typical difference vs RV |
|---|---|---|---|
| base (100 ms, 50 ms latency, k refit hourly) | −$151 | 4 / 0 | 4.9% |
| k fit once per day | −$171 | 3 / 0 | 5.3% |
| k constant (0.46/$) | −$189 | 2 / 0 | 3.0% |
| 250 ms decisions | −$132 | 3 / 0 | 2.7% |
| 1 s decisions | −$117 | 0 / 0 | 4.9% |
| 10 ms latency | −$139 | 4 / 0 | 6.1% |
| 200 ms latency | −$165 | 2 / 0 | 4.2% |

- **AS loses money in every setting, at every gamma.** Slower requoting
  loses less only because it trades less.
- **Holding k constant doesn't reveal hidden estimator differences.** Hourly
  k carries volatility information of its own, so it could have masked
  them. With k constant, differences get smaller, not larger.
- **Day-level risk metrics:** nothing survives the correction. One weak
  tendency: every nominally significant GARCH result on risk-adjusted P&L
  (10 of 10) points to GARCH doing 4–10% worse than RV. All settings reuse
  the same days, so that's a tendency, not a finding.

### 5. Microstructure note: the signature plot

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
- **Fill rules.** The first fill rule (any trade through the quote fills
  it) and a later one with guessed fill probabilities both overstated
  fills. They were replaced by the two data-driven bounds.
- **Single days mislead.** On two development days the estimator ranking
  appeared to flip between calm and volatile regimes, with large t-stats.
  Over 41 days the effect disappeared: the t-stats only reflected
  fill-simulation noise on one fixed price path.
- **An estimator's level can masquerade as skill.** One that runs low quotes
  tighter, trades more, and can look better for that reason alone. Hence
  level-matching.
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
- **One asset, 41 days, one exchange.** Within that, the regime label partly
  tracks time of day: high-volatility hours cluster in the US session.
- **Some settings were chosen on the two development days**, which are part
  of the 41: refit windows and the decision/latency baseline. The constant k
  for the robustness check uses the whole sample.
- **Fee levels are approximate:** 2 bps is Binance's standard maker fee;
  check the 2023–24 schedule.
- **Inventory updates instantly** on a fill, so the market maker learns of
  its fills slightly earlier than it would in reality.

## Running it

Scripts run as modules from the `src/` folder:

```bash
pip install -r requirements.txt
cd src

python -m scripts.select_days                    # writes data/study_days.csv
python -m scripts.run_walkforward --workers 3    # ~20 min: downloads each day, deletes raw data after
python -m scripts.analyze_walkforward            # options: --rule optimistic | --fee-bps 2 | --variant raw
python -m scripts.run_robustness --workers 3     # ~35 min
python -m scripts.analyze_robustness
python -m tests.test_simulator                   # fill-rule checks

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
    data.py                   Binance archive download and loaders
    estimators.py             volatility estimators (causal)
    simulator.py              event-driven simulator, k calibration
    walkforward.py            per-day pipeline: build estimators, score hours
    analysis.py               regime labels, paired tests, Model Confidence Set
    plotting.py               chart style
  scripts/                    entry points
    select_days.py            study-day sample
    run_walkforward.py        the 41-day study
    analyze_walkforward.py    main results tables and chart
    run_robustness.py         k / decision-interval / latency variants
    analyze_robustness.py     robustness summary
    signature_plot.py         RV vs sampling interval
    download_data.py          download one day
  tests/
    test_simulator.py         hand-built fill-rule checks
  legacy/                     superseded 1-second-bar pipeline (invalid P&L;
                              kept so the history can be reproduced)
data/study_days.csv           the 41 study days
paper/references.bib          references for the write-up
paper/literature_notes.md     what each reference is used for
output/walkforward_event/     main results: per-day files, summaries, charts
output/robustness_event/      robustness results
output/signature_plot.png
output/ (other files)         outputs from the legacy pipeline
```
