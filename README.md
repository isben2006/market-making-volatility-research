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
src/select_days.py         picks the 41 study days (stratified by daily range)
src/walkforward.py         the multi-day study: per-hour results for every day
src/analyze_walkforward.py regime labels, paired tests, Model Confidence Set
data/study_days.csv        the 41 study days
paper/references.bib       references for the write-up
paper/literature_notes.md  what each reference is used for + verification status
output/                    per-run charts, comparison charts, signature plot
output/walkforward/        per-day results, summaries, regime chart
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
9. **The walk-forward regime study** (`select_days.py`, `walkforward.py`,
   `analyze_walkforward.py`):
   - **Days:** 41 weekdays, stratified random sample across calm, mid and
     volatile daily-range terciles, covering all 11 months of the data
     window. Weekends are excluded because their much lower volume would
     mix a liquidity regime into the volatility comparison.
   - **Estimators:** RV, EWMA, GARCH, HAR (BV dropped), raw and
     level-matched. GARCH refits hourly on the trailing 3 hours. HAR refits
     hourly on the day so far with non-negative coefficients: a trailing 3h
     window was unstable for HAR (negative coefficients and variance
     forecasts, since 3 hours hold only ~6 independent observations of its
     30-minute component). Level-matching factors are also re-estimated
     hourly on the trailing 3 hours. These window choices were checked on
     the two development days.
   - **Simulation:** a batched simulator runs all 80 combinations
     (4 estimators × 2 variants × 10 seeds) in one pass, with common random
     numbers. It reproduces the original simulator exactly and is ~12x
     faster; a day now takes ~20 seconds including the download.
   - **Scoring:** each hour from 03:00 to 24:00 UTC is scored separately
     (861 hours in total).
   - **Regimes:** each hour is labeled low, mid or high by its *trailing*
     3-hour realized vol (60s returns, data before the hour only), split
     into terciles across all hours. Cutoffs: 29% and 46% annualized.
   - **Inference:** the unit is day × regime, since hours within a day share
     one price path and carry inventory over. Tests are paired differences
     vs RV across days, plus the 10% Model Confidence Set (Hansen, Lunde &
     Nason 2011).

## Main result: event-driven walk-forward (current)

**Setup** (`walkforward_event.py`, `analyze_walkforward_event.py`):
- Same 41 days, regimes and estimators as before, on the event-driven
  simulator: 100 ms requoting, 50 ms latency, tick rounding, post-only
  quotes.
- AS's `k` is calibrated from the data every hour (λ(δ) = A·e^(−kδ) fits
  with R² ≈ 0.98–0.995). It ranges from about 0.2 per dollar in volatile
  hours to about 1.2 in calm ones.
- Gamma is a grid: 1e-5, 1e-4, 1e-3, 1e-2.
- Both fill bounds (conservative and optimistic), raw and level-matched
  sigma, fees applied afterwards (0 and 2 bps).

**1. Calibrated Avellaneda-Stoikov loses money on BTC perps at every gamma,
under both fill bounds** (RV, level-matched, mean per day over 21 hours):

| gamma | avg spread | fills/day | edge (bps of notional) | P&L/day, no fees | P&L/day, 2 bps fee | days positive |
|---|---|---|---|---|---|---|
| 1e-5 | $5.00 | 20,644 | −0.73 | −$348 | −$1,304 | 11/41 |
| 1e-4 | $5.28 | 18,333 | −0.82 | −$342 | −$1,174 | 4/41 |
| 1e-3 | $8.12 | 10,545 | −0.68 | −$151 | −$594 | 2/41 |
| 1e-2 | $36.50 | 2,827 | −0.47 | −$25 | −$134 | 14/41 |

The optimistic bound is within a few percent of the conservative one, so
the losses don't depend on the unknown queue position. The cause is
adverse selection: quotes are only filled when the price runs through them.
This matches Falces Marín et al. (2022), whose genetically tuned AS has a
negative Sharpe ratio on 28 of 30 BTC days even with full L2 data.

**2. Forecast accuracy differs clearly between estimators** (unchanged from
the earlier study, since it doesn't use the simulator). EWMA beats RV on
30/32, 37/39 and 34/35 days in low, mid and high vol; HAR has the best
median in low and mid vol.

**3. But the volatility estimator barely changes market-making P&L.**
- Without fees: 36 comparisons (3 estimators vs RV × 3 regimes × 4 gammas).
  The median difference is ~5% of the loss level, only 2–5 reach |t| ≥ 2
  (about what chance alone gives), none survive a multiple-testing
  correction, and the signs flip between gammas.
- The same holds under the optimistic bound and with raw sigma.
- With a 2 bps fee more differences become significant, because fees charge
  for *volume*. An estimator that over-forecasts quotes wider, trades less
  and pays less in fees: GARCH in high vol at γ=1e-3, +5.1, t=4.4. That's a
  trading-volume effect, not better forecasting.

**Conclusion:** once fills are realistic, adverse selection dominates. Which
volatility estimator an AS market maker uses is a second-order choice for
P&L, even though the estimators differ clearly as forecasters. The
forecasting literature's ranking doesn't carry over to market making.

Chart: `output/walkforward_event/regime_comparison_matched_conservative_fee0.png`.
Summaries for each variant: `output/walkforward_event/summary_*.csv`.

**4. Robustness** (`robustness_event.py`, `analyze_robustness.py`): the 41
days rerun under six variations, each changing one thing (conservative
fills, level-matched sigma, no fees):

| Setting | RV P&L/day at γ=1e-3 | Estimator vs RV, 36 comparisons: \|t\| ≥ 2 / Bonferroni-significant | Typical difference vs RV level |
|---|---|---|---|
| base (100 ms, 50 ms latency, k refit hourly) | −$151 | 4 / 0 | 4.9% |
| k fit once per day (burn-in) | −$171 | 3 / 0 | 5.3% |
| k constant (0.46/$) | −$189 | 2 / 0 | 3.0% |
| 250 ms decisions | −$132 | 3 / 0 | 2.7% |
| 1 s decisions | −$117 | 0 / 0 | 4.9% |
| 10 ms latency | −$139 | 4 / 0 | 6.1% |
| 200 ms latency | −$165 | 2 / 0 | 4.2% |

- **AS loses money in every setting, at every gamma.** Slower requoting
  loses less, simply because it trades less: with 1 s decisions, fills drop
  ~40–45% at the two lowest gammas.
- **Fixing k doesn't reveal estimator differences.** Hourly k carries
  volatility information of its own, so it could have masked differences
  between sigma estimators. With k constant, differences are smaller (3%
  typical), not larger.
- **Across 252 estimator comparisons, none survive a multiple-testing
  correction.** The 0–4 nominal hits per setting are about what chance
  gives (~2).
- **Day-level risk metrics** (max drawdown, P&L to mean absolute inventory,
  Sharpe-style ratio of 5-minute P&L): none survive the correction either.
  One weak tendency: every nominally significant result for GARCH on P&L to
  inventory and on the Sharpe-style ratio (10 of 10) points to GARCH doing
  slightly *worse* than RV, by about 4–10%. All settings reuse the same 41
  days, so this consistency isn't independent confirmation; treat it as a
  tendency, not a finding.

Summaries: `output/robustness_event/summary_*.csv`.

> **Superseded P&L results (2026-10-12).** Every P&L number below, and in
> the "Earlier findings" section, came from a simulator with a lookahead bug.
> Each second's quote used that second's *last* mid and was then filled by
> trades from *earlier* in the same second, which hid adverse selection.
> Fills were followed by price moves in the market maker's favor (+$2.57 per
> fill after 10s), which can't happen in a real market. With causal quotes,
> every estimator loses money on both development days.
>
> The **forecast-accuracy (QLIKE) results stand**: they never used the
> simulator. The P&L comparison is being redone with an event-driven
> simulator (`src/event_sim.py`): 100 ms requoting, order latency, tick
> rounding, post-only quotes, and conservative/optimistic fill bounds. See
> "What's next".

## Main result: 41-day walk-forward

Level-matched sigma; the raw variant gives the same conclusions. QLIKE is
forecast loss (lower is better). P&L is per unit of volatility (higher is
better). "Beats RV" counts the days on which the estimator did better than
RV. Chart: `output/walkforward/regime_comparison_matched.png`.

**Forecast accuracy (median QLIKE across days):**

| Regime (days) | RV | EWMA | GARCH | HAR |
|---|---|---|---|---|
| low (32) | 0.624 | 0.508, beats RV 30/32 | 0.500, 21/32 | **0.413**, 25/32 |
| mid (39) | 0.296 | 0.268, 37/39 | 0.343, 24/39 | **0.261**, 28/39 |
| high (35) | 0.245 | **0.224**, 34/35 | 0.248, 18/35 | 0.227, 21/35 |

**Market-making P&L (per unit vol, paired difference vs RV, t-stat, days beating RV):**

| Regime | RV level | EWMA | GARCH | HAR |
|---|---|---|---|---|
| low | 15.8 | −0.13 (t=−0.3), 15/32 | −0.18 (t=−0.2), 15/32 | −0.14 (t=−0.1), 15/32 |
| mid | 16.0 | +0.13 (t=+0.5), 20/39 | −0.13 (t=−0.2), 17/39 | −0.63 (t=−0.7), 22/39 |
| high | 15.5 | −0.23 (t=−1.0), 18/35 | **−2.57 (t=−4.4), 8/35** | **−2.30 (t=−2.5), 9/35** |

1. **Better forecasts don't produce more market-making P&L.** EWMA and HAR
   forecast better than RV in every regime; EWMA beats it on 30/32, 37/39
   and 34/35 days. Yet in low and mid vol, no estimator's P&L differs from
   RV's: the differences are within 4% of RV's P&L (mostly about 1%),
   t-statistics are near zero, and win rates are near half. All four
   estimators are in the Model Confidence Set.
2. **In high vol, the sophisticated models earn *less*.** GARCH and HAR
   trail RV by 16.5% and 14.8%, and are excluded from the Model Confidence
   Set.
   GARCH's gap (t=−4.4) survives a Bonferroni correction for the 18 tests;
   HAR's (t=−2.5) doesn't on its own, but the Model Confidence Set excludes
   both. The simple estimators, RV and EWMA, are statistically tied.
3. **Mechanism:** in high-vol hours, GARCH and HAR over-forecast. Their
   median forecast is 1.18x and 1.22x the realized variance, versus 1.01x
   for RV and EWMA. High-vol hours tend to follow bursts, and volatility
   then mean-reverts faster than these persistent models expect. Over-
   forecasting widens quotes (average spread $9.24 and $10.57, vs $8.66 for
   RV) and loses fills (456 and 426 per hour, vs 501). QLIKE barely notices,
   because it penalizes under-forecasts much more than over-forecasts. The
   market maker's loss is the opposite way round: over-forecasting costs it
   volume. **The forecasting literature's loss function doesn't match the
   market maker's economic loss.**
4. **Volatility-normalized P&L is stable across regimes** (15.5–16.0 per
   unit vol for RV), while RV's dollar P&L per hour roughly triples from low
   to high vol ($2.91 → $8.57). So the normalization does its job.

**Caveats:**
- Regime is partly time of day. High-vol hours cluster at 14:00–18:00 UTC
  (US session); low-vol hours at 04:00–09:00 UTC.
- Fills are simulated, not real.
- One asset and 41 days.
- The refit windows were chosen on the two development days, which are part
  of the 41.

## Earlier findings (development days)

These came from the two development days, before the walk-forward existed.
Where the walk-forward contradicts them, that's noted.

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
    Better forecasts make more money. *The walk-forward contradicts this:
    across 32 low-vol days there's no P&L difference. This was one day's
    path.*
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

**Current state (2026-10-12):**
- Found the lookahead bug described at the top of the main result.
- Built `src/event_sim.py`, an event-driven simulator on raw L1 and trade
  events. It's tested against hand-built fill scenarios in
  `src/test_event_sim.py`.
- With it, the AS market maker as currently parameterized loses money under
  both fill bounds, even requoting every 100 ms:

  | Day | Spread captured per fill | Markout 0.1s after fill | Markout 5s after fill |
  |---|---|---|---|
  | calm | $0.96 | −$1.00 | −$2.60 |
  | volatile | $3.41 | −$2.17 | −$3.26 |

  Its quotes sit $1–8 from a $0.10-wide market, so they only fill when
  the price sweeps through them. That's adverse selection, not a timing
  artifact.

That question was resolved by calibrating `k` from data and sweeping
gamma; the robustness checks and risk metrics are done too (see "Main
result"). Remaining:

1. **Writing the paper** (`paper/`).
2. Optional: a **Tardis L2 check** on the free first-of-month days. Our two
   fill bounds already cover every possible queue position, so true
   queue-position fills can only land between them; the check would only
   place reality inside a range that's already 3–4% wide. That's a footnote,
   not a result.
3. Optional: the GARCH/HAR **training-window** sensitivity (3h trailing for
   GARCH, expanding for HAR). This only changes sigma, and the main finding
   is that sigma differences barely move P&L.

The older plan below predates this finding:

1. **Robustness checks** on the judgment-call parameters (the 300s window,
   gamma, the fill-model targets, the 3h training window), and controlling
   for time of day: compare estimators within the same UTC hours.
2. **More days.** At ~20s per day, all 228 weekdays in the window take
   about 30 minutes. That would add power, particularly in the high-vol
   regime where the effect is.
3. **An economic loss for forecasts.** Score forecasts with a loss shaped
   like the market maker's (penalizing over-forecasts), to see whether it
   ranks estimators the way P&L does.
4. Calibrating `k` from `bookDepth` instead of guessing it.
5. Writing the paper (`paper/`).

## Running it

```bash
pip install -r requirements.txt
python src/download_data.py --date 2024-01-10
python src/download_data.py --date 2023-07-19
python src/as_skeleton.py               # add --date YYYY-MM-DD for any other day
python src/compare_estimators.py --seeds 20     # add --date YYYY-MM-DD for any other day (~5 min/day)
python src/signature_plot.py --dates 2024-01-10 2023-07-19

# walk-forward study (~20 min; downloads each day, deletes its raw data after)
python src/select_days.py
python src/walkforward_event.py --workers 3
python src/analyze_walkforward_event.py       # --rule optimistic | --fee-bps 2 | --variant raw
python src/test_event_sim.py                  # fill-rule checks for the event simulator
python src/robustness_event.py --workers 3    # ~35 min: k / decision-interval / latency variants
python src/analyze_robustness.py
# (walkforward.py / analyze_walkforward.py are the superseded 1-second-bar version)
```
