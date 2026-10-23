# Literature notes

What each reference is used for in the write-up. BibTeX keys refer to
`references.bib`.

**Verification status**
- **bib checked:** authors, title, journal, volume, issue, pages and DOI
  confirmed on 2026-10-22 against Crossref (journal articles), arXiv, or
  publisher and library records (books, the Risk article).
- **content checked:** the claim we cite was confirmed in the paper itself
  (full text or abstract, as noted).
- **content from memory:** the bibliographic entry is checked, but the claim
  we attribute to the paper hasn't been confirmed in its text yet. Read the
  relevant section before citing the claim.

All claims were checked on 2026-10-22. The source used is named in each
status: full text, the abstract, a table of contents or index, or secondary
sources (other papers describing it). Secondary-only claims are the weakest;
read the original before leaning on them.

## 1. The market making model

| Key | Status | Use in the paper |
|---|---|---|
| `avellaneda2008` | bib checked; content checked (full text: eqs. 1, 8, 12, 13, 30) | The model: arithmetic Brownian mid (eq. 1), exponential utility (eq. 13), reservation price `r = s - q*gamma*sigma^2*(T-t)`, the spread formula, and the exponential fill intensity `lambda(delta) = A*exp(-k*delta)` (eqs. 8, 30, 12). The spread formula (30) comes from their approximate solution; say so. Sigma is an input the model doesn't estimate, which is the gap this project studies. Section 6 of the README uses the spread formula to split the quoted spread into sigma's part and k's part. |
| `gueant2013` | bib checked; content checked (abstract) | Solves the AS problem under inventory constraints, by turning the HJB equations into linear ODEs, with closed-form approximations of the quotes. Cite as the main extension of AS, not as something we implement. |
| `gueant2017` | bib checked; content checked (abstract) | A framework that generalizes and reconciles AS-style models, with closed-form approximations of optimal quotes, single- and multi-asset. |
| `cartea2015` | bib checked; content checked (index: market making with adverse selection, p. 261) | Textbook reference for market making when the market maker is adversely selected. Cite for adverse selection as the market maker's key cost, which plain AS doesn't model. "Short-term signals" isn't confirmed; don't attribute it without checking the chapter. |
| `cartea2017` | bib checked; content checked (abstract) | Market making when the trader doubts the model: ambiguity about market-order arrival rates, limit-order fill probabilities, and midprice dynamics. Robust quotes reduce inventory risk and adverse selection costs, and raise the Sharpe ratio. Supports asking how sensitive AS is to its inputs; doesn't study sigma estimators. |
| `yang2020stochvol` | bib checked; content checked (abstract) | AS-type quoting with Heston stochastic volatility, solved as a stochastic control problem. Theory showing a time-varying sigma matters for optimal quotes; no empirical estimator comparison. |
| `stoikov2018` | bib checked; content checked (secondary sources) | The micro-price: the mid adjusted for the expected future move, from the queue imbalance and the spread via a Markov chain; a martingale, unlike the mid. Cite in future work as an adverse-selection defense. |
| `falcesmarin2022` | bib checked; content checked (full text and code read 2026-10-12) | Closest prior work: AS on BTC-USD with reinforcement learning tuning gamma and a skew. Details below. |

**Falces Marín et al. (2022), what they do:**
- **Data:** L2 order book updates plus trades, BTC-USD, 7 Dec 2020 to 8 Jan
  2021, 12 hours a day. 30 test days. The exchange isn't named.
- **Sigma:** variance of the mid over the last w ticks, with w=25 picked by
  a genetic search to maximize Sharpe. That's a plain rolling-variance
  estimator, with no comparison against alternatives. **Supports our gap:**
  they treat sigma as a tuning parameter, not an estimation problem.
- **k:** from order-book update arrival rates (Aldridge), so no A parameter.
- **Fills**, from their code (github.com/javifalces/HFTFramework,
  `OrderMatchEngine.java`), not the paper, which doesn't describe them:
  - The full book is rebuilt from each L2 snapshot, and our order joins the
    back of the queue at its price.
  - Each real trade is replayed against the book: displayed depth at better
    prices and ahead of us at our price is consumed first; only what's left
    fills us, with partial fills.
  - Fills also happen when a new snapshot crosses a resting order.
  - Order latency is configurable (fixed or Poisson). Fees are an optional
    switch, off in their documented examples.
  - Caveat: the engine has commits up to April 2026, so this may not be
    exactly the version that produced the 2022 results.
- **Needs L2 depth we don't have** (Binance's free archive only has L1 for
  futures): queue position comes from displayed depth at and in front of the
  order's price.
- **Every strategy they test loses money**, from their own Tables 2 and 5:
  - **AS with genetically tuned parameters:** median daily Sharpe −0.26,
    positive on 2 of 30 days. P&L-to-MAP is negative on 28 of 30 days.
  - **Their best RL variant:** median Sharpe −0.09.
  - **The two constant-spread baselines, which quote at the best price:**
    median Sharpe −0.45 and −0.39.

  They only compare models relative to each other. This supports our
  finding: with realistic fills, AS loses to adverse selection on BTC even
  with full L2 and queue modeling.

**Gap to state in the paper:** no source we found evaluates volatility
estimators *inside* a market maker. The forecasting literature scores them on
forecast accuracy (MSE or QLIKE against realized variance); the AS literature
treats sigma as given (`avellaneda2008`, `gueant2013`), models it
theoretically (`yang2020stochvol`) or tunes it (`falcesmarin2022`). This
project scores standard estimators on market-making outcomes. State it as
"to our knowledge".

## 2. Market microstructure: adverse selection and spreads

| Key | Status | Use in the paper |
|---|---|---|
| `glostenmilgrom1985` | bib checked; content checked (abstract) | The adverse-selection theory of the spread: informed traders cause a positive spread even for a risk-neutral, zero-profit specialist, who recoups losses to the informed from liquidity traders. Motivates the decomposition. |
| `huangstoll1996` | bib checked; content checked (abstract) | Measures execution costs with quoted, effective and realized spreads; the realized spread is the revenue left to the liquidity supplier after the price moves. Our decomposition follows the same logic: spread captured (effective) = realized spread + adverse selection. Their 1997 *Review of Financial Studies* paper is the formal three-way spread decomposition; cite 1996 for the realized-spread measure. |

## 3. The estimators

| Key | Status | Use in the paper |
|---|---|---|
| `andersen1998` | bib checked; content checked (abstract) | Justifies realized variance as the volatility proxy for evaluating forecasts. |
| `lps2015` | bib checked; content checked (earlier) | **Main justification for RV.** 400+ estimators on 31 assets: little significantly beats 5-minute RV. Their benchmark-free tests do find some more sophisticated measures win. |
| `bollerslev1986` | bib checked | GARCH(1,1). |
| `hansenlunde2005` | bib checked; content checked (earlier) | **Main justification for GARCH(1,1).** 330 ARCH-type models: nothing beats GARCH(1,1) on FX; leverage models beat it on IBM. |
| `riskmetrics1996` | content checked (secondary sources only: lambda = 0.94 for daily data) | EWMA, `sigma^2_t = lambda*sigma^2_{t-1} + (1-lambda)*r^2_{t-1}`, the industry default; lambda = 0.94 for daily data. Our fitted GARCH has alpha+beta ≈ 1 (integrated GARCH), and EWMA is a special case of that. |
| `corsi2009` | bib checked; content checked (abstract and secondary sources) | **HAR-RV**, the standard benchmark in realized-volatility forecasting. Originally daily, weekly and monthly components; our 1/5/30-minute components are an intraday adaptation, to be stated as such. |
| `bns2004` | bib checked; content checked (abstract) | Bipower variation, fairly robust to rare jumps; RV minus bipower variation isolates the jump part. Mostly used as a HAR component (HAR-J, HAR-CJ) rather than on its own. Explains why it was dropped. |
| `parkinson1980`, `garmanklass1980` | bib checked | Range-based estimators, mentioned only to justify excluding them: they're built for daily OHLC bars, and with tick data RV dominates. |
| `hummingbot_instantvol` | content checked (source code read) | The best-known open-source AS implementation estimates sigma as `sqrt(sum(diff(prices)^2) / n)` over a tick buffer: a plain rolling realized volatility. Justifies RV as the practitioner baseline. |

## 4. Forecast evaluation

| Key | Status | Use in the paper |
|---|---|---|
| `patton2011` | bib checked; content checked (secondary sources) | MSE and QLIKE are the common loss functions that rank variance forecasts consistently even when the proxy is noisy. QLIKE is less sensitive than MSE to large right-tail observations, so a few extreme periods don't dominate it; it penalizes under-forecasts more than over-forecasts. Patton & Sheppard (2009), a handbook chapter, recommend QLIKE over MSE; add that reference if cited. |
| `hln2011` | bib checked; content checked (abstract) | A set built to contain the best model with a given confidence, like a confidence interval for models; uninformative data give a large set. Model Confidence Set, used via `arch.bootstrap.MCS` at the 10% level to find the estimators statistically tied for best within each regime. |
| `fleming2001` | bib checked; content checked (abstract) | Judging volatility models by economic value rather than statistical fit. For short-horizon investors, volatility timing beats static portfolios, robust to estimation risk and costs. A useful contrast: there, better volatility information has economic value; for an AS market maker, standard estimators don't. |

**Result to discuss against `patton2011`:** HAR has the lowest QLIKE in every
regime and RV is excluded from the Model Confidence Set in all three, yet no
estimator changes AS P&L significantly (0 of 36 comparisons). Statistical
forecast accuracy doesn't translate into market-making value here. The
decomposition and the sigma-in-the-quotes analysis explain why.

## 5. Statistical inference

| Key | Status | Use in the paper |
|---|---|---|
| `lakens2017` | bib checked; content checked (abstract) | Practical guide to equivalence testing with TOST: reject effects large enough to matter, with bounds set from the smallest effect size of interest. **It recommends prespecifying the bounds.** Ours were not prespecified (the analysis came after the results), so the paper must say so and report the smallest passing bound against an economic benchmark (the Oracle's gain). |
| `schuirmann1987` | bib checked; content checked (abstract) | The original TOST procedure, from bioequivalence testing; it reaches the same conclusion as checking whether the 1 − 2α confidence interval lies inside the bounds, which is how we compute it. |
| `hoenigheisey2001` | bib checked; content checked (full text) | "For any test the observed power is a 1:1 function of the p value", so it adds no information. Justifies reporting the minimum detectable effect instead. |
| `berger1982` | bib checked; content checked (secondary sources only) | Intersection-union tests: when a conclusion requires every one of several tests to reject, each can be run at the full level, with no multiplicity correction. Justifies TOST without Bonferroni for the joint equivalence claim. |

## 6. Backtesting

| Key | Status | Use in the paper |
|---|---|---|
| `lopezdeprado2018` | bib checked; content checked (table of contents) | Chapter 11 on the dangers of backtesting and backtest overfitting; chapter 12 on the walk-forward method and its pitfalls. Supports the hold-out check. "Lookahead bias" by name isn't confirmed; cite the chapters for overfitting and walk-forward. |

## 7. Microstructure noise and sampling frequency

Relevant because our RV uses 1-second returns, which is in the range where
noise biases RV.

| Key | Status | Use in the paper |
|---|---|---|
| `andersen2000` | bib checked (Kellogg listing; volume shown as 3 there, commonly cited as 13); content checked (secondary sources: the plot comes from their 1999 manuscript, published as this Risk article, March 2000) | Introduces the volatility signature plot: RV against sampling interval. We use it to choose 60 s returns for the regime label. |
| `bandirussell2006` | bib checked; content checked (abstract) | Observed high-frequency returns mix efficient-price variance and noise variance; they separate the two using returns sampled at different frequencies. |
| `bandirussell2008` | bib checked; content checked (abstract) | MSE-optimal sampling frequency: trades noise-induced bias against variance reduction. |
| `hansenlunde2006` | bib checked; content checked (abstract) | On the 30 Dow stocks, noise is time-dependent and correlated with efficient-price increments, in both transaction and **quotation** data. Supports our signature-plot finding that mid-quote noise isn't the textbook iid bid-ask bounce. (An earlier "3-minute optimal sampling for IBM" claim was never confirmed and is dropped.) |
| `griffinoomen2008` | bib checked; content checked (earlier) | Tick time beats transaction time sampling in MSE; noise is ~iid in transaction time but dependent in tick time. |
| `zhang2005` | bib checked | Two-scales realized variance (noise-robust). |
| `bnhls2008` | bib checked | Realized kernels (noise-robust). |
| `jacod2009` | bib checked | Pre-averaging (noise-robust). |

**What our signature plot found, to discuss against these papers:** these
noise papers mostly study transaction prices, where bid-ask bounce inflates
RV at short intervals. On Binance BTCUSDT perp *mid-quotes* the opposite
happens on a calm day: 79% of 1-second returns are zero and lag-1
autocorrelation is +0.13, so 1-second RV sits ~17% below its peak at 15-20 s
sampling. On a volatile day it's ~6% below its peak at 5 s. So the bias
depends on the regime, and estimators designed for bounce noise don't target
it. Noise-robust estimators would only be needed if the signature plot
showed large upward bias at our sampling frequency; it didn't.

## 8. Crypto-specific evidence

| Key | Status | Use in the paper |
|---|---|---|
| `aganin2023` | content checked (earlier); bib from the journal page (not in Crossref) | Bitcoin and E-mini S&P 500, 5-minute RV, one-day-ahead: HAR beats GARCH for both, by more for Bitcoin. Rolling 399-day windows, 2018–2021, Model Confidence Set at 1%. |
| `bergsli2022` | bib checked; content checked (earlier) | Bitcoin, several GARCH models vs two HAR models, RV as proxy. EGARCH and APARCH best among GARCH; HAR's edge strongest at short horizons. Together with `aganin2023`, supports including HAR. |
| `scaillet2020` | bib checked; content checked (abstract) | High-frequency jumps in Bitcoin (Mt. Gox, 2011–2013): frequent and clustered, predicted by order-flow imbalance and widening spreads. Background for why BTC volatility is spiky. |

A University of Reading working paper (HAR variants with jumps and
structural breaks on Bitcoin, 2012–2018; finds HARQ-F-J best out of sample)
was also found, but its citation details weren't confirmed:
<https://reading-clone.eprints-hosting.org/87932/1/R%26R_Forecasting%20the%20Volatility%20of%20Bitcoin_complete.pdf>

## Before submitting

Every entry's bibliographic details and cited claims are checked. Three rest
on secondary sources only; read the originals before relying on them:
`berger1982` (the intersection-union principle), `riskmetrics1996` (lambda =
0.94) and `stoikov2018` (the micro-price). `huangstoll1996`'s realized-spread
use is confirmed from the abstract; their exact formulas weren't checked.
