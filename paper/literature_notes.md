# Literature notes

What each reference is used for in the write-up. BibTeX keys refer to
`references.bib`.

**Verification status**
- **checked**: bibliographic details confirmed against the publisher or
  repository page on 2026-10-10.
- **from memory**: details are from recall and look right, but check
  volume/pages/DOI before submitting.
- **unverified claim**: a specific claim that came from a search summary,
  not the primary source. Don't cite the claim until you've read it there.

## 1. The market making model

| Key | Status | Use in the paper |
|---|---|---|
| `avellaneda2008` | from memory | The model. Reservation price `r = s - q*gamma*sigma^2*(T-t)`, spread formula, and the exponential fill intensity `lambda(delta) = A*exp(-k*delta)`. Sigma is an input the model does not estimate, which is the gap this project studies. |
| `gueant2013` | from memory | Closed-form asymptotic quotes and the standard reference for calibrating `A` and `k`. Relevant when `k` is calibrated from `bookDepth`. |
| `gueant2017` | from memory | General framework reconciling AS-style models; closed-form approximations. |
| `stochvol_lob2016` | authors missing | AS extension with stochastic volatility. Motivates why a time-varying sigma estimate matters. |

**Gap to state in the paper:** no source found evaluates volatility
estimators *inside* a market maker. The literature scores them on forecast
accuracy (MSE/QLIKE against realized variance). This project scores them on
market-making outcomes, which is its contribution.

## 2. The estimators

| Key | Status | Use in the paper |
|---|---|---|
| `andersen1998` | from memory | Justifies realized variance as the volatility proxy for evaluating forecasts. |
| `lps2015` | checked (title, journal, year); pages from memory | **Main justification for RV.** 400+ estimators on 31 assets: little significantly beats 5-minute RV. Note their benchmark-free tests do find some more sophisticated measures win, and a low-frequency truncated RV does best for forecasting. |
| `bollerslev1986` | from memory | GARCH(1,1). |
| `hansenlunde2005` | checked | **Main justification for GARCH(1,1).** 330 ARCH-type models: nothing beats GARCH(1,1) on FX; leverage models beat it on IBM. A later replication finds the answer depends on which realized-variance proxy is used. |
| `riskmetrics1996` | from memory | EWMA, `sigma^2_t = lambda*sigma^2_{t-1} + (1-lambda)*r^2_{t-1}`. The industry default. Our fitted GARCH has alpha+beta ~ 1 (integrated GARCH), and EWMA is a special case of that, so EWMA tests whether fitting GARCH's parameters adds anything. |
| `corsi2009` | from memory | **HAR-RV**, the standard benchmark in realized-volatility forecasting. Originally daily/weekly/monthly components; our intraday horizons are an adaptation, to be stated as such. |
| `bns2004` | from memory | Bipower variation, robust to jumps. In the literature it is mostly used as a HAR component (HAR-J, HAR-CJ) rather than on its own. |
| `parkinson1980`, `garmanklass1980` | from memory | Range-based estimators, mentioned only to justify excluding them: they're built for daily OHLC bars, and with tick data RV dominates. |

## 3. Microstructure noise and sampling frequency

Relevant because our RV uses 1-second returns, which is in the range where
noise biases RV upward.

| Key | Status | Use in the paper |
|---|---|---|
| `bandirussell2006` | from memory | Separating noise from volatility; noise biases very-high-frequency RV. |
| `bandirussell2008` | from memory | MSE-optimal sampling frequency: trades noise-induced bias against variance reduction. Justifies running a signature plot before picking a sampling frequency. |
| `hansenlunde2006` | from memory | Realized variance and microstructure noise. **Unverified claim:** a search summary said they report ~3-minute average optimal sampling for IBM. Don't cite that number until confirmed in the paper. |
| `griffinoomen2008` | checked | Tick time beats transaction time sampling in MSE; noise is ~iid in transaction time but dependent in tick time. (An earlier draft of these notes wrongly attributed this to Hansen & Lunde.) |
| `zhang2005` | from memory | Two-scales realized variance (noise-robust). |
| `bnhls2008` | from memory | Realized kernels (noise-robust). |
| `jacod2009` | from memory | Pre-averaging (noise-robust). |

**What our signature plot found (2026-10-10), to discuss against these
papers:** these noise papers mostly study transaction prices, where bid-ask
bounce inflates RV at short intervals. On Binance BTCUSDT perp *mid-quotes*
the opposite happens on a calm day: 79% of 1-second returns are zero and
lag-1 autocorrelation is +0.13, so 1-second RV sits ~17% below its peak at
15-20s sampling. On a volatile day it's ~6% below its peak at 5s, and 10s
returns turn negatively autocorrelated (-0.11). So the bias depends on the
regime, and estimators designed for bounce noise don't target it.

Noise-robust estimators were only going to be needed if the signature plot
showed large upward noise bias at the sampling frequency we use; it didn't. A streaming-estimation comparison (arXiv:2003.13062) found
multi-scale best below optimal bandwidth, flat-top realized kernel best above
it, and pre-averaging in between.

## 4. Evaluation

| Key | Status | Use in the paper |
|---|---|---|
| `patton2011` | from memory | QLIKE and MSE are robust loss functions when the volatility proxy is noisy. Use QLIKE to score forecast accuracy alongside P&L, which makes results comparable with the literature. |

## 5. Crypto-specific evidence

| Key | Status | Use in the paper |
|---|---|---|
| `aganin2023` | checked | Bitcoin and E-mini S&P 500, 5-minute RV, one-day-ahead: HAR beats GARCH for both, by more for Bitcoin. Rolling 399-day windows, 2018–2021, Model Confidence Set at 1%. |
| `bergsli2022` | checked | Bitcoin, several GARCH models vs two HAR models, RV as proxy. EGARCH and APARCH best among GARCH; HAR's edge strongest at short horizons. |
| `btcjumps2017` | authors missing | High-frequency jump analysis of Bitcoin, reports noise variance at 5/10-minute sampling. Supports using jump-robust estimators on BTC. |

A University of Reading working paper (HAR variants with jumps and
structural breaks on Bitcoin, 2012–2018; finds HARQ-F-J best out of sample)
was also found but its citation details weren't confirmed:
<https://reading-clone.eprints-hosting.org/87932/1/R%26R_Forecasting%20the%20Volatility%20of%20Bitcoin_complete.pdf>

## 6. Practitioner implementations

| Key | Status | Use in the paper |
|---|---|---|
| `hummingbot_instantvol` | checked (source code read) | The best-known open-source AS implementation estimates sigma as `sqrt(sum(diff(prices)^2) / n)` over a tick buffer: a plain rolling realized volatility. Justifies RV as the practitioner baseline. |
