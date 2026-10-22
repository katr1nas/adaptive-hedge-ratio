# Adaptive Hedge Ratios, CPO and Portfolio Allocation

A quantitative research repository exploring adaptive pairs trading, machine-learning-based parameter selection for mean reversion, and portfolio allocation across strategy return streams.

> **Research status:** Experimental backtests, not evidence of a deployable trading edge. Reported results have not been independently reproduced here.

## 1. Dynamic Hedge Ratio with a Kalman Filter

The first experiment asks whether a **time-varying hedge ratio** can improve mean-reversion pairs trading compared with a static ordinary least squares (OLS) estimate.

For two log-price series:

$$\log(A_t)=\alpha_t+\beta_t\log(B_t)+\epsilon_t.$$

A static model estimates the intercept and hedge ratio on a training sample and keeps them fixed. The Kalman model treats both as latent states:

$$\theta_t=\begin{bmatrix}\alpha_t\\\beta_t\end{bmatrix},\qquad \theta_t=\theta_{t-1}+w_t.$$

As observations arrive, the filter updates its state estimate. The innovation provides a candidate trading spread. The **Kalman gain** determines how strongly a new observation changes the estimated state. Process noise determines how quickly the hedge ratio can adapt: too little adaptation risks a stale hedge ratio, while too much may fit short-term noise.

### Backtest design

- Train/test split: 60/40.
- Reported out-of-sample period: 2023–2025.
- Entry: $|z|>2$; exit: $|z|<0.5$; stop: $|z|>4$.
- Transaction-cost parameter: 0.0005.
- Hedge-ratio updates use observations available up to the relevant time.

### Reported results

| Method | Out-of-sample Sharpe |
|---|---:|
| Static OLS beta | -1.45 |
| Frozen-beta control | -0.40 |
| Kalman beta | **+0.55** |

Kalman improved the Sharpe ratio for **6 of 7 pairs**. Nevertheless, profitability was concentrated in a few episodes for some pairs (notably UPS/FDX). The result suggests that adaptive hedge ratios may address a weakness of static pairs trading, but does not establish a durable alpha source. See `research/kalman_filter.md` and `conclusion.md` for further discussion.

## 2. CPO: Conditional Parameter Optimization

The `strategies/cpo_model/` experiment studies whether a model can **select mean-reversion parameters conditional on current market features**, rather than always using a fixed z-score window and entry threshold. Here, CPO describes the project's conditional parameter-selection approach; the acronym is used as a descriptive label, not a claim of a standardized published method.

### Baseline mean-reversion rule

For a rolling window $n$:

$$z_t=\frac{P_t-\text{SMA}_n(P)_t}{\text{Std}_n(P)_t}.$$

For threshold $h$, the implemented signal is:

$$s_t=\begin{cases}
1 & z_t < -h \\\\
-1 & z_t > h \\\\
0 & \text{otherwise}
\end{cases}$$

The strategy return is computed as:

$$r^{\mathrm{strategy}}_t=s_{t-1}\left(\frac{P_t}{P_{t-1}}-1\right).$$

Thus the position signal is shifted by one bar before being multiplied by the asset return. The code uses **ES=F** data downloaded through `yfinance` from 2017-01-01 to 2026-01-01. No transaction costs are deducted in this CPO backtest implementation.

### Candidate parameters and features

The code evaluates a grid of **36 parameter combinations**:

- Rolling windows: `5, 10, 20, 15, 30, 40`.
- Z-score thresholds: `0.5, 1, 1.5, 1.7, 2, 2.5`.

Each candidate/day is described by five features: `window`, `threshold`, `z_score`, five-day rolling return volatility, and five-day price momentum. The target is the candidate strategy return shifted one row forward (`strategy_return.shift(-1)`), intended to represent the following period's outcome.

### Model training and selection

`RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=10, n_jobs=-1)` predicts candidate returns. Unique dates are split chronologically into **60% training, 20% validation, and 20% testing**, so candidate rows for a given date remain in the same split.

For each date, `get_result()` sorts candidates by predicted return and keeps the one with the highest prediction. Its `actual_return` column is then used to construct the selected strategy return series. The validation benchmark is the fixed parameter pair `window=20`, `threshold=1.0`.

**Important validation caveat:** the target is created by shifting returns within each parameter-combination DataFrame before concatenation, which is appropriate for per-candidate next-period labels. However, the model's selected return is assigned to the feature date and later reindexed to the next trading date. The precise execution timeline and feature availability should be audited before interpreting the selected series as executable out-of-sample P&L. In particular, confirm that every feature used to choose the next session's trade was observable before that trade, and that the return corresponds to the intended holding period. The model is fitted once on the training split; the validation set is used for comparison, not shown as a hyperparameter tuning procedure.

## 3. Portfolio Construction

The repository combines the **CPO** and **Kalman** strategy return streams. `portfolio/correlation.py` reads `outputs/cpo_returns.csv` and `outputs/kalman_returns.csv`, aligns their dates using an inner join, drops missing observations, and reports individual strategy metrics and correlation. Three weighting schemes are compared.

### Equal weight

The portfolio simply averages the two daily strategy returns:

$$r_{p,t}=0.5r_{\mathrm{CPO},t}+0.5r_{\mathrm{Kalman},t}.$$

This is the reference allocation and does not adapt to estimated risk or expected return.

### Inverse-volatility weighting

`portfolio/portfolio.py` estimates each strategy's rolling volatility using **63 trading days**, with at least 21 observations, annualized by $\sqrt{252}$. Raw weights are proportional to inverse volatility:

$$\tilde w_{i,t}=\frac{1/\hat\sigma_{i,t}}{\sum_j1/\hat\sigma_{j,t}}.$$

Weights are **shifted one day** before applying them to strategy returns, avoiding same-day volatility information in the allocation. Lower-volatility strategies receive larger weights; expected returns and correlations do not directly enter this rule.

## 4. Kelly Criterion and Fractional Kelly

The Kelly criterion chooses exposure to maximize expected logarithmic wealth growth. For a binary bet with win probability $p$, loss probability $q=1-p$, and net win payoff $b$, the classical fraction is:

$$f^*=\frac{bp-q}{b}.$$

For a vector of strategy returns, a common small-return approximation gives the unconstrained multi-asset Kelly exposure:

$$\mathbf{k}=\Sigma^{-1}\boldsymbol\mu,$$

where $\boldsymbol\mu$ is the vector of estimated expected returns and $\Sigma$ is the covariance matrix. This approximation is sensitive to noisy return estimates and an ill-conditioned covariance matrix.

### How this repository implements Kelly

`portfolio/kelly_sizing.py` uses:

- `LOOKBACK = 252` (changeable to 128 for the alternative experiment).
- `KELLY_FRACTION = 0.25`.
- Minimum/maximum final weight bounds of `0.10` and `0.90`.
- Equal-weight anchor of `[0.5, 0.5]`.

At each eligible date, it takes the **preceding** `LOOKBACK` observations (excluding the current date), estimates historical means and covariance, and adds a small diagonal regularizer:

$\Sigma_{\mathrm{reg}}\mathbf{k}=\hat{\boldsymbol{\mu}}$

It solves $\Sigma_{\mathrm{reg}}\mathbf{k}=\hat{\boldsymbol{\mu}}$, clips negative components to zero (long-only), and normalizes the nonnegative vector to sum to one. If all components are zero, it uses equal weights. It then blends this Kelly-derived **tilt** with equal weights:

$$\mathbf{w}_t=0.75\begin{bmatrix}0.5\\\\0.5\end{bmatrix}+0.25\frac{\max(\mathbf{k},0)}{\sum_j\max(k_j,0)}.$$

Finally, weights are clipped to the configured bounds and renormalized. Portfolio returns are the weighted sum of same-date strategy returns, using weights estimated exclusively from preceding dates.

**Terminology note:** this is a **25% blend toward normalized, long-only Kelly weights**, not classical fractional Kelly leverage sizing ($0.25\Sigma^{-1}\mu$). Normalizing the Kelly vector discards its absolute leverage recommendation. With two strategies and a 25% tilt, the actual weights generally remain between 37.5% and 62.5%, making the configured 10%–90% bounds nonbinding under normal finite estimates.

## 5. Repository Structure

```text
adaptive-hedge-ratio/
├── README.md
├── conclusion.md                 # Kalman experiment interpretation
├── requirements.txt
├── strategy.ipynb                # Research notebook
├── research/
│   ├── kalman_filter.md          # Kalman research notes
│   └── *.pdf                     # Supporting research material
├── strategies/
│   ├── kalman_hedge_ratio/
│   │   ├── kalman_hedge.py       # Adaptive pairs-trading implementation
│   │   └── loader.py
│   ├── cpo_model/
│   │   ├── cpo_model.py          # Candidate grid, RF fit, evaluation, export
│   │   ├── loader.py             # Yahoo Finance data loading
│   │   ├── metrics.py            # Z-score and Sharpe functions
│   │   ├── train_test_split.py   # Chronological 60/20/20 split
│   │   └── get_result.py         # Best predicted candidate per date
│   └── vwap/                     # Separate VWAP research work in progress
└── portfolio/
    ├── correlation.py            # Load, align and compare strategy returns
    ├── metrics.py                # Annual return, volatility, Sharpe, drawdown
    ├── portfolio.py              # Inverse-volatility allocation
    ├── kelly_sizing.py           # Kelly-tilted allocation
    └── unite_returns.py          # Load portfolio returns for comparison
```

The scripts use local imports and expect strategy CSVs under `outputs/` (a generated directory, not present in the supplied archive). The CPO script creates `outputs/cpo_returns.csv`; the Kalman output must likewise exist before portfolio scripts are run. Run modules from their respective directories or adjust imports/package structure as needed. Some portfolio scripts execute calculations on import.

## 6. Portfolio Backtest Comparison

The reported runs use two Kelly lookback configurations and **different evaluation samples**. The 252-observation lookback run is reported for **2024-09-17 through 2025-12-31**, with 324 evaluated observations; the 128-lookback run has 198 observations. These are user-reported outputs rather than results recomputed from the ZIP.

| Allocation | Annualized return (128; n=198) | Max drawdown (128; n=198) | Annualized return (252; n=324) | Max drawdown (252; n=324) |
|---|---:|---:|---:|---:|
| Equal weight | 5.55% | -6.81% | 3.65% | -11.16% |
| Inverse volatility | **9.62%** | **-4.84%** | 3.56% | -10.98% |
| Fractional Kelly tilt | 5.04% | -6.64% | **4.43%** | **-10.36%** |

**Interpretation.** On the 198-observation sample, inverse-volatility weighting had the highest annualized return and smallest drawdown. On the 324-observation sample, the Kelly-tilted allocation led both metrics, but by a modest margin. No allocation method dominates both reported samples.

**Critical comparison limitation:** the number of evaluated dates changes between runs. The shorter Kelly lookback should ordinarily permit more early dates to be evaluated; the inverse relationship between lookback and reported sample size here suggests that other data-selection or comparison settings may also differ. Therefore, these results **cannot establish that lookback 128 or 252 is superior**. Moreover, `LOOKBACK` in the code changes the Kelly estimator only; inverse-volatility uses its own fixed 63-day window. For a controlled experiment, compare both Kelly settings over identical calendar dates and apply identical data availability, costs and metric calculations to every portfolio.

## 7. Limitations and Next Steps

1. **Execution timing:** audit CPO feature timestamps, next-period labels, and reindexing of realized returns.
2. **Trading costs:** include realistic fees, slippage, and turnover, particularly for frequently changing parameters or weights.
3. **Common test window:** evaluate both Kelly lookbacks and all allocation schemes over the same observations.
4. **Robustness:** report full Sharpe ratios, turnover, correlations, exposure and confidence intervals; test alternative regimes and parameter grids.
5. **Model risk:** examine sensitivity of Kelly allocations to estimated means and covariance, and compare with simpler shrinkage or volatility-based rules.
6. **Reproducibility:** pin dependencies, provide a deterministic run script, document the source of Kalman CSVs, and save experiment settings alongside output files.

## Conclusion

The Kalman study suggests that an adaptive hedge ratio can outperform a static OLS relationship on the tested pairs, although the evidence remains limited. CPO explores whether market-dependent parameter selection can improve a simple mean-reversion rule. Portfolio construction then tests whether combining these distinct return streams with equal, inverse-volatility, or Kelly-tilted weights changes the risk/return profile.

The current portfolio results favor inverse-volatility weighting in one sample and Kelly-tilted weighting in another. Because the samples differ, the next priority is **a reproducible, same-period out-of-sample comparison**, not additional backtest optimization.
