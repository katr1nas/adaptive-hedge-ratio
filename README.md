# Dynamic Hedge Ratio with a Kalman Filter

A pairs-trading research project investigating whether a **time-varying hedge ratio** can improve mean-reversion trading compared with a conventional static OLS hedge ratio.

The core hypothesis is simple:

> If the relationship between two cointegrated assets changes over time, a hedge ratio estimated once on the training period may become stale. A Kalman filter can adapt the hedge ratio gradually as new observations arrive.

## Research Setup

The strategy models the relationship between two log-price series:

$$
\log(A_t) = \alpha_t + \beta_t \log(B_t) + \epsilon_t
$$

Two approaches are compared:

### Static Hedge Ratio

The hedge ratio is estimated using OLS on the training period and then kept fixed throughout the backtest.

### Kalman Hedge Ratio

The intercept and hedge ratio are treated as latent states:

$$
\theta_t =
\begin{bmatrix}
\alpha_t \\
\beta_t
\end{bmatrix}
$$

The state evolves gradually over time:

$$
\theta_t = \theta_{t-1} + w_t
$$

The Kalman filter updates the estimate as new market observations arrive.

The resulting innovation is used as the trading spread.

## Backtest Design

* Training/testing split: 60/40
* Out-of-sample period: 2023–2025
* Entry threshold: \(|z| > 2\)
* Exit threshold: \(|z| < 0.5\)
* Stop threshold: \(|z| > 4\)
* Transaction cost: 0.0005
* Hedge ratio estimated only from information available up to each observation

The strategy is evaluated using out-of-sample Sharpe ratio, return and trade count.

## Results

Across 7 tested pairs:

| Method           | Out-of-sample Sharpe |
| ---------------- | -------------------: |
| Static β         |                -1.45 |
| Frozen β control |                -0.40 |
| Kalman β         |            **+0.55** |

The Kalman approach improved the Sharpe ratio on **6 of 7 pairs**.

The result is encouraging, but it should not yet be interpreted as evidence of a robust trading edge.

The weakest example was UPS/FDX, where the Kalman strategy achieved a Sharpe of approximately **0.33** and a significant portion of profitability came from only three episodes.

This raises the possibility that some of the aggregate improvement is regime-dependent.

## What I Learned

### Cointegration is not correlation

Correlation measures whether two assets tend to move together.

Cointegration asks whether a linear combination of their non-stationary price series is stationary.

For pairs trading, the second property is much more relevant because it provides a basis for modelling a mean-reverting spread.

### Train/Test Separation

Parameters and statistical relationships must be estimated without using future test observations.

The purpose of the out-of-sample period is to test whether the relationship discovered during research survives unseen data.

### Kalman Gain

The Kalman gain determines how strongly the new observation changes the current state estimate.

A high gain means the filter reacts strongly to new information.

A low gain means the filter trusts its existing estimate more.

### Why β Should Not Move Like the Spread

The spread can move substantially from one observation to another without the underlying economic relationship between the assets changing equally quickly.

If β reacts too quickly, the model can begin fitting short-term noise instead of tracking genuine structural changes.

The process-noise parameter therefore controls an important trade-off:

* too little adaptation → stale hedge ratio
* too much adaptation → noisy hedge ratio

## Current Conclusion

The experiment provides evidence that allowing the hedge ratio to evolve over time can substantially improve a static pairs-trading model.

However, the evidence is not sufficient to claim that the Kalman filter creates a robust alpha source.

The next step is therefore **validation rather than additional complexity**.

Planned tests:

1. Test EURUSD/GBPUSD.
2. Test sensitivity to the Kalman process-noise parameter.
3. Analyse individual trade episodes.
4. Check whether performance survives across different market regimes.
5. Compare the adaptive model against stronger controls.

The objective is not to make the backtest look better.

The objective is to determine whether adaptive hedge ratios capture a repeatable property of financial markets.
