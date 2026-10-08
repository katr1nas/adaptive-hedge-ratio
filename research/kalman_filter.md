# Kalman Filter — Research Notes

## 1. Motivation

A conventional pairs-trading strategy estimates:

$$
A_t = \alpha + \beta B_t + \epsilon_t
$$

using historical data.

The problem is that β is assumed to remain constant.

Financial relationships can change over time, so a hedge ratio estimated several years ago may no longer represent the current relationship.

The Kalman filter provides a way to estimate a time-varying β.

---

## 2. State-Space Model

The state consists of the intercept and hedge ratio:

$$
\theta_t =
\begin{bmatrix}
\alpha_t \\
\beta_t
\end{bmatrix}
$$

The state evolves according to:

$$
\theta_t = \theta_{t-1} + w_t
$$

where \(w_t\) represents process noise.

The observation equation is:

$$
y_t =
\begin{bmatrix}
1 & x_t
\end{bmatrix}
\theta_t + \epsilon_t
$$

where:

* \(y_t\) = log price of asset A
* \(x_t\) = log price of asset B
* \(\alpha_t\) = dynamic intercept
* \(\beta_t\) = dynamic hedge ratio

---

## 3. Prior vs Posterior

Before observing the current observation, the model has a prior state:

$$
\theta_{t|t-1}
$$

After observing \(y_t\), the state is updated:

$$
\theta_{t|t}
$$

For trading, the prior hedge ratio is important because the signal should be generated using information available before the current observation is incorporated into the state.

The implementation therefore records the prior β before updating the state.

---

## 4. Kalman Gain

The Kalman gain determines how much the new observation changes the state.

Conceptually:

$$
\text{new state}
=
\text{old state}
+
K_t \times \text{innovation}
$$

A large Kalman gain means the filter reacts strongly to new information.

A small gain means the filter changes slowly.

This creates a useful interpretation for pairs trading:

> The Kalman filter is not simply trying to predict the spread. It is estimating how much the relationship between the two assets should be revised after each observation.

---

## 5. β Should Move More Slowly Than the Spread

This became one of the main conceptual lessons of the experiment.

The spread can move sharply because of short-term market noise.

That does not necessarily mean the underlying relationship between the two assets has changed by the same amount.

If β is allowed to react almost one-for-one with every movement in the spread, the hedge ratio can begin fitting noise.

Therefore the model needs controlled state dynamics.

The process noise determines how much β is allowed to change.

This creates a bias-variance style trade-off:

**Too rigid**

* β becomes stale.
* Structural changes are missed.
* The static model remains exposed to changing relationships.

**Too adaptive**

* β becomes noisy.
* Short-term fluctuations are interpreted as structural changes.
* The model may overfit the spread.

The useful region lies between these extremes.

---

## 6. Cointegration vs Correlation

Two assets can have high correlation without being cointegrated.

Correlation describes co-movement.

Cointegration describes whether a linear combination of the series is stationary.

For a pairs-trading strategy, the second property is more directly related to the existence of a mean-reverting spread.

This distinction is important because visually similar price charts do not automatically imply a tradable statistical relationship.

---

## 7. First Experiment

Seven pairs were tested using an out-of-sample period covering 2023–2025.

Results:

* Static β Sharpe: **−1.45**
* Kalman β Sharpe: **+0.55**
* Frozen β control: **−0.40**
* Kalman improved 6/7 pairs.

The result suggests that allowing β to adapt can address an important weakness of static pairs trading.

However, one pair (UPS/FDX) produced a Sharpe of only approximately 0.33, with profitability concentrated in three major episodes.

Therefore the result is promising but not conclusive.

---

## 8. Research Principle

A backtest producing a good Sharpe ratio is not the final answer.

The more important questions are:

* Does the result survive different parameters?
* Does it survive different instruments?
* Does it survive different market regimes?
* Is the return distributed across many independent trades?
* Does the adaptive β actually explain the improvement?
* Or is the result caused by a small number of unusually profitable episodes?

The next stage of the project should answer these questions.
