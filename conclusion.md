# Conclusion — Static β vs Kalman β

The first experiment compared a conventional static OLS hedge ratio with an adaptive Kalman-filter hedge ratio across seven pairs using out-of-sample data from 2023–2025.

The results were:

| Model            | Out-of-sample Sharpe |
| ---------------- | -------------------: |
| Static β         |                -1.45 |
| Frozen β control |                -0.40 |
| Kalman β         |            **+0.55** |

The Kalman approach improved performance on six of the seven tested pairs.

This is a substantial improvement over the static hedge-ratio model and suggests that one source of failure in conventional pairs trading is the assumption that the hedge ratio remains constant.

However, the experiment does not establish a robust trading edge.

The UPS/FDX pair produced a Sharpe ratio of approximately 0.33, and much of its profitability was concentrated in only three episodes. This raises the possibility that part of the aggregate result is driven by specific market regimes rather than a universally persistent effect.

The main conclusion is therefore:

> **Adaptive β appears capable of fixing an important weakness of static pairs trading, but the existence of a durable edge remains unproven.**

The most important conceptual lessons from the experiment were:

1. Cointegration is not the same as correlation.
2. Out-of-sample testing is essential.
3. The Kalman gain controls how strongly the model reacts to new information.
4. A hedge ratio should adapt to structural changes without simply following every short-term movement in the spread.
5. Aggregate Sharpe can hide concentration in a small number of trades or market regimes.

The next stage is validation rather than adding model complexity.

The methodology will be tested on EURUSD/GBPUSD, followed by sensitivity analysis of the Kalman process-noise parameter and analysis of individual trade episodes.

The objective is not to maximize the backtest.

The objective is to determine whether adaptive hedge ratios capture a repeatable and economically meaningful property of the relationship between financial assets.
