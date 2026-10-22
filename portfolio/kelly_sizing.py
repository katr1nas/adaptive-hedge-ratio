import numpy as np
import pandas as pd
from pathlib import Path
from correlation import load_returns

LOOKBACK = 252
KELLY_FRACTION = 0.25
MIN_WEUGHT = 0.10
MAX_WEIGHT = 0.90

equal_weights = np.array([0.5, 0.5], dtype=float)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

cpo_returns = load_returns('cpo_returns.csv', 'cpo')
kalman_returns = load_returns('kalman_returns.csv', 'kalman')

strategy_returns = pd.concat(
    [cpo_returns, kalman_returns],
    axis=1,
    join='inner'
).dropna()

kelly_weights = pd.DataFrame(
    index=strategy_returns.index,
    columns=strategy_returns.columns,
    dtype=float,
)

for i in range(LOOKBACK, len(strategy_returns)):
    history = strategy_returns.iloc[i - LOOKBACK:i]

    expected_returns = history.mean().to_numpy()
    covariance = history.cov().to_numpy()

    covariance = covariance + np.eye(len(covariance)) * 1e-8
    raw_kelly = np.linalg.solve(covariance, expected_returns)

    raw_kelly = np.clip(raw_kelly, 0, None)

    if raw_kelly.sum() == 0:
        kelly_tilt = equal_weights
    else:
        kelly_tilt = raw_kelly / raw_kelly.sum()
    
    weights = (
        (1 - KELLY_FRACTION) * equal_weights
        + KELLY_FRACTION * kelly_tilt
     )
    
    weights = np.clip(weights, MIN_WEUGHT, MAX_WEIGHT)
    weights = weights / weights.sum()

    kelly_weights.iloc[i] = weights

kelly_returns = (
    (kelly_weights * strategy_returns)
    .sum(axis=1, min_count=2)
    .dropna()
)

kelly_returns.name = 'fractional kelly'
kelly_returns.index.name = 'date'

kelly_returns.to_frame().to_csv(
    OUTPUT_DIR / 'fractional_kelly_returns.csv'
)