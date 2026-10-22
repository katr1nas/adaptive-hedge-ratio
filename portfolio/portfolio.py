import pandas as pd
import numpy as np
from pathlib import Path
from correlation import load_returns
from metrics import metrics as calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
equal_weight_return = load_returns("equal_weight_returns.csv", 'equal_weights_returns', column = 'equal_weight')


cpo_returns = load_returns('cpo_returns.csv', 'cpo')
kalman_returns = load_returns('kalman_returns.csv', 'kalman')

strategy_returns = pd.concat(
    [cpo_returns, kalman_returns],
    axis=1,
    join='inner'
).dropna()

rolling_volatility = (
    strategy_returns
    .rolling(window=63, min_periods=21)
    .std()
    * np.sqrt(252)
)

inverse_volatility = 1 / rolling_volatility

weights = inverse_volatility.div(
    inverse_volatility.sum(axis=1),
    axis=0
)

weights = weights.shift(1)

inverse_vol_returns = (
    (weights * strategy_returns)
    .sum(axis=1, min_count=2)
    .dropna()
)

inverse_vol_returns.name = 'inverse_volatility'
inverse_vol_returns.index.name = 'date'

inverse_vol_returns.to_frame().to_csv(
    OUTPUT_DIR / 'inverse_volatility_returns.csv'
)

print(calculate_metrics(inverse_vol_returns.to_frame()))
print('\nLatest weights:')
print(weights.dropna().tail())


comparison_returns = pd.concat(
    [equal_weight_return, inverse_vol_returns],
    axis=1,
    join="inner",
).dropna()

print(calculate_metrics(comparison_returns))
