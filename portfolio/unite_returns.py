from correlation import load_returns
from pathlib import Path
import pandas as pd
from metrics import metrics as calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

equal_weights = load_returns(
    'equal_weight_returns.csv',
    name='equal_weight',
    column='equal_weight'
)

inverse_volatility = load_returns(
    'inverse_volatility_returns.csv',
    name='inverse_volatility',
    column='inverse_volatility'
)

kelly_returns = load_returns(
    'fractional_kelly_returns.csv',
    name='fractional_kelly',
    column='fractional kelly'
)

comparison_returns = pd.concat(
    [
        equal_weights,
        inverse_volatility,
        kelly_returns
    ],
    axis=1,
    join='inner',
).dropna()

comparison_metrics = calculate_metrics(comparison_returns)

print("Comparison dates:")
print(
    comparison_returns.index.min(),
    "to",
    comparison_returns.index.max(),
)

print("\nObservations:", len(comparison_returns))

print("\nPortfolio comparison:")
print(comparison_metrics)

comparison_returns.to_csv(
    OUTPUT_DIR / "portfolio_comparison_returns.csv"
)