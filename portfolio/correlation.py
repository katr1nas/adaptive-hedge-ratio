import pandas as pd
from metrics import metrics as calculate_metrics, max_drawdown
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def load_returns(filename: str, name:str, column: str = 'return') -> pd.Series:
    frame = pd.read_csv(
        OUTPUT_DIR / filename,
        parse_dates=['date'],
        index_col='date'
    )

    return frame[column].rename(name).sort_index()


cpo_returns = load_returns('cpo_returns.csv', 'cpo')
kalman_returns = load_returns('kalman_returns.csv', 'kalman')


combined_returns = pd.concat(
    [cpo_returns, kalman_returns],
    axis=1,
    join='inner'
).dropna()

if __name__ == "__main__":
    metrics = calculate_metrics(combined_returns)

    correlation = combined_returns.corr()
    print('Date range:', combined_returns.index.min(), 'to', combined_returns.index.max())
    print('Observations:', len(combined_returns))

    print('\nMetrics:')
    print(metrics)

    print('\n Correlation:')
    print(correlation)

    equal_weight_returns = combined_returns.mean(axis=1)
    equal_weight_returns.name = 'equal_weight'

    portfolio_metrics = calculate_metrics(equal_weight_returns.to_frame())
    print('\nEqual-weight portfolio metrics:')
    print(portfolio_metrics)

    equal_weight_returns.to_frame().to_csv(
    OUTPUT_DIR / 'equal_weight_returns.csv'
    )
