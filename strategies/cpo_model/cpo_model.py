import pandas as pd
from metrics import calculate_z_score
from itertools import product
from loader import loader
import numpy as np
from train_test_split import train_test_split
from get_result import get_result
from metrics import sharpe_ratio
from pathlib import Path

df = loader('ES=F', '2017-01-01', '2026-01-01')

def generate_signals(z_score, threshold=2.0):
    signals = pd.Series(0, index=z_score.index)

    signals[z_score < -threshold] = 1
    signals[z_score > threshold] = -1

    return signals

def backtest_strategy(data, period=20, threshold=2.0):
    z_score = calculate_z_score(data, period)
    signals = generate_signals(z_score, threshold)
    returns = data['close'].pct_change()

    strategy_return = signals.shift(1) * returns

    return strategy_return.fillna(0)

windows = [5, 10, 20, 15, 30, 40]
threshold = [0.5, 1, 1.5, 1.7, 2, 2.5]

combinations = list(product(windows, threshold))

def generate_return(combination):
    window, threshold = combination

    z_score = calculate_z_score(df, window)
    strategy_return = backtest_strategy(df, window, threshold)

    returns = df['close'].pct_change()

    result = pd.DataFrame({
        'window': window,
        'threshold': threshold,
        'strategy_return': strategy_return,
        'z_score': z_score,
        'volatility': returns.rolling(5).std(),
        'momentum': df['close'].pct_change(5)
    })
    result['target'] = result['strategy_return'].shift(-1)

    return result

results = [
    generate_return(combination)
    for combination in combinations
]

cpo_df = pd.concat(results)
cpo_df = cpo_df.replace([np.inf, -np.inf], np.nan)
cpo_df = cpo_df.dropna()
cpo_df = cpo_df.sort_index()


returns = df['close'].pct_change()
df['volatility'] = returns.rolling(5).std()
df['momentum'] = df['close'].pct_change(5)

cpo_df['volatility'] = df['volatility']
cpo_df['momentum'] = df['momentum']

features = [
    'window',
    'threshold',
    'z_score',
    'volatility',
    'momentum'
]

X = cpo_df[features]
y = cpo_df['target']

X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y, cpo_df)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

cpo_rfg = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=10, n_jobs=-1)
cpo_rfg.fit(X_train, y_train)

val_pred = cpo_rfg.predict(X_val)
val_mse = mean_squared_error(y_val, val_pred)

fixed_mask = (
    (X_val['window'] == 20) &
    (X_val['threshold'] == 1.0)
)

fixed_returns = y_val.loc[fixed_mask].sort_index()

best_daily_val = get_result(X_val, val_pred, y_val)
cpo_returns = best_daily_val['actual_return']

print("CPO Sharpe:", sharpe_ratio(cpo_returns))
print("Fixed Sharpe:", sharpe_ratio(fixed_returns))

print("CPO Total Return:", (1 + cpo_returns).prod() - 1)
print("Fixed Total Return:", (1 + fixed_returns).prod() - 1)

def generate_returns():
    test_pred = cpo_rfg.predict(X_test)
    best_daily_test = get_result(X_test, test_pred, y_test)

    cpo_returns = best_daily_test['actual_return'].copy()
    next_dates = pd.Series(df.index[1:].values, index=df.index[:-1])
    cpo_returns.index = pd.DatetimeIndex(next_dates.reindex(cpo_returns.index).values)

    cpo_returns.name = 'return'
    cpo_returns.index.name = 'date'
    cpo_returns = cpo_returns.dropna().sort_index()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    output =  PROJECT_ROOT / "outputs" / "cpo_returns.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    cpo_returns.to_frame().to_csv(output)

    return cpo_returns

if __name__ == "__main__":
    cpo_returns = generate_returns()
    print('CPO sharpe:', sharpe_ratio(cpo_returns))
    print("CPO Total Return:", (1 + cpo_returns).prod() - 1)







