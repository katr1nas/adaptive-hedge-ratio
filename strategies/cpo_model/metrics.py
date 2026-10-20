import numpy as np

def sharpe_ratio(strategy_return):
    annual_return = strategy_return.mean() * 252
    annual_volatility = strategy_return.std() * np.sqrt(252)

    return annual_return / annual_volatility

def calculate_z_score(data, period=20):
    close = data['close'].squeeze()

    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - mean) / std

