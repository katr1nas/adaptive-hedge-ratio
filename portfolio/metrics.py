import numpy as np
import pandas as pd

def max_drawdown(returns):
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = drawdown.min()

    return max_drawdown

def metrics(returns):
    metrics = pd.DataFrame({
    "annual_return": (
        (1 + returns).prod() ** (252 / len(returns)) - 1
    ),
    "annual_volatility": returns.std() * np.sqrt(252),
    "sharpe": (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
    ),
    "max_drawdown": returns.apply(max_drawdown),
    })
    return metrics



