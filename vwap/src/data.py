import yfinance as yf
import pandas as pd

def load_data(symbols: list[str], period: str = '7d', interval: str = '1m') -> pd.DataFrame:
    df = yf.download(symbols, period=period, interval=interval)
    sym_1 = df.xs(symbols[0], axis=1, level='Ticker').copy()
    sym_2 = df.xs(symbols[1], axis=1, level='Ticker').copy()
    return sym_1, sym_2