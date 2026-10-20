import yfinance as yf
import pandas as pd

def load_data(symbol: str, period: str = '7d', interval: str = '1m') -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    
    return df
