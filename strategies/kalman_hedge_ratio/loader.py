import yfinance as yf
import pandas as pd

def load_data(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False)
    return data

