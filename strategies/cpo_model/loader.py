import yfinance as yf
import pandas as pd

def loader(symbol: str, start: str, end: str):
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.droplevel(1)
    df.columns = [c.lower() for c in df.columns]
    return df 
