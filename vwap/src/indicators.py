import pandas as pd
import numpy as np



class Indicator:
    def __init__(self, df):
        self.df = df

        required = {"high", "low", "close", "volume", "sessions"}
        missing = required.difference(df.columns)

        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        if not df.index.is_monotonic_increasing:
            raise ValueError("Data must be sorted chronologically")

        if df.index.has_duplicates:
            raise ValueError("Duplicate timestamps detected")
        
    def calculate_session_vwap(self) -> pd.Series:
        prices = self.df[["high", "low", "close"]]

        if self.df.empty:
            return pd.Series(index=self.df.index, dtype=float, name="vwap")

        numeric_prices = prices.apply(pd.to_numeric, errors="coerce")

        if numeric_prices.isna().any().any():
            raise ValueError("Prices must be numeric and non-missing")

        if not numeric_prices.map(lambda x: abs(x) != float("inf")).all().all():
            raise ValueError("Prices must be finite")

        typical_price = (
            self.df["high"] + self.df["low"] + self.df["close"]
        ) / 3.0

        price_volume = typical_price * self.df["volume"]

        cumulative_pv = price_volume.groupby(
            self.df["sessions"], sort=False
        ).cumsum()

        cumulative_volume = self.df["volume"].groupby(
            self.df["sessions"], sort=False
        ).cumsum()

        vwap = cumulative_pv.div(
            cumulative_volume.where(cumulative_volume > 0)
        )

        return vwap.rename("vwap")


    def calculate_vwap_bands(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame(
                index=self.df.index,
                columns=[
                    "vwap", "upper_1", "lower_1",
                    "upper_2", "lower_2", "upper_3", "lower_3"
                ],
                dtype=float,
            )

        prices = (
            self.df["high"].astype(float)
            + self.df["low"].astype(float)
            + self.df["close"].astype(float)
        ) / 3.0

        volume = pd.to_numeric(self.df["volume"], errors="raise").astype(float)

        if not np.isfinite(prices).all():
            raise ValueError("Prices must be finite")

        if not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError("Volume must be finite and nonnegative")

        sessions = self.df["sessions"]
        cumulative_volume = volume.groupby(sessions, sort=False).cumsum()

        if (cumulative_volume <= 0).any():
            raise ValueError("Cumulative session volume must be positive")

        cumulative_pv = (prices * volume).groupby(sessions, sort=False).cumsum()
        cumulative_pv2 = (prices**2 * volume).groupby(sessions, sort=False).cumsum()

        vwap = cumulative_pv / cumulative_volume
        variance = (cumulative_pv2 / cumulative_volume - vwap**2).clip(lower=0)
        omega = np.sqrt(variance)

        result = pd.DataFrame(index=self.df.index)
        result["vwap"] = vwap

        for k in (1, 2, 3):
            result[f"upper_{k}"] = vwap + k * omega
            result[f"lower_{k}"] = vwap - k * omega

        return result
    

    def get_signal(self) -> pd.DataFrame:
        result = self.calculate_vwap_bands()

        band_width = result['upper_2'] - result['lower_2']

        signals = pd.DataFrame(index=self.df.index)
        signals["long"] = (self.df["close"] < result["lower_2"]) & (band_width > 0)
        signals["short"] = (self.df["close"] > result["upper_2"]) & (band_width > 0)

        return signals
