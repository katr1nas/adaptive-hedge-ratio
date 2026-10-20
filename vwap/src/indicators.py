import pandas as pd
import numpy as np
from data import load_data
from sessions import prepare_rth



class Indicator:
    def __init__(self, df):
        self.df = prepare_rth(df)

        required = {"high", "low", "close", "volume", "sessions"}
        missing = required.difference(self.df.columns)

        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        if not self.df.index.is_monotonic_increasing:
            raise ValueError("Data must be sorted chronologically")

        if self.df.index.has_duplicates:
            raise ValueError("Duplicate timestamps detected")
        
        if self.df["sessions"].isna().any():
            raise ValueError("Session labels must not be missing")
        
    def calculate_session_vwap(self) -> pd.Series:
        prices = self.df[["high", "low", "close"]].apply(
            pd.to_numeric, errors="raise"
            ).astype(float)

        if self.df.empty:
            return pd.Series(index=self.df.index, dtype=float, name="vwap")
        
        volume = pd.to_numeric(
            self.df['volume'], errors='raise'
        ).astype(float)

        if not np.isfinite(prices.to_numpy()).all():
            raise ValueError("Prices must be finite")
        
        if not np.isfinite(volume.to_numpy()).all() or (volume < 0).any():
            raise ValueError("Volume must be finite and nonnegative")
        
        typical_price = prices.mean(axis=1) 
        price_volume = typical_price * volume

        cumulative_pv = price_volume.groupby(
            self.df["sessions"], sort=False
        ).cumsum()

        cumulative_volume = volume.groupby(
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

        valid_volume = cumulative_volume > 0


        cumulative_pv = (prices * volume).groupby(sessions, sort=False).cumsum()
        cumulative_pv2 = (prices**2 * volume).groupby(sessions, sort=False).cumsum()

        vwap = cumulative_pv.div(
            cumulative_volume.where(valid_volume)
        )

        second_moment = cumulative_pv2.div(
            cumulative_volume.where(valid_volume)
            )
        
        variance = (second_moment - vwap**2).clip(lower=0)
        omega = np.sqrt(variance)

        result = pd.DataFrame(index=self.df.index)
        result["vwap"] = vwap

        for k in (1, 2, 3):
            result[f"upper_{k}"] = vwap + k * omega
            result[f"lower_{k}"] = vwap - k * omega

        return result
    

    def get_signal(self) -> pd.DataFrame:
        bands = self.calculate_vwap_bands()

        close = pd.to_numeric(self.df['close'], errors='raise')
        lower = bands['lower_2']
        upper = bands['upper_2']
        same_session = self.df['sessions'].eq(
            self.df['sessions'].shift(1)
        )

        valid_bands = (
            np.isfinite(upper)
            & np.isfinite(lower)
            & (upper > lower)
            )
 

        previous_valid = valid_bands.shift(1, fill_value=False)

        cross_below = (
            same_session
            & previous_valid
            & valid_bands
            & (close.shift(1) >= lower.shift(1))
            & (close < lower)
        )

        cross_above = (
            same_session
            & previous_valid
            & valid_bands
            & (close.shift(1) <= upper.shift(1))
            & (close > upper)
        )

        signals = pd.DataFrame(index=self.df.index)
        signals["long"] = cross_below.fillna(False).astype(bool)
        signals["short"] = cross_above.fillna(False).astype(bool)

        return signals
       
    
