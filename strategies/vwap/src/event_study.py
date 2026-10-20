import pandas as pd
import numpy as np
from indicators import Indicator
from data import load_data

class EventStudy:
    def __init__(self, df):
        self.df = df
        
        self.new_session = self.df['sessions'].ne(self.df['sessions'].shift())
        self.ind = Indicator(self.df)
        self.signals = self.ind.get_signal()
        self.bands = self.ind.calculate_vwap_bands()

        self.long_starts = self.signals['long'] & (
            ~self.signals['long'].shift(1, fill_value=False) | self.new_session
        )
        
        self.short_starts = self.signals['short'] & (
            ~self.signals['short'].shift(1, fill_value=False) | self.new_session
        )
        
        events = self.df.loc[
            self.long_starts | self.short_starts,
            ['close', 'sessions']].copy()
    
        events['vwap'] = self.bands.loc[events.index, 'vwap']   
        events['signal'] = np.where(
            self.long_starts.loc[events.index],
            'long',
            'short'
            )
        events['entry'] = df['open'].shift(-1).loc[events.index]
        events['timestamp'] = events.index
        events['upper_2'] = self.bands.loc[events.index, 'upper_2'].copy()
        events['lower_2'] = self.bands.loc[events.index, 'lower_2'].copy()
    
        self.events = events

    def calculate_data(self):
        horizons = (5, 15, 30)

        for h in horizons:
            future_close = self.df["close"].shift(-h)
            future_session = self.df["sessions"].shift(-h)
            valid = self.df["sessions"].eq(future_session)
            long_mask = self.events["signal"].eq("long")
            short_mask = self.events["signal"].eq("short")

            future_at_event = future_close.reindex(self.events.index)
            valid_at_event = valid.reindex(self.events.index).fillna(False)

            change = pd.Series(np.nan, index=self.events.index, dtype=float)
            change.loc[long_mask] = (
                future_at_event.loc[long_mask] - self.events.loc[long_mask, "close"]
            )
            change.loc[short_mask] = (
                self.events.loc[short_mask, "close"] - future_at_event.loc[short_mask]
            )

            self.events[f"return_{h}"] = change.where(valid_at_event)

        return self.events