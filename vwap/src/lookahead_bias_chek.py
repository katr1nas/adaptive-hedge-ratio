from indicators import Indicator
from data import load_data
import pandas as pd
from sessions import prepare_rth
from event_study import EventStudy

symbols = load_data(['ES=F', 'NQ=F'])
df = symbols[0]

df.columns = [
    str(c).lower() if isinstance(c, str) else c
    for c in df.columns
]

df_rth = prepare_rth(df)


def build_signals(df):
    ind = Indicator(df.copy())
    return ind.calculate_vwap_bands().join(ind.get_signal())


def test_prefix_invariance(df, cutoff):
    full = build_signals(df)
    prefix = df.iloc[:cutoff + 1].copy()
    truncated = build_signals(prefix)

    cols = ['vwap', 'upper_2', 'lower_2', 'long', 'short']

    pd.testing.assert_frame_equal(
        full.loc[prefix.index, cols],
        truncated[cols],
        check_dtype=False,
        check_freq=False,
    )

    print(f"PASS: first {cutoff + 1} rows are invariant.")


for cutoff in (40, 500, 2000, 5000):
    test_prefix_invariance(df_rth, cutoff)

es = EventStudy(df_rth.copy())
study = es.calculate_data()
print(study.head(10))
