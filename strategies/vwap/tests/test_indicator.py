import pandas as pd
import pytest 

from src.indicators import calculate_session_vwap 

def make_df():
    return pd.DataFrame({
        "high": [100, 102, 110],
        "low": [100, 102, 110],
        "close": [100, 102, 110],
        "volume": [10, 30, 20],
        "sessions": ["A", "A", "B"],
    })

def test_vwap_calculation_and_session_rest():
    df = make_df()

    result = calculate_session_vwap(df)

    assert result.tolist() == pytest.approx(
        [100.0, 101.5, 110.0]
    )

def test_zero_volume_returns_nan_until_volume_exists():
    df = pd.DataFrame({
        "high": [100, 102],
        "low": [100, 102],
        "close": [100, 102],
        "volume": [0, 10],
        "sessions": ["A", "A"],
    })

    result = calculate_session_vwap(df)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(102.0)


def test_missing_column_raises_error():
    df = make_df().drop(columns=["high"])

    with pytest.raises(ValueError, match="Missing columns"):
        calculate_session_vwap(df)


def test_unsorted_index_raises_error():
    df = make_df()
    df.index = [2, 1, 3]

    with pytest.raises(ValueError, match="sorted chronologically"):
        calculate_session_vwap(df)