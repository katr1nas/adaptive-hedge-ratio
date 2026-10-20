import pandas as pd
from zoneinfo import ZoneInfo


def prepare_rth(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Handle a single-instrument OHLCV DataFrame.
    data.columns = data.columns.str.lower()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex")

    if data.index.tz is None:
        raise ValueError("Timestamps must be timezone-aware")

    # Explicitly construct the timezone object.
    ny_tz = ZoneInfo("America/New_York")
    data.index = data.index.tz_convert(ny_tz)

    # Regular trading hours: 09:30 inclusive, 16:00 exclusive.
    data = data.between_time(
        "09:30",
        "16:00",
        inclusive="left",
    ).copy()

    data["sessions"] = data.index.date

    return data