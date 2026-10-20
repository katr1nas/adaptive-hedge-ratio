from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


COMMON_DATE_COLUMNS = ("date", "Date", "datetime", "Datetime", "timestamp", "Timestamp")
DuplicatePolicy = Literal["raise", "first", "last", "mean", "sum"]


@dataclass(frozen=True)
class ReturnSource:
    name: str
    path: Path
    return_column: str
    date_column: str | None


def _find_date_column(frame: pd.DataFrame, requested: str | None, source: ReturnSource) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"{source.name}: date column {requested!r} was not found. Available columns: {list(frame.columns)}")
        return requested

    for column in COMMON_DATE_COLUMNS:
        if column in frame.columns:
            return column
    first_column = frame.columns[0]
    parsed = pd.to_datetime(frame[first_column], errors="coerce")
    if parsed.notna().all():
        return first_column

    raise ValueError(
        f"{source.name}: could not identify a date column. Pass --{source.name}-date-col. "
        f"Available columns: {list(frame.columns)}"
    )


def _resolve_duplicates(series: pd.Series, policy: DuplicatePolicy, name: str) -> pd.Series:
    duplicate_count = int(series.index.duplicated(keep=False).sum())
    if not duplicate_count:
        return series

    if policy == "raise":
        dates = series.index[series.index.duplicated(keep=False)].unique()[:5]
        raise ValueError(
            f"{name}: found {duplicate_count} rows on duplicate dates (examples: {list(dates)}). "
            "Choose an explicit --duplicate-policy if aggregation is intended."
        )

    if policy in {"first", "last"}:
        return series[~series.index.duplicated(keep=policy)]

    return series.groupby(level=0).mean() if policy == "mean" else series.groupby(level=0).sum(min_count=1)


def load_returns(source: ReturnSource, duplicate_policy: DuplicatePolicy, normalize_dates: bool) -> pd.Series:
    """Load one return column, validate it, and return a sorted datetime Series."""
    if not source.path.is_file():
        raise FileNotFoundError(f"{source.name}: input file does not exist: {source.path}")

    frame = pd.read_csv(source.path)
    date_column = _find_date_column(frame, source.date_column, source)
    if source.return_column not in frame.columns:
        raise ValueError(
            f"{source.name}: return column {source.return_column!r} was not found. "
            f"Available columns: {list(frame.columns)}"
        )

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        bad_rows = list(frame.index[dates.isna()][:5])
        raise ValueError(f"{source.name}: unparseable dates at CSV rows {bad_rows}")
    if normalize_dates:
        dates = dates.dt.normalize()

    returns = pd.to_numeric(frame[source.return_column], errors="coerce")
    non_numeric = frame[source.return_column].notna() & returns.isna()
    if non_numeric.any():
        bad_rows = list(frame.index[non_numeric][:5])
        raise ValueError(f"{source.name}: non-numeric returns at CSV rows {bad_rows}")
    if np.isinf(returns).any():
        raise ValueError(f"{source.name}: returns contain infinity")

    series = pd.Series(returns.to_numpy(), index=pd.DatetimeIndex(dates), name=source.name)
    return _resolve_duplicates(series, duplicate_policy, source.name).sort_index()


def combine_returns(
    sources: list[ReturnSource],
    join: Literal["inner", "outer"],
    duplicate_policy: DuplicatePolicy,
    normalize_dates: bool,
    drop_missing: bool,
) -> pd.DataFrame:
    series = [load_returns(source, duplicate_policy, normalize_dates) for source in sources]
    combined = pd.concat(series, axis=1, join=join).sort_index()
    combined.index.name = "date"
    if drop_missing:
        combined = combined.dropna(how="any")
    return combined


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Date-align CPO, VWAP, and Kalman return CSVs.")
    for strategy, default_file, default_return in (
        ("cpo", "cpo_model/cpo_returns.csv", "actual_return"),
        ("vwap", "vwap/vwap_returns.csv", "strategy_return"),
        ("kalman", "portfolio/kalman_returns.csv", "pnl"),
    ):
        parser.add_argument(f"--{strategy}-file", default=default_file, help=f"CSV containing {strategy.upper()} returns")
        parser.add_argument(f"--{strategy}-return-col", default=default_return, help=f"Return column in the {strategy.upper()} CSV")
        parser.add_argument(f"--{strategy}-date-col", default=None, help="Date column (auto-detected when omitted)")

    parser.add_argument("--output", default="portfolio/combined_returns.csv", help="Destination CSV")
    parser.add_argument("--join", choices=("inner", "outer"), default="inner", help="Date alignment: common dates only (inner) or every date (outer)")
    parser.add_argument("--duplicate-policy", choices=("raise", "first", "last", "mean", "sum"), default="raise", help="Required choice for duplicate dates; default rejects them")
    parser.add_argument("--keep-time", action="store_true", help="Keep timestamps instead of collapsing them to calendar dates")
    parser.add_argument("--drop-missing", action="store_true", help="Explicitly remove rows with a missing strategy return; never fills missing returns")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    sources = [
        ReturnSource("cpo", Path(args.cpo_file), args.cpo_return_col, args.cpo_date_col),
        ReturnSource("vwap", Path(args.vwap_file), args.vwap_return_col, args.vwap_date_col),
        ReturnSource("kalman", Path(args.kalman_file), args.kalman_return_col, args.kalman_date_col),
    ]
    combined = combine_returns(
        sources=sources,
        join=args.join,
        duplicate_policy=args.duplicate_policy,
        normalize_dates=not args.keep_time,
        drop_missing=args.drop_missing,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=True)
    missing = combined.isna().sum().to_dict()
    print(f"Wrote {len(combined)} aligned rows to {output}")
    print(f"Missing returns retained in output: {missing}")


if __name__ == "__main__":
    main()
