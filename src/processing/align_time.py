"""Align multiple data sources to a common hourly UTC index."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def to_hourly_utc(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Return a copy of ``df`` indexed by hourly tz-aware UTC timestamps.

    If the incoming frequency is sub-hourly it is averaged to hourly.
    If it is already hourly, the frame is passed through unchanged
    aside from sorting + dedup.
    """
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True)
    out = out.drop_duplicates(subset=timestamp_col).sort_values(timestamp_col)
    out = out.set_index(timestamp_col)
    numeric = out.select_dtypes(include="number")
    hourly = numeric.resample("1h").mean()
    return hourly


def align_datasets(
    frames: Dict[str, pd.DataFrame],
    timezone: str = "America/Los_Angeles",
) -> pd.DataFrame:
    """Merge multiple hourly frames into a single wide DataFrame.

    Parameters
    ----------
    frames : dict[str, pandas.DataFrame]
        Mapping of ``name -> frame``.  Each frame must have a
        ``timestamp`` column.  Columns from each frame are retained as-is
        (callers should pre-rename if collisions are possible).
    timezone : str
        Local timezone to add as an auxiliary column for convenience.

    Returns
    -------
    pandas.DataFrame
        Wide frame indexed by UTC hourly timestamps, with extra columns
        ``local_time``, ``hour``, ``month``, ``dayofweek``.
    """
    hourly_frames = [to_hourly_utc(f) for f in frames.values()]
    merged = pd.concat(hourly_frames, axis=1, join="inner")
    merged = merged.sort_index()

    local = merged.index.tz_convert(timezone)
    merged["local_time"] = local
    merged["hour"] = local.hour
    merged["month"] = local.month
    merged["dayofweek"] = local.dayofweek
    return merged
