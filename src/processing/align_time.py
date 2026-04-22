"""Align multiple data sources to a common hourly UTC index."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def to_hourly_utc(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Return a copy of ``df`` on an hourly tz-aware UTC DatetimeIndex.

    Works for two shapes:
      1. A frame with a ``timestamp_col`` (default ``timestamp``).
      2. A frame that already has a ``DatetimeIndex`` (its name is
         preserved, e.g. ``datetime``).

    Sub-hourly data is averaged to hourly; already-hourly data is
    deduplicated and sorted.
    """
    out = df.copy()

    if timestamp_col in out.columns:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True)
        out = (
            out.drop_duplicates(subset=timestamp_col)
            .sort_values(timestamp_col)
            .set_index(timestamp_col)
        )
    elif isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        out = out[~out.index.duplicated(keep="first")].sort_index()
    else:
        raise ValueError(
            f"DataFrame needs either a '{timestamp_col}' column or a DatetimeIndex."
        )

    numeric = out.select_dtypes(include="number")
    return numeric.resample("1h").mean()


def align_datasets(
    frames: Dict[str, pd.DataFrame],
    timezone: str = "America/Los_Angeles",
    price_col: str = "price_usd_per_mwh",
) -> pd.DataFrame:
    """Merge multiple hourly frames into a single wide DataFrame.

    Each frame is first passed through :func:`to_hourly_utc`, then the
    resulting frames are inner-joined on their UTC hourly index.  Local
    time helpers (``local_time``, ``hour``, ``month``, ``dayofweek``)
    are appended for downstream slicing.  Hours where the real CAISO
    CSV has no price row are dropped so value metrics aren't biased
    by NaN prices.
    """
    hourly_frames = [to_hourly_utc(f) for f in frames.values()]
    merged = pd.concat(hourly_frames, axis=1, join="inner").sort_index()

    # Drop hours where price is missing (e.g. CAISO CSV gaps outside its range).
    if price_col in merged.columns:
        merged = merged.dropna(subset=[price_col])

    local = merged.index.tz_convert(timezone)
    merged["local_time"] = local
    merged["hour"] = local.hour
    merged["month"] = local.month
    merged["dayofweek"] = local.dayofweek
    return merged
