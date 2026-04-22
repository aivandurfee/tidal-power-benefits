"""Filter merged hourly data into conditional subsets.

Each function returns a *new* DataFrame and leaves the input unchanged.
The merged frame is expected to have:
  - a tz-aware UTC index
  - columns ``hour``, ``month`` (local time) added by ``align_time``
  - a price column (default ``price_usd_per_mwh``)
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def evening_hours(df: pd.DataFrame, hours: Iterable[int]) -> pd.DataFrame:
    """Return rows whose local hour is in ``hours`` (e.g. 17..21)."""
    return df[df["hour"].isin(list(hours))].copy()


def winter_months(df: pd.DataFrame, months: Iterable[int]) -> pd.DataFrame:
    """Return rows whose local month is in ``months`` (e.g. [12,1,2])."""
    return df[df["month"].isin(list(months))].copy()


def high_price_hours(
    df: pd.DataFrame,
    quantile: float = 0.90,
    price_col: str = "price_usd_per_mwh",
) -> pd.DataFrame:
    """Return rows whose price is above the given quantile threshold."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1 (exclusive)")
    threshold = df[price_col].quantile(quantile)
    return df[df[price_col] >= threshold].copy()


def build_slices(df: pd.DataFrame, time_slices_cfg: dict) -> dict[str, pd.DataFrame]:
    """Build a dict of named slice DataFrames from the config block.

    Expected config keys: ``evening_hours``, ``winter_months``,
    ``summer_months``, ``high_price_quantile``.
    """
    slices: dict[str, pd.DataFrame] = {"all_hours": df.copy()}
    if "evening_hours" in time_slices_cfg:
        slices["evening"] = evening_hours(df, time_slices_cfg["evening_hours"])
    if "winter_months" in time_slices_cfg:
        slices["winter"] = winter_months(df, time_slices_cfg["winter_months"])
    if "summer_months" in time_slices_cfg:
        slices["summer"] = winter_months(df, time_slices_cfg["summer_months"])
    if "high_price_quantile" in time_slices_cfg:
        slices["high_price"] = high_price_hours(
            df, quantile=time_slices_cfg["high_price_quantile"]
        )
    return slices
