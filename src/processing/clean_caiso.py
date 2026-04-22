"""Clean and validate raw CAISO data."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLS = ("timestamp", "solar_mw", "wind_mw", "price_usd_per_mwh")


def clean_caiso(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw CAISO frame.

    Steps:
      1. Validate required columns are present.
      2. Parse ``timestamp`` to tz-aware UTC.
      3. Drop duplicate timestamps, sort ascending.
      4. Clip negative solar/wind to 0 (physical floor).
      5. Forward-fill small gaps (<= 3 hours) then drop any remaining NaNs.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw CAISO frame.

    Returns
    -------
    pandas.DataFrame
        Cleaned frame with the same schema.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CAISO frame missing columns: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = (
        out.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    for col in ("solar_mw", "wind_mw"):
        out[col] = out[col].clip(lower=0)

    out = out.set_index("timestamp")
    out = out.ffill(limit=3).dropna()
    return out.reset_index()
