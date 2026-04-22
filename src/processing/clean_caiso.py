"""Clean and validate raw CAISO data.

Accepts frames produced by either path in ``fetch_caiso``:
  * Real data (loaded from CAISO_<year>_combined.csv) – already
    normalised by the loader, so cleaning is mostly a no-op safety
    check.
  * Mock / synthetic data – same schema, same safety checks.

Expected columns after ``fetch_caiso_data`` runs:
    timestamp           – tz-aware UTC
    solar_mw            – MW (may still contain small negatives from the
                          raw CAISO export; clipped here as well)
    wind_mw             – MW
    price_usd_per_mwh   – $/MWh (real or synthesised)
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLS = ("timestamp", "solar_mw", "wind_mw", "price_usd_per_mwh")


def clean_caiso(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a CAISO frame that has already been loaded by ``fetch_caiso``.

    Steps
    -----
    1. Validate that all required columns are present.
    2. Parse ``timestamp`` to tz-aware UTC (safe to call even if it is
       already a DatetimeTZDtype).
    3. Drop duplicate timestamps; sort ascending.
    4. Clip negative solar / wind to 0 (physical floor).
    5. Forward-fill gaps of ≤ 3 hours, then drop any remaining NaNs.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame from ``fetch_caiso_data``.

    Returns
    -------
    pandas.DataFrame
        Cleaned frame with the same schema, reset integer index.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CAISO frame is missing required columns: {missing}\n"
            "Make sure fetch_caiso_data() ran successfully before calling "
            "clean_caiso()."
        )

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = (
        out.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Physical floor: generation cannot be negative
    for col in ("solar_mw", "wind_mw"):
        out[col] = out[col].clip(lower=0)

    out = out.set_index("timestamp")
    out = out.ffill(limit=3).dropna()
    return out.reset_index()
