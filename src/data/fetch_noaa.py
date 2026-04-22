"""Fetch NOAA tidal current data.

For now this module returns a **simulated** hourly tidal-current series
so downstream modules can be developed without a live API.  The real
implementation should call the NOAA Tides & Currents API:

    https://api.tidesandcurrents.noaa.gov/api/prod/

The returned DataFrame always has the columns:
    - timestamp (tz-aware, UTC)
    - current_speed_mps   (tidal current magnitude in m/s)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _simulate_tidal_currents(year: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic hourly tidal-current series for one year.

    Uses two dominant semi-diurnal / diurnal constituents (M2 ~12.42 h,
    K1 ~23.93 h) plus noise.  Peak speeds are around 2.5 m/s.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:00",
        freq="h",
        tz="UTC",
    )
    hours = np.arange(len(idx))
    m2 = 1.5 * np.sin(2 * np.pi * hours / 12.42)
    k1 = 0.7 * np.sin(2 * np.pi * hours / 23.93 + 0.4)
    spring_neap = 0.3 * np.sin(2 * np.pi * hours / (14.77 * 24))
    noise = rng.normal(0.0, 0.1, size=len(idx))
    speed = np.abs(m2 + k1 + spring_neap + noise)
    return pd.DataFrame({"timestamp": idx, "current_speed_mps": speed})


def fetch_noaa_currents(
    year: int,
    station_id: str = "9414290",
    use_mock: bool = True,
    raw_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load NOAA tidal current data for a given year.

    Parameters
    ----------
    year : int
        Calendar year to fetch.
    station_id : str
        NOAA station identifier (ignored when ``use_mock=True``).
    use_mock : bool
        If True, generate synthetic data. If False, a real API call
        should be implemented (currently raises NotImplementedError).
    raw_dir : str or Path, optional
        If provided, the fetched frame is also written to
        ``raw_dir/noaa_<station>_<year>.csv``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp`` (UTC), ``current_speed_mps``.
    """
    if use_mock:
        df = _simulate_tidal_currents(year=year)
    else:
        raise NotImplementedError(
            "Live NOAA API fetch not implemented. Set sources.noaa_use_mock=true "
            "in config/params.yaml to use simulated data."
        )

    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = raw_dir / f"noaa_{station_id}_{year}.csv"
        df.to_csv(out, index=False)

    return df
