"""Fetch CAISO solar, wind, and price data.

Uses a mock generator so the pipeline runs end-to-end without network
access.  The real implementation should hit the CAISO OASIS SingleZip
endpoint or a pre-downloaded CSV dump under ``data/raw``.

Returned DataFrame columns:
    - timestamp (tz-aware, UTC)
    - solar_mw
    - wind_mw
    - price_usd_per_mwh   (LMP)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _simulate_caiso(year: int, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic hourly solar, wind, and LMP series for one year."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:00",
        freq="h",
        tz="UTC",
    )
    local = idx.tz_convert("America/Los_Angeles")
    hour = local.hour.to_numpy()
    doy = local.dayofyear.to_numpy()

    solar_shape = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (doy - 80) / 365)
    solar_mw = 12000 * solar_shape * seasonal + rng.normal(0, 300, len(idx))
    solar_mw = np.clip(solar_mw, 0, None)

    wind_mw = (
        3000
        + 1500 * np.sin(2 * np.pi * doy / 365)
        + 800 * np.sin(2 * np.pi * hour / 24 + 1.2)
        + rng.normal(0, 400, len(idx))
    )
    wind_mw = np.clip(wind_mw, 0, None)

    base_price = 45.0
    duck_curve = 25 * np.clip(np.sin(np.pi * (hour - 16) / 8), 0, None)
    scarcity = rng.gamma(shape=1.2, scale=6.0, size=len(idx))
    price = base_price + duck_curve + scarcity - 0.0015 * solar_mw
    price = np.clip(price, -20, 500)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "solar_mw": solar_mw,
            "wind_mw": wind_mw,
            "price_usd_per_mwh": price,
        }
    )


def fetch_caiso_data(
    year: int,
    node: str = "TH_SP15_GEN-APND",
    use_mock: bool = True,
    raw_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load CAISO hourly solar, wind, and price data.

    Parameters
    ----------
    year : int
        Calendar year.
    node : str
        CAISO pricing node (ignored when ``use_mock=True``).
    use_mock : bool
        Whether to generate synthetic data.
    raw_dir : str or Path, optional
        Optional cache directory.

    Returns
    -------
    pandas.DataFrame
        Hourly CAISO frame in UTC.
    """
    if use_mock:
        df = _simulate_caiso(year=year)
    else:
        raise NotImplementedError(
            "Live CAISO fetch not implemented. Set sources.caiso_use_mock=true "
            "in config/params.yaml to use simulated data."
        )

    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = raw_dir / f"caiso_{node}_{year}.csv"
        df.to_csv(out, index=False)

    return df
