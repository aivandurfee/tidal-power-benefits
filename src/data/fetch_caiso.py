"""Fetch CAISO solar and wind data.

Real data path:
    Loads from a pre-downloaded CSV (e.g. CAISO_2024_combined.csv) under
    ``data/raw``.  The file is expected to have at minimum the columns:
        - interval_start_utc   (ISO-8601, tz-aware)
        - solar                (MW)
        - wind                 (MW)

    If a ``price_usd_per_mwh`` column is absent the loader synthesises a
    plausible LMP series so the rest of the pipeline stays functional.

Mock data path:
    Falls back to a synthetic generator so the pipeline runs end-to-end
    without any files on disk (controlled by ``sources.caiso_use_mock``
    in ``config/params.yaml``).

Returned DataFrame columns (all callers depend on these names):
    - timestamp           (tz-aware UTC, used as the join key)
    - solar_mw            (MW, non-negative)
    - wind_mw             (MW, non-negative)
    - price_usd_per_mwh   ($/MWh, LMP – real or synthesised)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Real-data loader
# ---------------------------------------------------------------------------

def _load_caiso_csv(csv_path: Path) -> pd.DataFrame:
    """Read a CAISO combined CSV and normalise to the pipeline schema.

    Handles:
    * Negative solar readings (clipped to 0 – occur at night due to
      meter offsets in the raw CAISO export).
    * A small number of NaN rows (forward-filled up to 3 h, remainder
      dropped) – matching the behaviour of ``clean_caiso``.
    * Missing ``price_usd_per_mwh`` column – synthesised from a simple
      duck-curve model so downstream value-metric code always has a price
      series to work with.
    """
    df = pd.read_csv(csv_path)

    # ---- timestamp --------------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["interval_start_utc"], utc=True)

    # ---- rename generation columns ----------------------------------------
    df = df.rename(columns={"solar": "solar_mw", "wind": "wind_mw"})

    # ---- keep only the columns the pipeline needs -------------------------
    keep = ["timestamp", "solar_mw", "wind_mw"]
    if "price_usd_per_mwh" in df.columns:
        keep.append("price_usd_per_mwh")
    df = df[keep].copy()

    # ---- clip negative generation to 0 ------------------------------------
    df["solar_mw"] = df["solar_mw"].clip(lower=0)
    df["wind_mw"] = df["wind_mw"].clip(lower=0)

    # ---- fill short gaps, drop the rest -----------------------------------
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )
    df = df.set_index("timestamp").ffill(limit=3).dropna().reset_index()

    # ---- synthesise LMP if not present ------------------------------------
    if "price_usd_per_mwh" not in df.columns:
        df["price_usd_per_mwh"] = _synthesise_price(df)

    return df


def _synthesise_price(df: pd.DataFrame) -> np.ndarray:
    """Return a plausible hourly LMP series correlated with the duck curve.

    Uses local hour extracted from the UTC timestamp so it works even
    when the timezone conversion hasn't happened yet.
    """
    rng = np.random.default_rng(seed=99)
    local = df["timestamp"].dt.tz_convert("America/Los_Angeles")
    hour = local.dt.hour.to_numpy()
    solar = df["solar_mw"].to_numpy()

    base_price = 45.0
    # Evening ramp-up (hours 17-21)
    duck_curve = 25 * np.clip(np.sin(np.pi * (hour - 16) / 8), 0, None)
    scarcity = rng.gamma(shape=1.2, scale=6.0, size=len(df))
    # Solar suppresses midday prices
    price = base_price + duck_curve + scarcity - 0.0015 * solar
    return np.clip(price, -20, 500)


# ---------------------------------------------------------------------------
# Mock / synthetic data (retained for unit-test / CI usage)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_caiso_data(
    year: int,
    node: str = "TH_NP15_GEN-APND",
    use_mock: bool = False,
    raw_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load CAISO hourly solar, wind, and price data.

    Parameters
    ----------
    year : int
        Calendar year.  When loading from a CSV the loader accepts any
        year present in the file; this parameter is used to find the
        right CSV by convention (``CAISO_<year>_combined.csv``).
    node : str
        CAISO pricing node label – kept for API compatibility / logging.
    use_mock : bool
        If True, return fully synthetic data (useful for CI / tests).
        Defaults to False so the real CSV is used when available.
    raw_dir : str or Path, optional
        Directory that contains the raw CSV files.  If omitted the
        function looks in ``data/raw`` relative to the repo root.

    Returns
    -------
    pandas.DataFrame
        Hourly frame with columns: ``timestamp`` (UTC), ``solar_mw``,
        ``wind_mw``, ``price_usd_per_mwh``.

    Raises
    ------
    FileNotFoundError
        If ``use_mock=False`` and no matching CSV is found in ``raw_dir``.
    """
    if use_mock:
        return _simulate_caiso(year=year)

    # ---- locate the CSV ---------------------------------------------------
    if raw_dir is None:
        # Default: <repo_root>/data/raw
        raw_dir = Path(__file__).resolve().parents[3] / "data" / "raw"
    raw_dir = Path(raw_dir)

    # Accept the uploaded filename directly or the conventional name.
    candidates = [
        raw_dir / f"CAISO_{year}_combined.csv",
        raw_dir / f"caiso_{node}_{year}.csv",
    ]
    csv_path: Optional[Path] = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"No CAISO CSV found for year={year} in {raw_dir}.\n"
            f"Looked for: {[str(c) for c in candidates]}\n"
            "Copy your CSV there or set sources.caiso_use_mock=true in "
            "config/params.yaml to fall back to synthetic data."
        )

    return _load_caiso_csv(csv_path)
