"""Fetch CAISO solar, wind, and LMP price data.

Real data path
--------------
Generation columns (solar, wind) are loaded from a pre-downloaded CSV
(``CAISO_<year>_combined.csv``) under ``data/raw``.

LMP price column is loaded **separately** from ``LMP_<year>_combined.csv``
(also under ``data/raw``) via :func:`src.data.fetch_lmp.load_lmp_csv`.
The two frames are merged on their UTC timestamp before being returned.

If the LMP file is absent the loader falls back to a synthetic price series
so the rest of the pipeline stays functional, but a clear warning is emitted.

Mock data path
--------------
When ``use_mock=True`` (controlled by ``sources.caiso_use_mock`` in
``config/params.yaml``), fully synthetic generation AND price data are
returned.  Useful for CI / tests.

Returned DataFrame columns (all callers depend on these names):
    - timestamp           (tz-aware UTC, used as the join key)
    - solar_mw            (MW, non-negative)
    - wind_mw             (MW, non-negative)
    - price_usd_per_mwh   ($/MWh, real LMP or synthesised fallback)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.fetch_lmp import load_lmp_csv


log = logging.getLogger("tidal_power")


# ---------------------------------------------------------------------------
# Real-data loader
# ---------------------------------------------------------------------------

def _load_caiso_csv(csv_path: Path) -> pd.DataFrame:
    """Read a CAISO generation CSV and normalise to the pipeline schema.

    Returns a frame with columns: ``timestamp``, ``solar_mw``, ``wind_mw``.
    Price is NOT attached here; the caller merges it from the LMP file.
    """
    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(df["interval_start_utc"], utc=True)
    df = df.rename(columns={"solar": "solar_mw", "wind": "wind_mw"})

    keep = ["timestamp", "solar_mw", "wind_mw"]
    df = df[keep].copy()

    df["solar_mw"] = df["solar_mw"].clip(lower=0)
    df["wind_mw"] = df["wind_mw"].clip(lower=0)

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )
    df = df.set_index("timestamp").ffill(limit=3).dropna().reset_index()
    return df


def _merge_lmp(
    gen_df: pd.DataFrame,
    lmp_path: Optional[Path],
    node: Optional[str],
) -> pd.DataFrame:
    """Merge real LMP prices onto the generation frame.

    If ``lmp_path`` is None or the file is missing, falls back to the
    synthetic duck-curve price series with a warning.
    """
    if lmp_path is not None and lmp_path.exists():
        lmp = load_lmp_csv(lmp_path, node=node)
        # lmp has columns: timestamp (UTC), price_usd_per_mwh
        merged = pd.merge(gen_df, lmp, on="timestamp", how="left")
        missing_price = merged["price_usd_per_mwh"].isna().sum()
        if missing_price > 0:
            log.warning(
                "%d generation rows have no matching LMP timestamp; "
                "forward-filling up to 3 hours then dropping remainder.",
                missing_price,
            )
            merged = (
                merged.set_index("timestamp")
                .ffill(limit=3)
                .dropna(subset=["price_usd_per_mwh"])
                .reset_index()
            )
        log.info(
            "LMP merge complete: %d rows retained, "
            "price range $%.2f – $%.2f/MWh",
            len(merged),
            merged["price_usd_per_mwh"].min(),
            merged["price_usd_per_mwh"].max(),
        )
        return merged
    else:
        if lmp_path is not None:
            log.warning(
                "LMP file not found at %s – falling back to synthetic price. "
                "Copy LMP_<year>_combined.csv to data/raw/ to use real prices.",
                lmp_path,
            )
        else:
            log.warning(
                "No LMP file path configured (paths.lmp_file) – "
                "falling back to synthetic price."
            )
        gen_df["price_usd_per_mwh"] = _synthesise_price(gen_df)
        return gen_df


# ---------------------------------------------------------------------------
# Synthetic price fallback (CI / mock only)
# ---------------------------------------------------------------------------

def _synthesise_price(df: pd.DataFrame) -> np.ndarray:
    """Return a plausible hourly LMP series correlated with the duck curve.

    Used ONLY as a fallback when the real LMP file is unavailable.
    """
    rng = np.random.default_rng(seed=99)
    local = df["timestamp"].dt.tz_convert("America/Los_Angeles")
    hour = local.dt.hour.to_numpy()
    solar = df["solar_mw"].to_numpy()

    base_price = 45.0
    duck_curve = 25 * np.clip(np.sin(np.pi * (hour - 16) / 8), 0, None)
    scarcity = rng.gamma(shape=1.2, scale=6.0, size=len(df))
    price = base_price + duck_curve + scarcity - 0.0015 * solar
    return np.clip(price, -20, 500)


# ---------------------------------------------------------------------------
# Mock / synthetic data (for unit-test / CI usage)
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
    lmp_file: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load CAISO hourly solar, wind, and LMP price data.

    Parameters
    ----------
    year : int
        Calendar year.
    node : str
        CAISO pricing node label used to filter the LMP file when it
        contains multiple nodes.
    use_mock : bool
        If True, return fully synthetic data (useful for CI / tests).
    raw_dir : str or Path, optional
        Directory that contains the raw CSV files.  Defaults to
        ``data/raw`` relative to the repo root.
    lmp_file : str or Path, optional
        Explicit path to the LMP CSV.  When None, the loader looks for
        ``LMP_<year>_combined.csv`` inside ``raw_dir``.

    Returns
    -------
    pandas.DataFrame
        Hourly frame with columns: ``timestamp`` (UTC), ``solar_mw``,
        ``wind_mw``, ``price_usd_per_mwh``.

    Raises
    ------
    FileNotFoundError
        If ``use_mock=False`` and no generation CSV is found in ``raw_dir``.
    """
    if use_mock:
        return _simulate_caiso(year=year)

    # ---- Resolve raw_dir --------------------------------------------------
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[3] / "data" / "raw"
    raw_dir = Path(raw_dir)

    # ---- Locate generation CSV --------------------------------------------
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
            f"No CAISO generation CSV found for year={year} in {raw_dir}.\n"
            f"Looked for: {[str(c) for c in candidates]}\n"
            "Copy your CSV there or set sources.caiso_use_mock=true in "
            "config/params.yaml to fall back to synthetic data."
        )

    # ---- Load generation --------------------------------------------------
    gen = _load_caiso_csv(csv_path)

    # ---- Resolve LMP file path --------------------------------------------
    if lmp_file is None:
        lmp_path: Optional[Path] = raw_dir / f"LMP_{year}_combined.csv"
    else:
        lmp_path = Path(lmp_file)

    # ---- Merge real LMP prices -------------------------------------------
    return _merge_lmp(gen, lmp_path, node=node)
