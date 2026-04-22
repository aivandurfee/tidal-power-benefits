"""NOAA tidal data pipeline.

Pulls hourly water-level observations from the NOAA Tides & Currents
``datagetter`` API, caches them under ``data/raw``, reindexes to a
strict hourly UTC timeline, interpolates gaps, and derives a velocity
proxy from dh/dt.

The real endpoint (used when ``use_mock=False``)::

    https://api.tidesandcurrents.noaa.gov/api/prod/datagetter
        ?product=water_level
        &interval=h
        &station={station_id}
        &begin_date={YYYYMMDD}
        &end_date={YYYYMMDD}
        &units=metric
        &time_zone=gmt
        &format=json

Output schema (DataFrame indexed by ``datetime`` in UTC):

    water_level   -- meters above MLLW
    velocity_mps  -- signed dh/dt, smoothed (m/s)

No third-party HTTP libraries are required; we use ``urllib`` from
the standard library plus pandas + numpy.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


NOAA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


# ---------------------------------------------------------------------------
# Mock generator (kept so the rest of the pipeline runs offline)
# ---------------------------------------------------------------------------
def _simulate_water_level(year: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic hourly water levels for ``year``.

    Uses the two dominant tidal constituents (M2 ~12.42 h, K1 ~23.93 h)
    plus a spring-neap envelope.  Returned frame is indexed by a
    tz-aware UTC ``DatetimeIndex`` named ``datetime`` with a single
    ``water_level`` column (meters).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year}-12-31 23:00",
        freq="h",
        tz="UTC",
        name="datetime",
    )
    hours = np.arange(len(idx))
    m2 = 1.5 * np.sin(2 * np.pi * hours / 12.42)
    k1 = 0.5 * np.sin(2 * np.pi * hours / 23.93 + 0.4)
    spring_neap = 0.3 * np.sin(2 * np.pi * hours / (14.77 * 24))
    noise = rng.normal(0.0, 0.05, size=len(idx))
    water_level = m2 + k1 + spring_neap + noise
    return pd.DataFrame({"water_level": water_level}, index=idx)


# ---------------------------------------------------------------------------
# Real NOAA API access
# ---------------------------------------------------------------------------
def _fetch_noaa_chunk(
    station_id: str,
    begin: pd.Timestamp,
    end: pd.Timestamp,
    product: str = "water_level",
) -> pd.DataFrame:
    """Fetch one chunk of hourly NOAA water-level data (<= 31 days)."""
    params = {
        "product": product,
        "application": "tidal_power_benefits",
        "station": station_id,
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "interval": "h",
        "datum": "MLLW",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
    }
    url = f"{NOAA_URL}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if "error" in payload:
        raise RuntimeError(f"NOAA API error for {begin.date()}..{end.date()}: "
                           f"{payload['error'].get('message')}")
    records = payload.get("data", [])
    if not records:
        return pd.DataFrame(columns=["datetime", "water_level"])

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["t"], utc=True)
    df["water_level"] = pd.to_numeric(df["v"], errors="coerce")
    return df[["datetime", "water_level"]]


def _fetch_noaa_year(station_id: str, year: int, product: str = "water_level") -> pd.DataFrame:
    """Fetch the full year of hourly NOAA data by looping monthly chunks."""
    chunks: List[pd.DataFrame] = []
    months = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    for start in months:
        end = (start + pd.offsets.MonthEnd(1)).normalize()
        chunks.append(_fetch_noaa_chunk(station_id, start, end, product=product))
    if not chunks:
        raise RuntimeError("No data returned from NOAA API.")
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def _to_clean_hourly(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Force the frame onto a strict hourly UTC index for ``year``.

    - Drops duplicate timestamps (keeps the mean).
    - Reindexes to a full hourly range (8760 or 8784 rows).
    - Linearly interpolates gaps, then back/forward fills edges.
    """
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out = (
        out.groupby("datetime", as_index=True)["water_level"]
        .mean()
        .to_frame()
        .sort_index()
    )

    full_idx = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year}-12-31 23:00",
        freq="h",
        tz="UTC",
        name="datetime",
    )
    out = out.reindex(full_idx)
    out["water_level"] = (
        out["water_level"].interpolate(method="linear").bfill().ffill()
    )
    return out


def _add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the RAW dh/dt velocity proxy to the frame.

        v_raw (m/s) = d(water_level)/dt      with dt = 3600 s

    This series is intentionally unscaled and unsmoothed: it is the
    pure NOAA-derived physical quantity.  Scaling
    (``tidal.velocity_scale``) and smoothing happen ONCE, downstream,
    in ``src/models/tidal_model.prepare_velocity``.  Keeping that
    transformation out of the data-loader guarantees a single source
    of truth for the ``velocity_mps`` column.
    """
    out = df.copy()
    raw_velocity = out["water_level"].diff() / 3600.0  # m/s
    out["velocity_mps"] = raw_velocity.bfill().ffill()
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def fetch_noaa_currents(
    year: int,
    station_id: str = "9414290",
    use_mock: bool = False,
    raw_dir: str | Path | None = None,
    product: str = "water_level",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a clean hourly water-level + velocity series for ``year``.

    Parameters
    ----------
    year : int
        Calendar year (e.g. 2024).  The full year is returned, including
        leap-day hours when applicable.
    station_id : str
        NOAA Tides & Currents station id (default 9414290 = San
        Francisco).
    use_mock : bool
        When ``True``, a deterministic synthetic water-level series is
        used instead of calling the NOAA API.  Useful for offline CI
        and for keeping the rest of the pipeline runnable.
    raw_dir : str or Path, optional
        Directory used to cache the raw hourly water-level frame as
        ``noaa_{station_id}_{year}.parquet``.
    product : str
        NOAA product name (default ``water_level``).
    force_refresh : bool
        If True, bypass the on-disk cache and re-query the API.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``datetime`` (tz-aware UTC, strictly hourly) with
        columns ``water_level`` (m) and ``velocity_mps`` (m/s).
    """
    cache_path: Path | None = None
    if raw_dir is not None:
        cache_path = Path(raw_dir) / f"noaa_{station_id}_{year}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    raw: pd.DataFrame | None = None
    if cache_path is not None and cache_path.exists() and not force_refresh:
        raw = pd.read_parquet(cache_path)
        if "datetime" in raw.columns:
            raw = raw.set_index("datetime")
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw.index.name = "datetime"

    if raw is None:
        if use_mock:
            raw = _simulate_water_level(year=year)
        else:
            api_df = _fetch_noaa_year(station_id, year, product=product)
            raw = _to_clean_hourly(api_df, year=year)

        if cache_path is not None:
            raw.reset_index().to_parquet(cache_path, index=False)

    raw = _to_clean_hourly(raw.reset_index(), year=year)

    out = _add_velocity(raw)

    _validate(out, year=year)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(df: pd.DataFrame, year: int) -> None:
    """Assert the hourly-index invariants required by the pipeline."""
    expected_rows = 8784 if pd.Timestamp(f"{year}-12-31").is_leap_year else 8760
    assert len(df) == expected_rows, (
        f"Expected {expected_rows} hourly rows for {year}, got {len(df)}"
    )
    assert df.index.min() == pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
    assert df.index.max() == pd.Timestamp(f"{year}-12-31 23:00", tz="UTC")
    assert df.index.is_monotonic_increasing, "datetime index is not sorted"
    assert df.index.is_unique, "duplicate timestamps in datetime index"
    assert df["water_level"].notna().all(), "water_level has NaNs after interpolation"
    assert df["velocity_mps"].notna().all(), "velocity_mps has NaNs"
