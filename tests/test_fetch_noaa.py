"""Smoke tests for the NOAA pipeline (mock path, no network)."""

from __future__ import annotations

import pandas as pd

from src.data.fetch_noaa import fetch_noaa_currents


def test_mock_fetch_has_leap_year_length(tmp_path):
    df = fetch_noaa_currents(
        year=2024, station_id="9414290", use_mock=True, raw_dir=tmp_path
    )
    assert len(df) == 8784
    assert df.index.min() == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert df.index.max() == pd.Timestamp("2024-12-31 23:00", tz="UTC")
    assert df.index.name == "datetime"
    assert {"water_level", "velocity_mps"} <= set(df.columns)
    assert df["water_level"].notna().all()
    assert df["velocity_mps"].notna().all()


def test_mock_fetch_non_leap_year(tmp_path):
    df = fetch_noaa_currents(
        year=2023, station_id="9414290", use_mock=True, raw_dir=tmp_path
    )
    assert len(df) == 8760


def test_mock_fetch_is_hourly_and_sorted(tmp_path):
    df = fetch_noaa_currents(
        year=2024, station_id="9414290", use_mock=True, raw_dir=tmp_path
    )
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    deltas = df.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1 and deltas[0] == pd.Timedelta("1h")
