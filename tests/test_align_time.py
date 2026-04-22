"""Tests for the time alignment helpers."""

from __future__ import annotations

import pandas as pd

from src.processing.align_time import align_datasets, to_hourly_utc


def test_to_hourly_utc_resamples_subhourly():
    idx = pd.date_range("2023-01-01", periods=4, freq="30min", tz="UTC")
    df = pd.DataFrame({"timestamp": idx, "x": [1.0, 2.0, 3.0, 4.0]})
    out = to_hourly_utc(df)
    assert out.index.freqstr in ("h", "H")
    assert len(out) == 2
    assert out["x"].iloc[0] == 1.5


def test_align_datasets_inner_join():
    a = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC"),
            "a": [1.0, 2.0, 3.0],
        }
    )
    b = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01 01:00", periods=3, freq="h", tz="UTC"),
            "b": [10.0, 20.0, 30.0],
        }
    )
    merged = align_datasets({"a": a, "b": b})
    assert list(merged.columns[:2]) == ["a", "b"]
    assert len(merged) == 2
    assert "hour" in merged.columns
