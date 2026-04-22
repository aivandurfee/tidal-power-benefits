"""Unit tests for the tidal power model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tidal_model import compute_tidal_generation, instantaneous_power_mw


def test_zero_speed_gives_zero_power():
    p = instantaneous_power_mw(np.array([0.0]), rho=1025, area=200, cp=0.35)
    assert p[0] == 0.0


def test_cubic_scaling():
    p1 = instantaneous_power_mw(np.array([1.0]), rho=1025, area=200, cp=0.35)
    p2 = instantaneous_power_mw(np.array([2.0]), rho=1025, area=200, cp=0.35)
    assert p2[0] == pytest.approx(8 * p1[0])


def test_rated_cap_applied():
    p = instantaneous_power_mw(
        np.array([5.0]), rho=1025, area=200, cp=0.35, rated_mw=1.0
    )
    assert p[0] == 1.0


def test_cut_in_and_cut_out():
    v = np.array([0.1, 1.5, 10.0])
    p = instantaneous_power_mw(v, rho=1025, area=200, cp=0.35, cut_in=0.5, cut_out=4.0)
    assert p[0] == 0.0 and p[2] == 0.0 and p[1] > 0.0


def test_compute_tidal_generation_columns():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC"),
            "current_speed_mps": [0.0, 1.0, 2.0, 3.0, 0.5],
        }
    )
    params = {"rho": 1025, "area": 200, "cp": 0.35, "rated_power_mw": 1.5}
    out = compute_tidal_generation(df, params)
    for c in ("tidal_power_mw", "tidal_energy_mwh", "tidal_capacity_factor"):
        assert c in out.columns
    assert (out["tidal_power_mw"] <= 1.5).all()
