"""Unit tests for the tidal power model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tidal_model import (
    compute_tidal_generation,
    instantaneous_power_mw,
    prepare_velocity,
)


# ---------------------------------------------------------------------------
# instantaneous_power_mw
# ---------------------------------------------------------------------------
def test_zero_speed_gives_zero_power():
    p = instantaneous_power_mw(np.array([0.0]), rho=1025, area=200, cp=0.35)
    assert p[0] == 0.0


def test_cubic_scaling():
    p1 = instantaneous_power_mw(np.array([1.0]), rho=1025, area=200, cp=0.35)
    p2 = instantaneous_power_mw(np.array([2.0]), rho=1025, area=200, cp=0.35)
    assert p2[0] == pytest.approx(8 * p1[0])


def test_negative_velocity_treated_as_magnitude():
    p_pos = instantaneous_power_mw(np.array([1.5]), rho=1025, area=200, cp=0.35)
    p_neg = instantaneous_power_mw(np.array([-1.5]), rho=1025, area=200, cp=0.35)
    assert p_pos[0] == pytest.approx(p_neg[0])


def test_rated_cap_applied():
    p = instantaneous_power_mw(
        np.array([5.0]), rho=1025, area=200, cp=0.35, rated_mw=1.0
    )
    assert p[0] == 1.0


def test_cut_in_and_cut_out():
    v = np.array([0.1, 1.5, 10.0])
    p = instantaneous_power_mw(v, rho=1025, area=200, cp=0.35, cut_in=0.5, cut_out=4.0)
    assert p[0] == 0.0 and p[2] == 0.0 and p[1] > 0.0


# ---------------------------------------------------------------------------
# prepare_velocity
# ---------------------------------------------------------------------------
def test_prepare_velocity_multiplies_by_scale():
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC", name="datetime")
    df = pd.DataFrame({"velocity_mps": [1e-4] * 6}, index=idx)

    out = prepare_velocity(df, {"velocity_scale": 10000.0})

    assert out["velocity_mps"].abs().mean() == pytest.approx(1.0)
    assert out["velocity_mps"].notna().all()


def test_prepare_velocity_requires_scale():
    df = pd.DataFrame({"velocity_mps": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="velocity_scale"):
        prepare_velocity(df, {})


def test_prepare_velocity_requires_column():
    df = pd.DataFrame({"water_level": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="velocity_mps"):
        prepare_velocity(df, {"velocity_scale": 1.0})


# ---------------------------------------------------------------------------
# compute_tidal_generation
# ---------------------------------------------------------------------------
def _farm_params(**overrides):
    """Helper: synthetic params large enough to pass sanity assertions."""
    base = {
        "rho": 1025, "cp": 0.4, "area": 300.0,
        "rated_mw": 2.0, "cut_in": 0.7, "cut_out": 3.0,
        "num_turbines": 100,
    }
    base.update(overrides)
    return base


def test_compute_tidal_generation_schema():
    # Long constant stream so total > 1000 MWh and nonzero_hours > 100.
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="datetime")
    df = pd.DataFrame({"velocity_mps": [1.5] * n}, index=idx)

    out = compute_tidal_generation(df, _farm_params())

    assert list(out.columns) == ["tidal_energy_mwh", "capacity_factor"]
    assert out.index.name == "datetime"
    assert (out["capacity_factor"] <= 1.0).all()
    assert (out["tidal_energy_mwh"] >= 0).all()


def test_compute_tidal_generation_accepts_datetime_column():
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
            "velocity_mps": [1.5] * 200,
        }
    )
    out = compute_tidal_generation(df, _farm_params())
    assert len(out) == 200
    assert out.index.name == "datetime"


def test_compute_tidal_generation_rejects_unscaled_velocity():
    """Raw dh/dt (~1e-4 m/s) must fail the guard instead of silently returning zeros."""
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC", name="datetime")
    df = pd.DataFrame({"velocity_mps": [1e-4] * 6}, index=idx)
    with pytest.raises(AssertionError, match="scaling not applied"):
        compute_tidal_generation(df, _farm_params())


def test_total_generation_too_small_raises():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="datetime")
    # Velocity below cut_in -> every hour is zero.
    df = pd.DataFrame({"velocity_mps": [0.3] * n}, index=idx)
    with pytest.raises(AssertionError, match="Too few generating hours|too small"):
        compute_tidal_generation(df, _farm_params())


def test_num_turbines_scales_linearly():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="datetime")
    df = pd.DataFrame({"velocity_mps": [1.5] * n}, index=idx)

    small = compute_tidal_generation(df, _farm_params(num_turbines=100))
    big = compute_tidal_generation(df, _farm_params(num_turbines=200))

    assert big["tidal_energy_mwh"].sum() == pytest.approx(
        2.0 * small["tidal_energy_mwh"].sum()
    )
