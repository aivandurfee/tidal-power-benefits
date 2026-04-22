"""Tests for the scenario sensitivity runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.sensitivity import (
    DEFAULT_SCENARIOS,
    plot_sensitivity,
    run_sensitivity_analysis,
)


def _synthetic_inputs(n_hours: int = 24 * 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small raw-NOAA frame and matching CAISO frame for tests."""
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC",
                        name="datetime")
    # Raw dh/dt in the ~1e-4 m/s range, like real NOAA data.
    rng = np.random.default_rng(0)
    v_raw = np.abs(np.sin(np.arange(n_hours) * 2 * np.pi / 12.42)) * 1.5e-4
    v_raw += rng.normal(0, 1e-5, size=n_hours)
    noaa = pd.DataFrame({"water_level": np.cumsum(v_raw), "velocity_mps": v_raw},
                        index=idx)

    caiso = pd.DataFrame({
        "timestamp": idx,
        "solar_mw": np.clip(rng.normal(5000, 1500, n_hours), 0, None),
        "wind_mw": np.clip(rng.normal(3000, 800, n_hours), 0, None),
        "price_usd_per_mwh": np.clip(rng.normal(55, 20, n_hours), 0, None),
    })
    return noaa, caiso


def test_default_scenarios_produce_row_per_scenario():
    noaa, caiso = _synthetic_inputs()
    base_params = {
        "rho": 1025, "cp": 0.4, "area": 300, "rated_mw": 2.0,
        "cut_in": 0.7, "cut_out": 3.0,
        "num_turbines": 150, "velocity_scale": 10000,
    }

    table = run_sensitivity_analysis(
        base_noaa_df=noaa, caiso_df=caiso,
        tidal_params=base_params, config={"timezone": "UTC"},
    )

    assert len(table) == len(DEFAULT_SCENARIOS)
    assert list(table.columns) == [
        "scenario", "num_turbines", "velocity_scale",
        "total_energy_mwh", "mean_generation_mw",
        "value_weighted_price", "value_factor",
    ]
    assert set(table["scenario"]) == {s["name"] for s in DEFAULT_SCENARIOS}


def test_more_turbines_give_more_energy():
    noaa, caiso = _synthetic_inputs()
    base_params = {
        "rho": 1025, "cp": 0.4, "area": 300, "rated_mw": 2.0,
        "cut_in": 0.7, "cut_out": 3.0,
        "num_turbines": 150, "velocity_scale": 10000,
    }

    table = run_sensitivity_analysis(
        base_noaa_df=noaa, caiso_df=caiso,
        tidal_params=base_params, config={"timezone": "UTC"},
    )
    small = table.loc[table["scenario"] == "small_farm", "total_energy_mwh"].iloc[0]
    large = table.loc[table["scenario"] == "large_farm", "total_energy_mwh"].iloc[0]
    assert large == pytest.approx(small * 250 / 100, rel=1e-3)


def test_runner_does_not_mutate_base_params():
    noaa, caiso = _synthetic_inputs()
    base_params = {
        "rho": 1025, "cp": 0.4, "area": 300, "rated_mw": 2.0,
        "cut_in": 0.7, "cut_out": 3.0,
        "num_turbines": 150, "velocity_scale": 10000,
    }
    before = dict(base_params)

    run_sensitivity_analysis(
        base_noaa_df=noaa, caiso_df=caiso,
        tidal_params=base_params, config={"timezone": "UTC"},
    )
    assert base_params == before


def test_plot_sensitivity_writes_file(tmp_path):
    noaa, caiso = _synthetic_inputs()
    base_params = {
        "rho": 1025, "cp": 0.4, "area": 300, "rated_mw": 2.0,
        "cut_in": 0.7, "cut_out": 3.0,
        "num_turbines": 150, "velocity_scale": 10000,
    }
    table = run_sensitivity_analysis(
        base_noaa_df=noaa, caiso_df=caiso,
        tidal_params=base_params, config={"timezone": "UTC"},
    )
    out = plot_sensitivity(table, tmp_path / "sens.png")
    assert out.exists()
    assert out.stat().st_size > 0
