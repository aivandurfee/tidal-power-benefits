"""Scenario sensitivity analysis for the tidal value study.

Sweeps over combinations of turbine count and velocity scaling and
reports how annual energy and the value-weighted price change.  All
heavy lifting (velocity prep, power model, alignment, value metrics)
is delegated to the existing modules -- nothing is duplicated here.

Typical usage from ``main.py``::

    from src.analysis.sensitivity import (
        DEFAULT_SCENARIOS, run_sensitivity_analysis, plot_sensitivity,
    )

    table = run_sensitivity_analysis(
        base_noaa_df=noaa_raw, caiso_df=caiso, tidal_params=cfg["tidal"],
        config=cfg,
    )
    plot_sensitivity(table, out_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.value_metrics import value_factor, value_weighted_price
from src.models.tidal_model import compute_tidal_generation, prepare_velocity
from src.processing.align_time import align_datasets


log = logging.getLogger("tidal_power")


# ---------------------------------------------------------------------------
# Default scenarios
# ---------------------------------------------------------------------------
DEFAULT_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "small_farm",  "num_turbines": 100, "velocity_scale": 10000},
    {"name": "medium_farm", "num_turbines": 150, "velocity_scale": 10000},
    {"name": "large_farm",  "num_turbines": 250, "velocity_scale": 10000},
    {"name": "high_flow",   "num_turbines": 150, "velocity_scale": 15000},
]


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------
def _run_one_scenario(
    scenario: Dict[str, Any],
    base_noaa_df: pd.DataFrame,
    caiso_df: pd.DataFrame,
    tidal_params: Dict[str, Any],
    timezone: str,
) -> Dict[str, Any]:
    """Run a single scenario end-to-end and return a summary row."""
    params = dict(tidal_params)  # shallow copy; never mutate caller's dict
    params["num_turbines"] = int(scenario["num_turbines"])
    params["velocity_scale"] = float(scenario["velocity_scale"])

    noaa_scaled = prepare_velocity(base_noaa_df, params)
    tidal = compute_tidal_generation(noaa_scaled, params)
    merged = align_datasets({"tidal": tidal, "caiso": caiso_df}, timezone=timezone)

    total_mwh = float(merged["tidal_energy_mwh"].sum())
    mean_mw = float(merged["tidal_energy_mwh"].mean())
    vw_price = value_weighted_price(merged, "tidal_energy_mwh")
    vf = value_factor(merged, "tidal_energy_mwh")

    return {
        "scenario": scenario["name"],
        "num_turbines": params["num_turbines"],
        "velocity_scale": params["velocity_scale"],
        "total_energy_mwh": total_mwh,
        "mean_generation_mw": mean_mw,
        "value_weighted_price": vw_price,
        "value_factor": vf,
    }


def run_sensitivity_analysis(
    base_noaa_df: pd.DataFrame,
    caiso_df: pd.DataFrame,
    tidal_params: Dict[str, Any],
    config: Dict[str, Any],
    scenarios: Optional[Sequence[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Sweep scenarios and return a tidy summary DataFrame.

    Parameters
    ----------
    base_noaa_df : pandas.DataFrame
        The RAW NOAA frame (``velocity_mps`` is raw dh/dt, unscaled).
        Must come straight from ``fetch_noaa_currents`` -- do not
        pre-apply ``prepare_velocity``, the runner does it per scenario.
    caiso_df : pandas.DataFrame
        Cleaned CAISO frame (output of ``clean_caiso``).
    tidal_params : dict
        Baseline tidal parameters; each scenario overrides
        ``num_turbines`` and ``velocity_scale``.
    config : dict
        Full project config; only ``timezone`` is read.
    scenarios : sequence of dict, optional
        Scenarios to evaluate; defaults to :data:`DEFAULT_SCENARIOS`.

    Returns
    -------
    pandas.DataFrame
        One row per scenario with the columns documented in the
        module docstring.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    timezone = config.get("timezone", "America/Los_Angeles")

    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        log.info(
            "Sensitivity scenario: name=%s turbines=%s velocity_scale=%s",
            scenario["name"], scenario["num_turbines"], scenario["velocity_scale"],
        )
        rows.append(
            _run_one_scenario(scenario, base_noaa_df, caiso_df, tidal_params, timezone)
        )

    return pd.DataFrame(rows, columns=[
        "scenario", "num_turbines", "velocity_scale",
        "total_energy_mwh", "mean_generation_mw",
        "value_weighted_price", "value_factor",
    ])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_sensitivity(
    table: pd.DataFrame,
    savepath: str | Path,
) -> Path:
    """Produce a two-panel sensitivity figure and save to ``savepath``.

    Left panel  : bar chart of ``value_factor`` per scenario
    Right panel : scatter of ``total_energy_mwh`` vs ``value_weighted_price``
                  with each scenario labeled.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728",
              "#ff7f0e", "#17becf", "#8c564b"]
    ax.bar(table["scenario"], table["value_factor"],
           color=colors[: len(table)])
    ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("Value factor (gen-weighted LMP / mean LMP)")
    ax.set_title("Value factor by scenario")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(
        table["total_energy_mwh"],
        table["value_weighted_price"],
        s=80, c=colors[: len(table)], edgecolor="black", zorder=3,
    )
    for _, row in table.iterrows():
        ax.annotate(
            row["scenario"],
            (row["total_energy_mwh"], row["value_weighted_price"]),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )
    ax.set_xlabel("Total tidal energy [MWh]")
    ax.set_ylabel("Value-weighted price [$/MWh]")
    ax.set_title("Scale vs value")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Tidal sensitivity analysis")
    fig.tight_layout()

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return savepath
