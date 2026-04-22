"""Simple tidal turbine power model.

Physical relation (kinetic-energy flux through rotor):

    P = 0.5 * rho * A * v^3 * Cp

where
    rho : seawater density [kg/m^3]
    A   : swept rotor area [m^2]
    v   : tidal current speed [m/s]
    Cp  : power coefficient (<= Betz limit ~0.59, typically 0.3-0.45)

The output is per-hour energy (MWh) because the input series is hourly
and average power (MW) over a one-hour window equals energy (MWh).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def instantaneous_power_mw(
    v: np.ndarray,
    rho: float,
    area: float,
    cp: float,
    cut_in: float = 0.0,
    cut_out: float = np.inf,
    rated_mw: Optional[float] = None,
) -> np.ndarray:
    """Return instantaneous turbine power in MW for each current speed.

    Parameters
    ----------
    v : numpy.ndarray
        Current speed in m/s.
    rho, area, cp : float
        Physical parameters. See module docstring.
    cut_in : float
        Below this speed the turbine produces 0 MW.
    cut_out : float
        Above this speed the turbine is curtailed (power set to 0).
    rated_mw : float, optional
        If given, power is capped at this rated output.
    """
    v = np.asarray(v, dtype=float)
    power_w = 0.5 * rho * area * np.power(v, 3) * cp
    power_mw = power_w / 1e6
    power_mw = np.where((v < cut_in) | (v > cut_out), 0.0, power_mw)
    if rated_mw is not None:
        power_mw = np.minimum(power_mw, rated_mw)
    return power_mw


def compute_tidal_generation(
    currents: pd.DataFrame,
    tidal_params: dict,
    speed_col: str = "current_speed_mps",
) -> pd.DataFrame:
    """Convert an hourly tidal-current series into generation (MWh).

    Parameters
    ----------
    currents : pandas.DataFrame
        Must contain a ``timestamp`` column and a current-speed column.
    tidal_params : dict
        Dict with keys ``rho``, ``area``, ``cp`` and optionally
        ``cut_in_speed``, ``cut_out_speed``, ``rated_power_mw``.
    speed_col : str
        Name of the current-speed column in ``currents``.

    Returns
    -------
    pandas.DataFrame
        Input frame plus ``tidal_power_mw``, ``tidal_energy_mwh``, and
        ``tidal_capacity_factor`` columns.
    """
    rated = tidal_params.get("rated_power_mw")
    power_mw = instantaneous_power_mw(
        v=currents[speed_col].to_numpy(),
        rho=tidal_params["rho"],
        area=tidal_params["area"],
        cp=tidal_params["cp"],
        cut_in=tidal_params.get("cut_in_speed", 0.0),
        cut_out=tidal_params.get("cut_out_speed", np.inf),
        rated_mw=rated,
    )

    out = currents.copy()
    out["tidal_power_mw"] = power_mw
    out["tidal_energy_mwh"] = power_mw  # 1-hour interval
    if rated:
        out["tidal_capacity_factor"] = power_mw / rated
    else:
        out["tidal_capacity_factor"] = np.nan
    return out
