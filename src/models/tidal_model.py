"""Simple tidal-turbine power model.

Pipeline contract
-----------------
1. ``src/data/fetch_noaa.py`` returns a frame whose ``velocity_mps``
   column is the RAW ``d(water_level)/dt`` in m/s (order 1e-4).
2. ``prepare_velocity`` (this module) scales and smooths that series
   in place -- ONCE -- and overwrites ``velocity_mps``.
3. ``compute_tidal_generation`` (this module) reads the resulting
   ``velocity_mps`` column AS-IS and does not recompute or rescale
   velocity.  This keeps a single source of truth for the velocity
   used by the power equation.

Physical relation (kinetic-energy flux through rotor)::

    P = 0.5 * rho * A * v^3 * Cp

where
    rho : seawater density [kg/m^3]        (~1025)
    A   : swept rotor area [m^2]
    v   : tidal current speed [m/s]        (magnitude)
    Cp  : power coefficient                 (<= Betz ~0.59, typical 0.3-0.45)

Because the upstream series is hourly, average power (MW) over a
1-hour window equals hourly energy (MWh).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd


log = logging.getLogger("tidal_power")


# ---------------------------------------------------------------------------
# Preprocessing: velocity scaling + smoothing
# ---------------------------------------------------------------------------
def prepare_velocity(
    df: pd.DataFrame,
    tidal_params: Dict[str, float],
    window: int = 3,
) -> pd.DataFrame:
    """Apply ``velocity_scale`` then smooth, overwriting ``velocity_mps``.

    This is the ONLY place the scaling factor is applied.  Call it
    exactly once between ``fetch_noaa_currents`` and
    ``compute_tidal_generation``.

        v_scaled = tidal_params["velocity_scale"] * v_raw
        v_out    = rolling_mean(v_scaled, window=3)        # AFTER scaling

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a ``velocity_mps`` column (the raw dh/dt from
        NOAA).  Index / other columns are preserved.
    tidal_params : dict
        Must contain ``velocity_scale``.
    window : int
        Rolling-mean window in hours (default 3).

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with ``velocity_mps`` overwritten by the
        scaled-then-smoothed series.
    """
    if "velocity_mps" not in df.columns:
        raise ValueError("prepare_velocity: input frame missing 'velocity_mps'")
    if "velocity_scale" not in tidal_params:
        raise ValueError("prepare_velocity: tidal_params['velocity_scale'] is required")

    scale = float(tidal_params["velocity_scale"])
    out = df.copy()
    scaled = out["velocity_mps"].astype(float) * scale
    smoothed = scaled.rolling(window=window, center=True, min_periods=1).mean()
    out["velocity_mps"] = smoothed.bfill().ffill()

    abs_v = out["velocity_mps"].abs()
    log.info(
        "prepare_velocity: scale=%.1f, mean=%.3f m/s, max=%.3f m/s",
        scale, float(abs_v.mean()), float(abs_v.max()),
    )
    return out


# ---------------------------------------------------------------------------
# Pure physics
# ---------------------------------------------------------------------------
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

    Negative velocities are treated as magnitude (flood vs ebb both
    produce power).  Cut-in / cut-out speeds and an optional rated-MW
    cap are applied after the cubic scaling.
    """
    v = np.abs(np.asarray(v, dtype=float))
    power_w = 0.5 * rho * area * np.power(v, 3) * cp
    power_mw = power_w / 1e6
    power_mw = np.where((v < cut_in) | (v > cut_out), 0.0, power_mw)
    if rated_mw is not None:
        power_mw = np.minimum(power_mw, rated_mw)
    return power_mw


# ---------------------------------------------------------------------------
# Generation model
# ---------------------------------------------------------------------------
def compute_tidal_generation(
    currents: pd.DataFrame,
    tidal_params: Dict[str, float],
) -> pd.DataFrame:
    """Convert an already-scaled tidal-velocity series into generation.

    This function NEVER recomputes or rescales velocity; it trusts the
    ``velocity_mps`` column in ``currents`` as the single source of
    truth.  Call :func:`prepare_velocity` upstream if you are starting
    from a raw NOAA frame.

    Parameters
    ----------
    currents : pandas.DataFrame
        Frame with an hourly tz-aware ``datetime`` index (or a
        ``datetime`` column) and a ``velocity_mps`` column that is
        already in realistic m/s units (mean in the 0.1 - 2.5 range).
    tidal_params : dict
        Expected keys: ``rho``, ``cp``, ``area``, ``rated_mw``.
        Optional: ``cut_in``, ``cut_out``, ``num_turbines`` (default 1).
        Legacy names ``rated_power_mw`` / ``cut_in_speed`` /
        ``cut_out_speed`` are also accepted.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``datetime`` with exactly two columns:
        ``tidal_energy_mwh`` and ``capacity_factor`` (``[0, 1]``).
    """
    df = currents.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" not in df.columns:
            raise ValueError(
                "Input must have either a DatetimeIndex or a 'datetime' column."
            )
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
    df = df.sort_index()

    assert "velocity_mps" in df.columns, "velocity_mps missing"
    assert df["velocity_mps"].notna().all(), "velocity_mps has NaNs"
    assert df["velocity_mps"].abs().mean() > 0.01, (
        "Velocity too small -- scaling not applied. "
        "Did you forget to call prepare_velocity()?"
    )

    rated_per_turbine = tidal_params.get("rated_mw") or tidal_params.get("rated_power_mw")
    if rated_per_turbine is None:
        raise ValueError("tidal_params['rated_mw'] is required for capacity factor")
    rated_per_turbine = float(rated_per_turbine)
    num_turbines = int(tidal_params.get("num_turbines", 1))
    if num_turbines < 1:
        raise ValueError("tidal_params['num_turbines'] must be >= 1")
    farm_rated_mw = rated_per_turbine * num_turbines

    cut_in = float(tidal_params.get("cut_in", tidal_params.get("cut_in_speed", 0.0)))
    cut_out = float(tidal_params.get("cut_out", tidal_params.get("cut_out_speed", np.inf)))

    velocity = df["velocity_mps"]
    log.info(
        "Using velocity_mps: mean=%.3f, max=%.3f",
        float(velocity.abs().mean()),
        float(velocity.abs().max()),
    )

    per_turbine_mw = instantaneous_power_mw(
        v=velocity.to_numpy(),
        rho=float(tidal_params["rho"]),
        area=float(tidal_params["area"]),
        cp=float(tidal_params["cp"]),
        cut_in=cut_in,
        cut_out=cut_out,
        rated_mw=rated_per_turbine,
    )
    # Farm = N identical turbines; cap applied per turbine, then summed.
    farm_power_mw = per_turbine_mw * num_turbines

    out = pd.DataFrame(
        {
            "tidal_energy_mwh": farm_power_mw,          # 1-hour window -> MW == MWh
            "capacity_factor": farm_power_mw / farm_rated_mw,
        },
        index=df.index,
    )
    out.index.name = "datetime"

    # ---- Generation stats + safety checks -----------------------------------
    total_energy = float(out["tidal_energy_mwh"].sum())
    mean_power = float(out["tidal_energy_mwh"].mean())
    max_power = float(out["tidal_energy_mwh"].max())
    nonzero_hours = int((out["tidal_energy_mwh"] > 0).sum())

    log.info(
        "Tidal generation stats: total=%.1f MWh, mean=%.3f MW, max=%.3f MW "
        "(turbines=%d, farm_rated=%.1f MW, nonzero_hours=%d)",
        total_energy, mean_power, max_power,
        num_turbines, farm_rated_mw, nonzero_hours,
    )

    assert (out["tidal_energy_mwh"] >= 0).all(), "negative tidal generation"
    assert out[["tidal_energy_mwh", "capacity_factor"]].notna().all().all(), (
        "tidal generation has NaNs"
    )
    assert total_energy > 1000, (
        f"Tidal generation too small ({total_energy:.1f} MWh) -- check scaling "
        "(area, num_turbines, velocity_scale, cut_in)"
    )
    assert total_energy < 1e7, (
        f"Tidal generation unrealistically large ({total_energy:.1f} MWh) -- "
        "check num_turbines / area"
    )
    assert nonzero_hours > 100, (
        f"Too few generating hours ({nonzero_hours}) -- cut_in may be too high "
        "or velocities too small"
    )
    return out
