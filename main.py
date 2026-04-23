"""End-to-end pipeline entry point.

Run from repo root::

    python main.py

Steps:
    1. Load config (``config/params.yaml``)
    2. Fetch NOAA water-level + velocity for the configured year
    3. Apply the tidal power model  -> datetime | tidal_energy_mwh | capacity_factor
    4. Fetch + clean CAISO solar/wind, merge real LMP prices
    5. Align all sources to an hourly UTC index
    6. Compute value metrics and print a summary table
    7. Persist the merged frame to ``data/processed/``
"""

from __future__ import annotations

import pandas as pd

from src.analysis.sensitivity import plot_sensitivity, run_sensitivity_analysis
from src.analysis.time_slices import build_slices
from src.analysis.value_metrics import summarize_resources
from src.data.fetch_caiso import fetch_caiso_data
from src.data.fetch_noaa import fetch_noaa_currents
from src.models.tidal_model import compute_tidal_generation, prepare_velocity
from src.processing.align_time import align_datasets
from src.processing.clean_caiso import clean_caiso
from src.utils.config import load_config, project_root
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger


log = get_logger()


def _log_tidal_validation(noaa: pd.DataFrame, tidal: pd.DataFrame, year: int) -> None:
    """Print the Part 6 validation block + sanity assertions."""
    expected_rows = 8784 if pd.Timestamp(f"{year}-12-31").is_leap_year else 8760

    log.info(
        "NOAA check | rows=%s  min=%s  max=%s",
        len(noaa), noaa.index.min(), noaa.index.max(),
    )
    log.info(
        "Tidal stats | mean |v|=%.4f m/s  max |v|=%.4f m/s  total=%.1f MWh",
        noaa["velocity_mps"].abs().mean(),
        noaa["velocity_mps"].abs().max(),
        float(tidal["tidal_energy_mwh"].sum()),
    )

    assert len(tidal) == expected_rows, f"tidal rows {len(tidal)} != {expected_rows}"
    assert (tidal["tidal_energy_mwh"] >= 0).all(), "negative generation"
    full = pd.date_range(
        f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h", tz="UTC"
    )
    assert tidal.index.equals(full), "tidal index does not match expected hourly range"


def run() -> pd.DataFrame:
    """Execute the full pipeline and return the merged hourly DataFrame."""
    cfg = load_config()
    root = project_root()
    raw_dir = root / cfg["paths"]["raw_dir"]
    out_path = root / cfg["paths"]["output_file"]

    year = int(cfg["year"])
    tz = cfg["timezone"]

    # Resolve the LMP file path (may be None if not set in config).
    lmp_file_cfg = cfg["paths"].get("lmp_file")
    lmp_file = root / lmp_file_cfg if lmp_file_cfg else None

    log.info("Fetching NOAA tidal data (year=%s, mock=%s, station=%s)",
             year, cfg["sources"]["noaa_use_mock"], cfg["sources"]["noaa_station_id"])
    noaa = fetch_noaa_currents(
        year=year,
        station_id=cfg["sources"]["noaa_station_id"],
        use_mock=cfg["sources"]["noaa_use_mock"],
        raw_dir=raw_dir,
        product=cfg["sources"].get("noaa_product", "water_level"),
    )

    noaa_raw = noaa                                              # keep the unscaled frame for sensitivity
    log.info("Preparing tidal velocity (scale + smooth)")
    noaa = prepare_velocity(noaa, tidal_params=cfg["tidal"])

    log.info("Running tidal power model")
    tidal = compute_tidal_generation(noaa, tidal_params=cfg["tidal"])
    _log_tidal_validation(noaa, tidal, year)

    log.info(
        "Fetching CAISO data (year=%s, mock=%s, lmp_file=%s)",
        year, cfg["sources"]["caiso_use_mock"], lmp_file,
    )
    caiso = fetch_caiso_data(
        year=year,
        node=cfg["sources"]["caiso_node"],
        use_mock=cfg["sources"]["caiso_use_mock"],
        raw_dir=raw_dir,
        lmp_file=lmp_file,
    )
    caiso = clean_caiso(caiso)

    log.info("Aligning datasets to hourly UTC")
    merged = align_datasets({"tidal": tidal, "caiso": caiso}, timezone=tz)

    log.info("Computing value metrics")
    gen_cols = ["tidal_energy_mwh", "solar_mw", "wind_mw"]
    summary = summarize_resources(merged, generation_cols=gen_cols)
    log.info("\n%s", summary.round(2).to_string())

    slices = build_slices(merged, cfg["time_slices"])
    for name, frame in slices.items():
        slice_summary = summarize_resources(frame, generation_cols=gen_cols)
        log.info(
            "Value-weighted price in slice '%s':\n%s",
            name,
            slice_summary[["value_weighted_price", "value_factor"]].round(2).to_string(),
        )

    log.info("Saving merged frame -> %s", out_path)
    save_dataframe(merged, out_path)

    log.info("Running sensitivity analysis")
    sensitivity = run_sensitivity_analysis(
        base_noaa_df=noaa_raw,
        caiso_df=caiso,
        tidal_params=cfg["tidal"],
        config=cfg,
    )
    log.info("\n%s", sensitivity.round(2).to_string(index=False))

    sens_csv = out_path.parent / "sensitivity_results.csv"
    sensitivity.to_csv(sens_csv, index=False)
    plot_path = out_path.parent / "sensitivity_plot.png"
    plot_sensitivity(sensitivity, plot_path)
    log.info("Sensitivity artefacts -> %s, %s", sens_csv, plot_path)

    return merged


if __name__ == "__main__":
    run()
