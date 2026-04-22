"""End-to-end pipeline entry point.

Run from repo root:

    python main.py

Steps:
    1. Load config (config/params.yaml)
    2. Fetch NOAA tidal currents and CAISO solar/wind/price
    3. Clean + align all sources to an hourly UTC index
    4. Apply the tidal power model
    5. Merge everything into one wide frame
    6. Compute value metrics and print a summary table
    7. Persist the merged frame to data/processed/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.time_slices import build_slices
from src.analysis.value_metrics import summarize_resources
from src.data.fetch_caiso import fetch_caiso_data
from src.data.fetch_noaa import fetch_noaa_currents
from src.models.tidal_model import compute_tidal_generation
from src.processing.align_time import align_datasets
from src.processing.clean_caiso import clean_caiso
from src.utils.config import load_config, project_root
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger


log = get_logger()


def run() -> pd.DataFrame:
    """Execute the full pipeline and return the merged hourly DataFrame."""
    cfg = load_config()
    root = project_root()
    raw_dir = root / cfg["paths"]["raw_dir"]
    out_path = root / cfg["paths"]["output_file"]

    year = cfg["year"]
    tz = cfg["timezone"]

    log.info("Fetching NOAA tidal currents (year=%s, mock=%s)",
             year, cfg["sources"]["noaa_use_mock"])
    noaa = fetch_noaa_currents(
        year=year,
        station_id=cfg["sources"]["noaa_station_id"],
        use_mock=cfg["sources"]["noaa_use_mock"],
        raw_dir=raw_dir,
    )

    log.info("Fetching CAISO data (year=%s, mock=%s)",
             year, cfg["sources"]["caiso_use_mock"])
    caiso = fetch_caiso_data(
        year=year,
        node=cfg["sources"]["caiso_node"],
        use_mock=cfg["sources"]["caiso_use_mock"],
        raw_dir=raw_dir,
    )
    caiso = clean_caiso(caiso)

    log.info("Running tidal power model")
    tidal = compute_tidal_generation(noaa, tidal_params=cfg["tidal"])

    log.info("Aligning datasets to hourly UTC")
    merged = align_datasets({"tidal": tidal, "caiso": caiso}, timezone=tz)

    log.info("Computing value metrics")
    gen_cols = ["tidal_energy_mwh", "solar_mw", "wind_mw"]
    summary = summarize_resources(merged, generation_cols=gen_cols)
    log.info("\n%s", summary.round(2).to_string())

    slices = build_slices(merged, cfg["time_slices"])
    for name, frame in slices.items():
        slice_summary = summarize_resources(frame, generation_cols=gen_cols)
        log.info("Value-weighted price in slice '%s':\n%s",
                 name,
                 slice_summary[["value_weighted_price", "value_factor"]].round(2).to_string())

    log.info("Saving merged frame -> %s", out_path)
    save_dataframe(merged, out_path)
    return merged


if __name__ == "__main__":
    run()
