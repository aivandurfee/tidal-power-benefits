"""Load real CAISO Locational Marginal Price (LMP) data from a CSV file.

Supports the most common CAISO LMP export layouts:

    Layout A – OASIS-style (single-node export)
        interval_start_utc, LMP_PRC  (or PRICE, price, lmp, …)

    Layout B – multi-node export with a node column
        interval_start_utc, node (or NODE, OPR_MNEMONICS, …), LMP_PRC

    Layout C – already-normalised
        timestamp (or interval_start_utc), price_usd_per_mwh

The loader applies this resolution order:
  1. Look for an exact ``price_usd_per_mwh`` column  → use it directly.
  2. Look for a column whose lower-cased name is in the LMP_PRICE_ALIASES
     set  → rename to ``price_usd_per_mwh``.
  3. If a node column is present, filter to the configured node first, then
     apply step 2.

Output
------
pandas.DataFrame with columns:
    timestamp           – tz-aware UTC DatetimeIndex
    price_usd_per_mwh   – $/MWh
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd


log = logging.getLogger("tidal_power")

# Lower-cased column name aliases that represent an LMP price column.
LMP_PRICE_ALIASES: set[str] = {
    "lmp_prc",
    "lmp",
    "price",
    "price_usd_per_mwh",
    "lmp_price",
    "mw",          # some older exports put $/MWh in a column named MW
    "value",
    "lmp_value",
    "prc",
}

# Lower-cased column name aliases for the timestamp.
TIMESTAMP_ALIASES: set[str] = {
    "interval_start_utc",
    "interval_start",
    "datetime",
    "timestamp",
    "date_time",
    "starttime",
    "start_time",
    "intervalstarttime_gmt",
}

# Lower-cased column name aliases for the node / location identifier.
NODE_ALIASES: set[str] = {
    "node",
    "node_id",
    "opr_mnemonics",
    "resource_name",
    "location",
    "pnode",
    "pnode_name",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_col(columns: list[str], aliases: set[str]) -> Optional[str]:
    """Return the first column whose lower-cased name is in ``aliases``."""
    for col in columns:
        if col.strip().lower() in aliases:
            return col
    return None


def _parse_timestamp_col(series: pd.Series) -> pd.Series:
    """Parse a timestamp series to tz-aware UTC, handling common formats."""
    parsed = pd.to_datetime(series, utc=False, errors="coerce")
    if parsed.dt.tz is None:
        # Treat naive timestamps as UTC (CAISO OASIS GMT exports).
        parsed = parsed.dt.tz_localize("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_lmp_csv(
    csv_path: str | Path,
    node: Optional[str] = None,
) -> pd.DataFrame:
    """Load an LMP CSV and return a normalised hourly price frame.

    Parameters
    ----------
    csv_path : str or Path
        Path to the LMP CSV file.
    node : str, optional
        If the file contains multiple nodes (Layout B), filter to this node
        before averaging.  When ``None`` and multiple nodes are present, the
        loader averages across all nodes at each timestamp and emits a
        warning.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp`` (tz-aware UTC), ``price_usd_per_mwh``.
        Sorted ascending, duplicates averaged, no NaNs.

    Raises
    ------
    ValueError
        If no recognisable timestamp or price column can be found.
    FileNotFoundError
        If ``csv_path`` does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"LMP CSV not found: {csv_path}\n"
            "Copy your LMP file there or update paths.lmp_file in "
            "config/params.yaml."
        )

    log.info("Loading LMP data from %s", csv_path)
    raw = pd.read_csv(csv_path)
    log.info("  Raw shape: %s rows × %s cols", *raw.shape)
    log.info("  Columns: %s", list(raw.columns))

    cols = list(raw.columns)

    # ---- Locate timestamp column ------------------------------------------
    ts_col = _find_col(cols, TIMESTAMP_ALIASES)
    if ts_col is None:
        raise ValueError(
            f"Cannot find a timestamp column in {csv_path.name}.\n"
            f"Looked for (case-insensitive): {sorted(TIMESTAMP_ALIASES)}\n"
            f"Found columns: {cols}"
        )

    # ---- Locate price column ----------------------------------------------
    price_col = _find_col(cols, LMP_PRICE_ALIASES)
    if price_col is None:
        raise ValueError(
            f"Cannot find a price column in {csv_path.name}.\n"
            f"Looked for (case-insensitive): {sorted(LMP_PRICE_ALIASES)}\n"
            f"Found columns: {cols}"
        )

    # ---- Locate optional node column --------------------------------------
    node_col = _find_col(cols, NODE_ALIASES)

    # ---- Filter by node if applicable ------------------------------------
    if node_col is not None:
        unique_nodes = raw[node_col].dropna().unique()
        if len(unique_nodes) > 1:
            if node is not None:
                # Try exact match first, then substring match.
                mask = raw[node_col].str.upper() == node.upper()
                if mask.sum() == 0:
                    mask = raw[node_col].str.upper().str.contains(
                        node.upper(), regex=False, na=False
                    )
                if mask.sum() == 0:
                    log.warning(
                        "Node '%s' not found in column '%s'. "
                        "Available: %s. Averaging across all nodes.",
                        node, node_col, list(unique_nodes[:10]),
                    )
                else:
                    raw = raw[mask].copy()
                    log.info("  Filtered to node '%s' (%d rows)", node, len(raw))
            else:
                log.warning(
                    "LMP file contains %d nodes but no node filter specified. "
                    "Averaging LMP across all nodes. Set sources.caiso_node in "
                    "config/params.yaml to filter to a specific node.",
                    len(unique_nodes),
                )

    # ---- Build normalised frame ------------------------------------------
    df = pd.DataFrame()
    df["timestamp"] = _parse_timestamp_col(raw[ts_col])
    df["price_usd_per_mwh"] = pd.to_numeric(raw[price_col], errors="coerce")

    # Drop rows with unparseable values.
    before = len(df)
    df = df.dropna(subset=["timestamp", "price_usd_per_mwh"])
    dropped = before - len(df)
    if dropped:
        log.warning("  Dropped %d rows with NaN timestamp or price.", dropped)

    # Average duplicate timestamps (multi-node averages or duplicate rows).
    df = (
        df.groupby("timestamp", as_index=False)["price_usd_per_mwh"]
        .mean()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    log.info(
        "  LMP loaded: %d hourly rows | price range $%.2f – $%.2f/MWh",
        len(df),
        df["price_usd_per_mwh"].min(),
        df["price_usd_per_mwh"].max(),
    )
    return df
