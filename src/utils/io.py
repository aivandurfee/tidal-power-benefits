"""Small IO helpers for writing processed outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV or Parquet based on file extension.

    Parent directories are created if missing.  Returns the output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path)
    elif suffix in (".csv", ".txt"):
        df.to_csv(path, index=True)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
    return path
