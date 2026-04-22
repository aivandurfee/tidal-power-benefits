from __future__ import annotations
from typing import Dict
import pandas as pd

def to_hourly_utc(df, timestamp_col="timestamp"):
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True)
    out = out.drop_duplicates(subset=timestamp_col).sort_values(timestamp_col).set_index(timestamp_col)
    hourly = out.select_dtypes(include="number").resample("1h").mean()
    return hourly

def align_datasets(frames: Dict[str, pd.DataFrame], timezone="America/Los_Angeles",
                   price_col="price_usd_per_mwh"):
    hourly_frames = [to_hourly_utc(f) for f in frames.values()]
    merged = pd.concat(hourly_frames, axis=1, join="inner").sort_index()
    # Drop hours where price is missing (CAISO CSV gaps outside its date range)
    if price_col in merged.columns:
        merged = merged.dropna(subset=[price_col])
    local = merged.index.tz_convert(timezone)
    merged["local_time"] = local
    merged["hour"] = local.hour
    merged["month"] = local.month
    merged["dayofweek"] = local.dayofweek
    return merged
