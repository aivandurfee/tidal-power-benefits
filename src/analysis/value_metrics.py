"""Marginal-value metrics for variable generation resources.

The headline metric is the **value-weighted average price**, a.k.a.
the generation-weighted LMP:

    VW = sum(price_t * gen_t) / sum(gen_t)

Comparing this number to the simple time-average LMP tells us whether
a resource tends to produce when prices are high (ratio > 1) or low
(ratio < 1).
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def value_weighted_price(
    df: pd.DataFrame,
    generation_col: str,
    price_col: str = "price_usd_per_mwh",
) -> float:
    """Return sum(price * gen) / sum(gen) for the given columns."""
    gen = df[generation_col].to_numpy()
    price = df[price_col].to_numpy()
    total = gen.sum()
    if total <= 0:
        return float("nan")
    return float(np.sum(price * gen) / total)


def value_factor(
    df: pd.DataFrame,
    generation_col: str,
    price_col: str = "price_usd_per_mwh",
) -> float:
    """Return value-weighted price divided by simple mean price."""
    mean_price = df[price_col].mean()
    if mean_price == 0 or np.isnan(mean_price):
        return float("nan")
    return value_weighted_price(df, generation_col, price_col) / mean_price


def total_revenue(
    df: pd.DataFrame,
    generation_col: str,
    price_col: str = "price_usd_per_mwh",
) -> float:
    """Return sum(price * generation), i.e. market revenue in $."""
    return float((df[price_col] * df[generation_col]).sum())


def summarize_resources(
    df: pd.DataFrame,
    generation_cols: Iterable[str],
    price_col: str = "price_usd_per_mwh",
) -> pd.DataFrame:
    """Build a summary table of value metrics across multiple resources.

    Returns a DataFrame with one row per ``generation_col`` and columns:
    ``mean_generation_mw``, ``total_energy_mwh``, ``value_weighted_price``,
    ``value_factor``, ``revenue_usd``.
    """
    rows: Dict[str, Dict[str, float]] = {}
    for col in generation_cols:
        rows[col] = {
            "mean_generation_mw": float(df[col].mean()),
            "total_energy_mwh": float(df[col].sum()),
            "value_weighted_price": value_weighted_price(df, col, price_col),
            "value_factor": value_factor(df, col, price_col),
            "revenue_usd": total_revenue(df, col, price_col),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
