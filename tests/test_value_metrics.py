"""Unit tests for value metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.value_metrics import (
    summarize_resources,
    value_factor,
    value_weighted_price,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gen": [0.0, 10.0, 20.0, 5.0],
            "price_usd_per_mwh": [50.0, 100.0, 25.0, 75.0],
        }
    )


def test_value_weighted_price_matches_manual():
    df = _sample_df()
    expected = (0 * 50 + 10 * 100 + 20 * 25 + 5 * 75) / (0 + 10 + 20 + 5)
    assert value_weighted_price(df, "gen") == expected


def test_value_factor_is_ratio():
    df = _sample_df()
    mean_price = df["price_usd_per_mwh"].mean()
    vw = value_weighted_price(df, "gen")
    assert value_factor(df, "gen") == vw / mean_price


def test_zero_generation_returns_nan():
    df = pd.DataFrame({"gen": [0, 0, 0], "price_usd_per_mwh": [10, 20, 30]})
    assert np.isnan(value_weighted_price(df, "gen"))


def test_summarize_resources_shape():
    df = _sample_df().rename(columns={"gen": "solar_mw"})
    df["wind_mw"] = [1, 2, 3, 4]
    out = summarize_resources(df, ["solar_mw", "wind_mw"])
    assert list(out.index) == ["solar_mw", "wind_mw"]
    assert "value_weighted_price" in out.columns
