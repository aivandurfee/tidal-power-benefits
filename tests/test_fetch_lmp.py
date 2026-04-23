"""Unit tests for the real LMP loader (fetch_lmp.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.fetch_lmp import load_lmp_csv


def _write_csv(tmp_path, filename: str, content: str) -> "Path":
    p = tmp_path / filename
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Layout A: OASIS single-node style
# ---------------------------------------------------------------------------
def test_layout_a_oasis_style(tmp_path):
    csv = _write_csv(tmp_path, "lmp.csv", (
        "interval_start_utc,LMP_PRC\n"
        "2024-01-01 00:00:00+00:00,45.5\n"
        "2024-01-01 01:00:00+00:00,50.1\n"
        "2024-01-01 02:00:00+00:00,38.9\n"
    ))
    df = load_lmp_csv(csv)
    assert list(df.columns) == ["timestamp", "price_usd_per_mwh"]
    assert len(df) == 3
    assert df["price_usd_per_mwh"].iloc[0] == pytest.approx(45.5)
    assert df["timestamp"].dt.tz is not None


# ---------------------------------------------------------------------------
# Layout B: multi-node, filter to specific node
# ---------------------------------------------------------------------------
def test_layout_b_filters_to_node(tmp_path):
    csv = _write_csv(tmp_path, "lmp.csv", (
        "interval_start_utc,node,LMP_PRC\n"
        "2024-01-01 00:00:00+00:00,TH_NP15_GEN-APND,45.5\n"
        "2024-01-01 00:00:00+00:00,TH_SP15_GEN-APND,60.0\n"
        "2024-01-01 01:00:00+00:00,TH_NP15_GEN-APND,50.1\n"
        "2024-01-01 01:00:00+00:00,TH_SP15_GEN-APND,55.0\n"
    ))
    df = load_lmp_csv(csv, node="TH_NP15_GEN-APND")
    assert len(df) == 2
    assert df["price_usd_per_mwh"].iloc[0] == pytest.approx(45.5)


# ---------------------------------------------------------------------------
# Layout B: multi-node, no node filter → averages across nodes
# ---------------------------------------------------------------------------
def test_layout_b_averages_when_no_node_filter(tmp_path):
    csv = _write_csv(tmp_path, "lmp.csv", (
        "interval_start_utc,node,LMP_PRC\n"
        "2024-01-01 00:00:00+00:00,NODE_A,40.0\n"
        "2024-01-01 00:00:00+00:00,NODE_B,60.0\n"
    ))
    df = load_lmp_csv(csv, node=None)
    assert len(df) == 1
    assert df["price_usd_per_mwh"].iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Layout C: already-normalised column names
# ---------------------------------------------------------------------------
def test_layout_c_normalised_columns(tmp_path):
    csv = _write_csv(tmp_path, "lmp.csv", (
        "timestamp,price_usd_per_mwh\n"
        "2024-06-01 08:00:00+00:00,72.3\n"
        "2024-06-01 09:00:00+00:00,68.1\n"
    ))
    df = load_lmp_csv(csv)
    assert len(df) == 2
    assert df["price_usd_per_mwh"].iloc[1] == pytest.approx(68.1)


# ---------------------------------------------------------------------------
# Naive timestamps treated as UTC
# ---------------------------------------------------------------------------
def test_naive_timestamps_treated_as_utc(tmp_path):
    csv = _write_csv(tmp_path, "lmp.csv", (
        "interval_start_utc,LMP_PRC\n"
        "2024-03-10 02:00:00,55.0\n"
        "2024-03-10 03:00:00,60.0\n"
    ))
    df = load_lmp_csv(csv)
    assert str(df["timestamp"].dt.tz) in ("UTC", "datetime.timezone.utc")


# ---------------------------------------------------------------------------
# Missing file raises FileNotFoundError
# ---------------------------------------------------------------------------
def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="LMP CSV not found"):
        load_lmp_csv(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# Unrecognised price column raises ValueError
# ---------------------------------------------------------------------------
def test_unrecognised_price_column_raises(tmp_path):
    csv = _write_csv(tmp_path, "lmp.csv", (
        "interval_start_utc,weird_col\n"
        "2024-01-01 00:00:00+00:00,45.5\n"
    ))
    with pytest.raises(ValueError, match="price column"):
        load_lmp_csv(csv)


# ---------------------------------------------------------------------------
# Duplicate timestamps are averaged
# ---------------------------------------------------------------------------
def test_duplicate_timestamps_averaged(tmp_path):
    csv = _write_csv(tmp_path, "lmp.csv", (
        "interval_start_utc,LMP_PRC\n"
        "2024-01-01 00:00:00+00:00,40.0\n"
        "2024-01-01 00:00:00+00:00,60.0\n"
        "2024-01-01 01:00:00+00:00,50.0\n"
    ))
    df = load_lmp_csv(csv)
    assert len(df) == 2
    assert df["price_usd_per_mwh"].iloc[0] == pytest.approx(50.0)
