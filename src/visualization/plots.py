"""Plotting helpers.

All plotting functions return the Matplotlib ``Figure`` so callers can
either ``fig.savefig(...)`` or display in a notebook.  Styling is kept
minimal and default so team members can customize without fighting a
shared rcParams block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_generation_vs_price(
    df: pd.DataFrame,
    generation_cols: Iterable[str],
    price_col: str = "price_usd_per_mwh",
    savepath: Optional[str | Path] = None,
):
    """Scatter plot: hourly generation vs hourly price for each resource."""
    cols = list(generation_cols)
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), sharey=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.scatter(df[col], df[price_col], s=4, alpha=0.3)
        ax.set_xlabel(f"{col} [MW]")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"{price_col}")
    fig.suptitle("Hourly generation vs LMP")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_value_comparison(
    summary: pd.DataFrame,
    value_col: str = "value_weighted_price",
    savepath: Optional[str | Path] = None,
):
    """Bar chart comparing a value metric across resources.

    ``summary`` is expected to be the DataFrame returned by
    ``value_metrics.summarize_resources``.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    summary[value_col].plot.bar(ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title("Value comparison across resources")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_hourly_profile(
    df: pd.DataFrame,
    cols: Iterable[str],
    hour_col: str = "hour",
    savepath: Optional[str | Path] = None,
):
    """Average diurnal profile of each column grouped by local hour."""
    profile = df.groupby(hour_col)[list(cols)].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    profile.plot(ax=ax)
    ax.set_xlabel("Local hour of day")
    ax.set_ylabel("Average MW")
    ax.set_title("Mean diurnal profile")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
