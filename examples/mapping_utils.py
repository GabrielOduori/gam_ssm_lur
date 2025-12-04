"""
Small plotting helpers for NO₂ map visualizations used in experiments.

These consolidate the ad-hoc snippets for:
- Triptych scatter maps (Sentinel / LUR / EPA)
- Daily LUR maps across a week
- Interactive fused NO₂ map (Plotly)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def plot_triptych_no2(
    df: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    grid_col: str = "grid_id",
    sentinel_col: str = "sentinel_no2",
    lur_col: str = "lur_no2",
    epa_col: str = "epa_no2",
    outfile: Path | str = "no2_distributions_sentinel_lur_epa.png",
) -> None:
    """Scatter maps for Sentinel / LUR / EPA NO₂ on a common grid."""
    import matplotlib.pyplot as plt

    required = {grid_col, lat_col, lon_col, sentinel_col, lur_col, epa_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df_no2_avg = df.groupby(grid_col, as_index=False)[[sentinel_col, lur_col, epa_col]].mean()
    df_grouped = (
        df[[grid_col, lat_col, lon_col]].drop_duplicates().merge(df_no2_avg, on=grid_col)
    )

    plt.figure(figsize=(18, 6))
    for i, (title, col) in enumerate(
        [("Sentinel NO₂", sentinel_col), ("LUR NO₂", lur_col), ("EPA NO₂", epa_col)], start=1
    ):
        plt.subplot(1, 3, i)
        sc = plt.scatter(
            df_grouped[lon_col],
            df_grouped[lat_col],
            c=df_grouped[col],
            cmap="viridis",
            s=20,
        )
        plt.colorbar(sc, label=title.replace("₂", "2"))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title)
        plt.grid(True)

    plt.tight_layout()
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_weekly_lur_maps(
    df: pd.DataFrame,
    timestamp_col: str = "formatted_hour",
    value_col: str = "lur_no2",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    start_date: str = "2023-06-01",
    end_date: str = "2023-06-07",
    outfile: Path | str = "daily_no2_maps_week.png",
) -> None:
    """Daily averaged LUR NO₂ maps for a given date range (inclusive start, inclusive end)."""
    import matplotlib.pyplot as plt

    if timestamp_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"DataFrame must include {timestamp_col!r} and {value_col!r}")

    df_local = df.copy()
    df_local[timestamp_col] = pd.to_datetime(df_local[timestamp_col])
    dates = pd.date_range(start_date, end_date)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    lon_min, lon_max = df_local[lon_col].min(), df_local[lon_col].max()
    lat_min, lat_max = df_local[lat_col].min(), df_local[lat_col].max()

    sc = None
    for i, day in enumerate(dates):
        ax = axes[i]
        df_day = df_local[df_local[timestamp_col].dt.date == day.date()]
        df_day_grouped = df_day.groupby([lat_col, lon_col], as_index=False)[value_col].mean()

        sc = ax.scatter(
            df_day_grouped[lon_col],
            df_day_grouped[lat_col],
            c=df_day_grouped[value_col],
            cmap="viridis",
            s=18,
        )
        ax.set_title(day.strftime("%d %b %Y"))
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)

    cbar = fig.colorbar(sc, ax=axes[: len(dates)], orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("LUR NO₂")

    for j in range(len(dates), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Daily LUR NO₂ Maps ({dates[0].date()} – {dates[-1].date()})", fontsize=16, y=0.92)
    fig.tight_layout()

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_fused_no2_map(
    df: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    value_col: str = "fused_no2",
    outfile_html: Optional[Path | str] = None,
    mapbox_style: str = "carto-positron",
    zoom: float = 10.0,
) -> "plotly.graph_objs._figure.Figure":
    """Interactive fused NO₂ map using Plotly Mapbox."""
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("plotly is required for plot_fused_no2_map") from exc

    df_grouped = (
        df.dropna(subset=[lat_col, lon_col, value_col])
        .groupby([lat_col, lon_col], as_index=False)[value_col]
        .mean()
    )

    fig = px.scatter_mapbox(
        df_grouped,
        lat=lat_col,
        lon=lon_col,
        color=value_col,
        color_continuous_scale="RdBu",
        title="Fused NO₂ Concentration Across Locations",
        labels={value_col: "Fused NO₂ (µg/m³)"},
    )

    fig.update_layout(
        title_x=0.5,
        coloraxis_colorbar=dict(title="Fused NO₂ (µg/m³)"),
        mapbox_style=mapbox_style,
        mapbox_center={
            "lat": float(df_grouped[lat_col].mean()),
            "lon": float(df_grouped[lon_col].mean()),
        },
        mapbox_zoom=zoom,
        width=800,
        height=600,
    )

    if outfile_html:
        outfile_html = Path(outfile_html)
        outfile_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(outfile_html)

    return fig
