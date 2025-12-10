"""
Unified Visualization Module for GAM-SSM-LUR.

Provides reusable plotting functions for:
- Basic scatter maps (Sentinel / LUR / EPA)
- Gridded interpolated surface maps
- Temporal evolution visualizations
- Model comparison plots
- Uncertainty visualizations

This module eliminates code duplication across example scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RBFInterpolator, griddata
from scipy.ndimage import gaussian_filter


def adjust_colorbar_height(fig, ax_index, cbar_index):
    """
    Adjust a single colorbar to match the height of its corresponding axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axes and colorbar
    ax_index : int
        Index of the main axes in fig.axes
    cbar_index : int
        Index of the colorbar axes in fig.axes
    """
    ax = fig.axes[ax_index]
    cbar_ax = fig.axes[cbar_index]
    pos = ax.get_position()
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position([
        cbar_pos.x0,
        pos.y0,
        cbar_pos.width,
        pos.height
    ])


def adjust_colorbars_to_axes(fig, axes, colorbar_indices=None):
    """
    Adjust multiple colorbars to match the heights of their corresponding axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axes and colorbars
    axes : array-like
        Array of axes objects (can be 1D or 2D array from plt.subplots)
    colorbar_indices : list of int, optional
        Indices of colorbar axes in fig.axes. If None, assumes colorbars
        come immediately after the main axes in order.

    Examples
    --------
    # For a 1x3 subplot with 3 colorbars
    fig, axes = plt.subplots(1, 3)
    # ... create plots with colorbars ...
    plt.tight_layout()
    adjust_colorbars_to_axes(fig, axes)

    # For a 2x2 subplot with 4 colorbars
    fig, axes = plt.subplots(2, 2)
    # ... create plots with colorbars ...
    plt.tight_layout()
    adjust_colorbars_to_axes(fig, axes)
    """
    axes_flat = np.atleast_1d(axes).ravel()
    n_axes = len(axes_flat)

    if colorbar_indices is None:
        # Assume colorbars come after main axes
        colorbar_indices = list(range(n_axes, n_axes + n_axes))

    for i, cbar_idx in enumerate(colorbar_indices):
        if i < len(axes_flat) and cbar_idx < len(fig.axes):
            adjust_colorbar_height(fig, i, cbar_idx)


def adjust_shared_colorbar_height(fig, axes, cbar_index=-1):
    """
    Adjust a shared colorbar to span the full height of multiple axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axes and colorbar
    axes : array-like
        Array of axes objects that the colorbar should span
    cbar_index : int, optional
        Index of the colorbar axes in fig.axes (default: -1 for last axes)

    Examples
    --------
    # For a shared colorbar spanning all subplots
    fig, axes = plt.subplots(2, 3)
    # ... create plots ...
    fig.colorbar(im, ax=axes.ravel().tolist(), ...)
    plt.tight_layout()
    adjust_shared_colorbar_height(fig, axes)
    """
    cbar_ax = fig.axes[cbar_index]
    axes_flat = np.atleast_1d(axes).ravel()

    # Get position from first and last axes to span full height
    pos_first = axes_flat[0].get_position()
    pos_last = axes_flat[-1].get_position()
    cbar_pos = cbar_ax.get_position()

    cbar_ax.set_position([
        cbar_pos.x0,
        pos_last.y0,
        cbar_pos.width,
        pos_first.y1 - pos_last.y0
    ])


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

    # Adjust colorbar to match axes height
    active_axes = [axes[i] for i in range(len(dates))]
    adjust_shared_colorbar_height(fig, active_axes)

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


# =============================================================================
# GRIDDED/INTERPOLATED SURFACE MAPPING FUNCTIONS
# =============================================================================


def create_interpolation_grid(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create interpolation grid for spatial mapping.

    Parameters
    ----------
    lon_min, lon_max : float
        Longitude bounds
    lat_min, lat_max : float
        Latitude bounds
    resolution : int
        Grid resolution (number of points per axis)

    Returns
    -------
    Lon, Lat : np.ndarray
        Meshgrid of coordinates
    lon_grid, lat_grid : np.ndarray
        1D coordinate arrays
    """
    lon_grid = np.linspace(lon_min, lon_max, resolution)
    lat_grid = np.linspace(lat_min, lat_max, resolution)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)
    return Lon, Lat, lon_grid, lat_grid


def interpolate_to_grid(
    coordinates: np.ndarray,
    values: np.ndarray,
    Lon: np.ndarray,
    Lat: np.ndarray,
    method: str = 'cubic',
    smooth: bool = True,
    sigma: float = 1.5,
) -> np.ndarray:
    """Interpolate point data to regular grid.

    Parameters
    ----------
    coordinates : np.ndarray (n, 2)
        Point coordinates (lon, lat)
    values : np.ndarray (n,)
        Values at points
    Lon, Lat : np.ndarray
        Meshgrid of target coordinates
    method : str
        Interpolation method: 'linear', 'cubic', or 'rbf'
    smooth : bool
        Apply Gaussian smoothing after interpolation
    sigma : float
        Gaussian smoothing parameter (only if smooth=True)

    Returns
    -------
    grid : np.ndarray
        Interpolated values on grid
    """
    if method == 'rbf':
        # Radial basis function interpolation (smoother)
        rbf = RBFInterpolator(
            coordinates,
            values,
            kernel='thin_plate_spline',
            smoothing=0.1
        )
        grid_points = np.column_stack([Lon.ravel(), Lat.ravel()])
        grid = rbf(grid_points).reshape(Lon.shape)
    else:
        # scipy griddata
        grid = griddata(coordinates, values, (Lon, Lat), method=method)

        # Fill NaN values with nearest neighbor
        if np.any(np.isnan(grid)):
            grid_nearest = griddata(coordinates, values, (Lon, Lat), method='nearest')
            grid = np.where(np.isnan(grid), grid_nearest, grid)

    if smooth:
        grid = gaussian_filter(grid, sigma=sigma)

    return grid


def plot_gridded_surface(
    grid: np.ndarray,
    Lon: np.ndarray,
    Lat: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    colorbar_label: str = 'NO₂ (µg/m³)',
    show_coords: bool = True,
    contours: bool = False,
    n_contours: int = 10,
) -> plt.cm.ScalarMappable:
    """Plot a single gridded surface map.

    Parameters
    ----------
    grid : np.ndarray
        Gridded values
    Lon, Lat : np.ndarray
        Coordinate meshgrids
    ax : matplotlib axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    colorbar : bool
        Show colorbar
    colorbar_label : str
        Colorbar label
    show_coords : bool
        Show coordinate axis labels
    contours : bool
        Overlay contour lines
    n_contours : int
        Number of contour levels

    Returns
    -------
    im : matplotlib ScalarMappable
        The plotted surface
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot surface
    im = ax.pcolormesh(
        Lon, Lat, grid,
        cmap=cmap,
        shading='auto',
        vmin=vmin,
        vmax=vmax,
        rasterized=True
    )

    # Add contours
    if contours:
        levels = np.linspace(
            vmin if vmin is not None else grid.min(),
            vmax if vmax is not None else grid.max(),
            n_contours
        )
        cs = ax.contour(
            Lon, Lat, grid,
            levels=levels,
            colors='white',
            linewidths=0.5,
            alpha=0.5
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')

    # Colorbar - match height to map
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cbar.set_label(colorbar_label, fontsize=10)

    # Labels
    if show_coords:
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    ax.set_aspect('equal')

    return im


def create_gridded_comparison(
    coordinates: np.ndarray,
    observed: np.ndarray,
    baseline_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    output_path: Path,
    lon_bounds: Optional[Tuple[float, float]] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    resolution: int = 200,
    title_suffix: str = '',
) -> None:
    """Create three-panel gridded comparison figure.

    Parameters
    ----------
    coordinates : np.ndarray (n, 2)
        Point coordinates (lon, lat)
    observed : np.ndarray (n,)
        Observed values
    baseline_pred : np.ndarray (n,)
        Baseline model predictions
    hybrid_pred : np.ndarray (n,)
        Hybrid model predictions
    output_path : Path
        Output file path
    lon_bounds, lat_bounds : tuple, optional
        Coordinate bounds (min, max). If None, computed from data
    resolution : int
        Grid resolution
    title_suffix : str
        Additional text for main title
    """
    # Compute bounds if not provided
    if lon_bounds is None:
        lon_bounds = (coordinates[:, 0].min(), coordinates[:, 0].max())
    if lat_bounds is None:
        lat_bounds = (coordinates[:, 1].min(), coordinates[:, 1].max())

    # Create grid
    Lon, Lat, _, _ = create_interpolation_grid(
        lon_bounds[0], lon_bounds[1],
        lat_bounds[0], lat_bounds[1],
        resolution
    )

    # Interpolate all three
    obs_grid = interpolate_to_grid(coordinates, observed, Lon, Lat)
    baseline_grid = interpolate_to_grid(coordinates, baseline_pred, Lon, Lat)
    hybrid_grid = interpolate_to_grid(coordinates, hybrid_pred, Lon, Lat)

    # Common color limits
    vmin = min(obs_grid.min(), baseline_grid.min(), hybrid_grid.min())
    vmax = max(obs_grid.max(), baseline_grid.max(), hybrid_grid.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_gridded_surface(
        obs_grid, Lon, Lat, ax=axes[0],
        title='(a) Observed NO₂',
        vmin=vmin, vmax=vmax,
        contours=True, n_contours=8,
        colorbar=False
    )

    plot_gridded_surface(
        baseline_grid, Lon, Lat, ax=axes[1],
        title='(b) GAM-LUR Predicted',
        vmin=vmin, vmax=vmax,
        contours=True, n_contours=8,
        colorbar=False
    )

    im = plot_gridded_surface(
        hybrid_grid, Lon, Lat, ax=axes[2],
        title='(c) Hybrid GAM-SSM Predicted',
        vmin=vmin, vmax=vmax,
        contours=True, n_contours=8,
        colorbar=False
    )

    # Add shared colorbar with proper height
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, aspect=20, label='NO₂ (µg/m³)')

    fig.suptitle(
        f'NO₂ Concentration Surfaces{title_suffix}',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    adjust_shared_colorbar_height(fig, axes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_gridded_residual_map(
    coordinates: np.ndarray,
    observed: np.ndarray,
    baseline_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    output_path: Path,
    lon_bounds: Optional[Tuple[float, float]] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    resolution: int = 200,
) -> None:
    """Create gridded residual comparison with metrics panel.

    Parameters
    ----------
    coordinates : np.ndarray (n, 2)
        Point coordinates (lon, lat)
    observed : np.ndarray (n,)
        Observed values
    baseline_pred : np.ndarray (n,)
        Baseline predictions
    hybrid_pred : np.ndarray (n,)
        Hybrid predictions
    output_path : Path
        Output file path
    lon_bounds, lat_bounds : tuple, optional
        Coordinate bounds
    resolution : int
        Grid resolution
    """
    # Compute bounds if not provided
    if lon_bounds is None:
        lon_bounds = (coordinates[:, 0].min(), coordinates[:, 0].max())
    if lat_bounds is None:
        lat_bounds = (coordinates[:, 1].min(), coordinates[:, 1].max())

    # Create grid
    Lon, Lat, _, _ = create_interpolation_grid(
        lon_bounds[0], lon_bounds[1],
        lat_bounds[0], lat_bounds[1],
        resolution
    )

    # Interpolate
    obs_grid = interpolate_to_grid(coordinates, observed, Lon, Lat)
    baseline_grid = interpolate_to_grid(coordinates, baseline_pred, Lon, Lat)
    hybrid_grid = interpolate_to_grid(coordinates, hybrid_pred, Lon, Lat)

    # Residuals at points, then interpolate
    baseline_resid = observed - baseline_pred
    hybrid_resid = observed - hybrid_pred

    baseline_resid_grid = interpolate_to_grid(coordinates, baseline_resid, Lon, Lat)
    hybrid_resid_grid = interpolate_to_grid(coordinates, hybrid_resid, Lon, Lat)

    # Color limits
    vmin_conc = min(obs_grid.min(), baseline_grid.min(), hybrid_grid.min())
    vmax_conc = max(obs_grid.max(), baseline_grid.max(), hybrid_grid.max())
    vmax_res = max(np.abs(baseline_resid_grid).max(), np.abs(hybrid_resid_grid).max())

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.2, wspace=0.25)

    # Row 1: Concentrations
    ax1 = fig.add_subplot(gs[0, 0])
    plot_gridded_surface(
        obs_grid, Lon, Lat, ax=ax1,
        title='(a) Observed NO₂',
        vmin=vmin_conc, vmax=vmax_conc
    )

    ax2 = fig.add_subplot(gs[0, 1])
    plot_gridded_surface(
        baseline_grid, Lon, Lat, ax=ax2,
        title='(b) GAM-LUR Predicted',
        vmin=vmin_conc, vmax=vmax_conc
    )

    ax3 = fig.add_subplot(gs[0, 2])
    plot_gridded_surface(
        hybrid_grid, Lon, Lat, ax=ax3,
        title='(c) GAM-SSM Predicted',
        vmin=vmin_conc, vmax=vmax_conc
    )

    # Row 2: Residuals
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')

    # Compute metrics
    baseline_rmse = np.sqrt(np.mean(baseline_resid**2))
    hybrid_rmse = np.sqrt(np.mean(hybrid_resid**2))
    improvement = (baseline_rmse - hybrid_rmse) / baseline_rmse * 100

    metrics_text = (
        f"Model Performance\n"
        f"{'='*30}\n\n"
        f"GAM-LUR:\n"
        f"  RMSE: {baseline_rmse:.2f} µg/m³\n"
        f"  Mean residual: {baseline_resid.mean():.2f}\n"
        f"  SD residual: {baseline_resid.std():.2f}\n\n"
        f"GAM-SSM:\n"
        f"  RMSE: {hybrid_rmse:.2f} µg/m³\n"
        f"  Mean residual: {hybrid_resid.mean():.2f}\n"
        f"  SD residual: {hybrid_resid.std():.2f}\n\n"
        f"Improvement: {improvement:.1f}%"
    )
    ax4.text(
        0.1, 0.9, metrics_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    )
    ax4.set_title('(d) Performance Metrics', fontsize=12, fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 1])
    plot_gridded_surface(
        baseline_resid_grid, Lon, Lat, ax=ax5,
        title='(e) GAM-LUR Residuals',
        cmap='RdBu_r',
        vmin=-vmax_res, vmax=vmax_res,
        colorbar_label='Residual (µg/m³)'
    )

    ax6 = fig.add_subplot(gs[1, 2])
    plot_gridded_surface(
        hybrid_resid_grid, Lon, Lat, ax=ax6,
        title='(f) GAM-SSM Residuals',
        cmap='RdBu_r',
        vmin=-vmax_res, vmax=vmax_res,
        colorbar_label='Residual (µg/m³)'
    )

    fig.suptitle(
        'Spatial Comparison: GAM-LUR vs Hybrid GAM-SSM',
        fontsize=14, fontweight='bold', y=1.01
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_uncertainty_surface(
    coordinates: np.ndarray,
    predictions: np.ndarray,
    std_dev: np.ndarray,
    output_path: Path,
    lon_bounds: Optional[Tuple[float, float]] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    resolution: int = 200,
) -> None:
    """Create uncertainty surface visualization.

    Parameters
    ----------
    coordinates : np.ndarray (n, 2)
        Point coordinates
    predictions : np.ndarray (n,)
        Predicted mean values
    std_dev : np.ndarray (n,)
        Prediction standard deviations
    output_path : Path
        Output file path
    lon_bounds, lat_bounds : tuple, optional
        Coordinate bounds
    resolution : int
        Grid resolution
    """
    # Compute bounds if not provided
    if lon_bounds is None:
        lon_bounds = (coordinates[:, 0].min(), coordinates[:, 0].max())
    if lat_bounds is None:
        lat_bounds = (coordinates[:, 1].min(), coordinates[:, 1].max())

    # Create grid
    Lon, Lat, _, _ = create_interpolation_grid(
        lon_bounds[0], lon_bounds[1],
        lat_bounds[0], lat_bounds[1],
        resolution
    )

    # Interpolate mean and uncertainty
    mean_grid = interpolate_to_grid(coordinates, predictions, Lon, Lat)
    std_grid = interpolate_to_grid(coordinates, std_dev, Lon, Lat)

    # Coefficient of variation
    cv_grid = std_grid / mean_grid * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mean prediction
    im1 = plot_gridded_surface(
        mean_grid, Lon, Lat, ax=axes[0],
        title='(a) Predicted Mean NO₂',
        contours=True, n_contours=8,
        colorbar=False
    )

    # Prediction SD
    im2 = plot_gridded_surface(
        std_grid, Lon, Lat, ax=axes[1],
        title='(b) Prediction Uncertainty (SD)',
        cmap='YlOrRd',
        colorbar=False
    )

    # Coefficient of variation
    im3 = plot_gridded_surface(
        cv_grid, Lon, Lat, ax=axes[2],
        title='(c) Coefficient of Variation',
        cmap='YlOrRd',
        vmin=0, vmax=15,
        colorbar=False
    )

    # Add individual colorbars with proper height for each panel
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, aspect=20)
    cbar1.set_label('NO₂ (µg/m³)', fontsize=10)

    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, aspect=20)
    cbar2.set_label('SD (µg/m³)', fontsize=10)

    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, aspect=20)
    cbar3.set_label('CV (%)', fontsize=10)

    fig.suptitle(
        'GAM-SSM Prediction with Uncertainty',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    adjust_colorbars_to_axes(fig, axes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_temporal_gridded_sequence(
    coordinates: np.ndarray,
    observed_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    time_points: list[int],
    output_path: Path,
    lon_bounds: Optional[Tuple[float, float]] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    resolution: int = 150,
) -> None:
    """Create temporal sequence of gridded maps.

    Parameters
    ----------
    coordinates : np.ndarray (n_locations, 2)
        Point coordinates
    observed_matrix : np.ndarray (n_times, n_locations)
        Observed values over time
    predicted_matrix : np.ndarray (n_times, n_locations)
        Predicted values over time
    time_points : list[int]
        Time indices to plot
    output_path : Path
        Output file path
    lon_bounds, lat_bounds : tuple, optional
        Coordinate bounds
    resolution : int
        Grid resolution
    """
    # Compute bounds if not provided
    if lon_bounds is None:
        lon_bounds = (coordinates[:, 0].min(), coordinates[:, 0].max())
    if lat_bounds is None:
        lat_bounds = (coordinates[:, 1].min(), coordinates[:, 1].max())

    # Create grid
    Lon, Lat, _, _ = create_interpolation_grid(
        lon_bounds[0], lon_bounds[1],
        lat_bounds[0], lat_bounds[1],
        resolution
    )

    n_times = len(time_points)
    fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 8))

    # Interpolate all time points
    all_obs = []
    all_pred = []
    for t_idx in time_points:
        obs_grid = interpolate_to_grid(
            coordinates, observed_matrix[t_idx], Lon, Lat
        )
        pred_grid = interpolate_to_grid(
            coordinates, predicted_matrix[t_idx], Lon, Lat
        )
        all_obs.append(obs_grid)
        all_pred.append(pred_grid)

    # Compute global color limits
    vmin = min([g.min() for g in all_obs + all_pred])
    vmax = max([g.max() for g in all_obs + all_pred])

    for i, t_idx in enumerate(time_points):
        # Observed (top row)
        ax = axes[0, i]
        im = plot_gridded_surface(
            all_obs[i], Lon, Lat, ax=ax,
            title=f'Time {t_idx}',
            vmin=vmin, vmax=vmax,
            colorbar=False,  # We'll add a shared colorbar
            show_coords=(i == 0)
        )
        if i == 0:
            ax.set_ylabel('Observed\nLatitude', fontsize=10)
        else:
            ax.set_ylabel('')

        # Predicted (bottom row)
        ax = axes[1, i]
        plot_gridded_surface(
            all_pred[i], Lon, Lat, ax=ax,
            title='',
            vmin=vmin, vmax=vmax,
            colorbar=False,  # We'll add a shared colorbar
            show_coords=(i == 0)
        )
        if i == 0:
            ax.set_ylabel('GAM-SSM\nLatitude', fontsize=10)
        else:
            ax.set_ylabel('')

    # Add a single colorbar that spans both rows on the right
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.04, aspect=20, label='NO₂ (µg/m³)')

    fig.suptitle(
        'Temporal Evolution: Observed (top) vs GAM-SSM Predicted (bottom)',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    adjust_shared_colorbar_height(fig, axes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
