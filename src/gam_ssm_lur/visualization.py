"""
Comprehensive Visualization Module for GAM-SSM-LUR.

This module provides publication-ready visualizations including:
1. Spatial maps of observed, predicted, and smoothed NO₂ surfaces
2. Model comparison plots (GAM-only vs GAM-SSM)
3. Temporal evolution with uncertainty bands
4. Residual diagnostics
5. Feature importance visualizations
6. Convergence diagnostics
7. Animated temporal sequences

References
----------
.. [1] Matplotlib documentation: https://matplotlib.org/
.. [2] Cartopy for geospatial plotting: https://scitools.org.uk/cartopy/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-friendly palettes
CMAP_SEQUENTIAL = 'viridis'
CMAP_DIVERGING = 'RdBu_r'
CMAP_CATEGORICAL = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']


@dataclass
class PlotStyle:
    """Configuration for plot styling."""
    figsize: Tuple[float, float] = (10, 8)
    cmap: str = 'viridis'
    alpha: float = 0.8
    marker_size: float = 20
    line_width: float = 1.5
    title_fontsize: int = 12
    label_fontsize: int = 10
    colorbar_label: str = 'NO₂ (µg/m³)'


class SpatialVisualizer:
    """Visualize spatial distributions of air pollution predictions.
    
    Creates publication-ready maps showing:
    - Observed concentrations
    - Predicted concentrations (GAM, GAM-SSM)
    - Prediction uncertainty
    - Residual patterns
    - Model comparisons
    
    Parameters
    ----------
    coordinates : NDArray
        Spatial coordinates, shape (n_locations, 2) as (lon, lat) or (x, y)
    coordinate_system : {'latlon', 'projected'}
        Coordinate reference system
    extent : tuple, optional
        Map extent as (xmin, xmax, ymin, ymax)
    style : PlotStyle, optional
        Plot styling configuration
        
    Examples
    --------
    >>> viz = SpatialVisualizer(coordinates=coords, coordinate_system='latlon')
    >>> viz.plot_surface(predictions, title='Predicted NO₂')
    >>> viz.compare_models(gam_pred, hybrid_pred, observed)
    """
    
    def __init__(
        self,
        coordinates: NDArray,
        coordinate_system: Literal['latlon', 'projected'] = 'latlon',
        extent: Optional[Tuple[float, float, float, float]] = None,
        style: Optional[PlotStyle] = None,
    ):
        self.coordinates = np.asarray(coordinates)
        self.coordinate_system = coordinate_system
        self.style = style or PlotStyle()
        
        # Compute extent if not provided
        if extent is None:
            x, y = self.coordinates[:, 0], self.coordinates[:, 1]
            padding = 0.02 * max(x.max() - x.min(), y.max() - y.min())
            self.extent = (x.min() - padding, x.max() + padding,
                          y.min() - padding, y.max() + padding)
        else:
            self.extent = extent
            
    def plot_surface(
        self,
        values: NDArray,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        show_points: bool = False,
        interpolation: Literal['none', 'nearest', 'linear', 'cubic'] = 'none',
    ) -> plt.Axes:
        """Plot spatial surface of values.
        
        Parameters
        ----------
        values : NDArray
            Values at each coordinate, shape (n_locations,)
        ax : Axes, optional
            Matplotlib axes to plot on
        title : str, optional
            Plot title
        cmap : str, optional
            Colormap name
        vmin, vmax : float, optional
            Colorbar limits
        colorbar : bool
            Whether to show colorbar
        colorbar_label : str, optional
            Colorbar label
        show_points : bool
            Whether to show individual points vs interpolated surface
        interpolation : str
            Interpolation method for gridded display
            
        Returns
        -------
        ax : Axes
            The matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize)
            
        cmap = cmap or self.style.cmap
        colorbar_label = colorbar_label or self.style.colorbar_label
        
        x, y = self.coordinates[:, 0], self.coordinates[:, 1]
        
        if show_points or interpolation == 'none':
            # Scatter plot
            sc = ax.scatter(
                x, y, c=values,
                cmap=cmap, s=self.style.marker_size,
                alpha=self.style.alpha,
                vmin=vmin, vmax=vmax,
                edgecolors='none'
            )
        else:
            # Interpolated surface
            from scipy.interpolate import griddata
            
            # Create grid
            xi = np.linspace(self.extent[0], self.extent[1], 200)
            yi = np.linspace(self.extent[2], self.extent[3], 200)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate
            Zi = griddata((x, y), values, (Xi, Yi), method=interpolation)
            
            sc = ax.pcolormesh(
                Xi, Yi, Zi,
                cmap=cmap, shading='auto',
                vmin=vmin, vmax=vmax
            )
            
        if colorbar:
            cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label(colorbar_label, fontsize=self.style.label_fontsize)
            
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        
        if self.coordinate_system == 'latlon':
            ax.set_xlabel('Longitude', fontsize=self.style.label_fontsize)
            ax.set_ylabel('Latitude', fontsize=self.style.label_fontsize)
        else:
            ax.set_xlabel('Easting (m)', fontsize=self.style.label_fontsize)
            ax.set_ylabel('Northing (m)', fontsize=self.style.label_fontsize)
            
        if title:
            ax.set_title(title, fontsize=self.style.title_fontsize)
            
        ax.set_aspect('equal')
        
        return ax
        
    def plot_comparison_grid(
        self,
        observed: NDArray,
        gam_predicted: NDArray,
        hybrid_predicted: NDArray,
        time_idx: Optional[int] = None,
        suptitle: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Create side-by-side comparison of observed vs model predictions.
        
        Parameters
        ----------
        observed : NDArray
            Observed values, shape (n_locations,) or (n_times, n_locations)
        gam_predicted : NDArray
            GAM-only predictions
        hybrid_predicted : NDArray
            Hybrid GAM-SSM predictions
        time_idx : int, optional
            Time index to plot (if data is 2D)
        suptitle : str, optional
            Overall figure title
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        # Handle 2D arrays
        if observed.ndim == 2:
            t = time_idx or 0
            observed = observed[t]
            gam_predicted = gam_predicted[t]
            hybrid_predicted = hybrid_predicted[t]
            
        # Compute residuals
        gam_residuals = observed - gam_predicted
        hybrid_residuals = observed - hybrid_predicted
        
        # Compute common color limits
        vmin_conc = min(observed.min(), gam_predicted.min(), hybrid_predicted.min())
        vmax_conc = max(observed.max(), gam_predicted.max(), hybrid_predicted.max())
        
        vmax_res = max(np.abs(gam_residuals).max(), np.abs(hybrid_residuals).max())
        vmin_res = -vmax_res
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.3)
        
        # Row 1: Concentrations
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_surface(observed, ax=ax1, title='(a) Observed',
                         vmin=vmin_conc, vmax=vmax_conc)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_surface(gam_predicted, ax=ax2, title='(b) GAM-LUR Predicted',
                         vmin=vmin_conc, vmax=vmax_conc)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_surface(hybrid_predicted, ax=ax3, title='(c) GAM-SSM Predicted',
                         vmin=vmin_conc, vmax=vmax_conc)
        
        # Row 2: Residuals
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')  # Empty for balance
        
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_surface(gam_residuals, ax=ax5, title='(d) GAM-LUR Residuals',
                         cmap=CMAP_DIVERGING, vmin=vmin_res, vmax=vmax_res,
                         colorbar_label='Residual (µg/m³)')
        
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_surface(hybrid_residuals, ax=ax6, title='(e) GAM-SSM Residuals',
                         cmap=CMAP_DIVERGING, vmin=vmin_res, vmax=vmax_res,
                         colorbar_label='Residual (µg/m³)')
        
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
            
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison grid to {save_path}")
            
        return fig
        
    def plot_uncertainty_map(
        self,
        mean: NDArray,
        std: NDArray,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot prediction uncertainty map.
        
        Parameters
        ----------
        mean : NDArray
            Predicted mean values
        std : NDArray
            Prediction standard deviations
        ax : Axes, optional
            Matplotlib axes
        title : str, optional
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig = ax.figure
            axes = [ax, ax]
            
        # Plot mean
        self.plot_surface(mean, ax=axes[0], title='(a) Predicted Mean',
                         colorbar_label='NO₂ (µg/m³)')
        
        # Plot uncertainty
        self.plot_surface(std, ax=axes[1], title='(b) Prediction Uncertainty (SD)',
                         cmap='YlOrRd', colorbar_label='SD (µg/m³)')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_temporal_snapshots(
        self,
        values: NDArray,
        time_indices: List[int],
        time_labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot spatial maps at multiple time points.
        
        Parameters
        ----------
        values : NDArray
            Values array, shape (n_times, n_locations)
        time_indices : list of int
            Time indices to plot
        time_labels : list of str, optional
            Labels for each time point
        title : str, optional
            Overall title
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        n_times = len(time_indices)
        ncols = min(3, n_times)
        nrows = (n_times + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = np.atleast_2d(axes).flatten()
        
        # Common color limits
        vmin = values[time_indices].min()
        vmax = values[time_indices].max()
        
        for i, t_idx in enumerate(time_indices):
            label = time_labels[i] if time_labels else f'Day {t_idx}'
            panel_label = chr(ord('a') + i)
            self.plot_surface(
                values[t_idx], ax=axes[i],
                title=f'({panel_label}) {label}',
                vmin=vmin, vmax=vmax,
                colorbar=(i == len(time_indices) - 1)
            )
            
        # Hide unused axes
        for i in range(len(time_indices), len(axes)):
            axes[i].axis('off')
            
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_rmse_map(
        self,
        observed: NDArray,
        predicted: NDArray,
        title: str = 'Spatial RMSE Distribution',
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot spatial distribution of RMSE.
        
        Parameters
        ----------
        observed : NDArray
            Observed values, shape (n_times, n_locations)
        predicted : NDArray
            Predicted values, shape (n_times, n_locations)
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        # Compute RMSE at each location
        residuals = observed - predicted
        rmse_per_location = np.sqrt(np.mean(residuals**2, axis=0))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        self.plot_surface(
            rmse_per_location, ax=ax,
            title=title,
            cmap='YlOrRd',
            colorbar_label='RMSE (µg/m³)'
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class TemporalVisualizer:
    """Visualize temporal patterns and model dynamics.
    
    Creates visualizations for:
    - Time series with uncertainty bands
    - Multi-location comparisons
    - Diurnal/weekly patterns
    - Temporal variance evolution
    """
    
    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = style or PlotStyle()
        
    def plot_time_series(
        self,
        observed: NDArray,
        predicted: NDArray,
        lower: Optional[NDArray] = None,
        upper: Optional[NDArray] = None,
        time_index: Optional[NDArray] = None,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        xlabel: str = 'Time',
        ylabel: str = 'NO₂ (µg/m³)',
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot time series with optional prediction intervals.
        
        Parameters
        ----------
        observed : NDArray
            Observed values, shape (n_times,)
        predicted : NDArray
            Predicted values, shape (n_times,)
        lower, upper : NDArray, optional
            Prediction interval bounds
        time_index : NDArray, optional
            Time values for x-axis (datetime or numeric)
        ax : Axes, optional
            Matplotlib axes
        title : str, optional
            Plot title
        xlabel, ylabel : str
            Axis labels
        show_legend : bool
            Whether to show legend
            
        Returns
        -------
        ax : Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
            
        if time_index is None:
            time_index = np.arange(len(observed))
            
        # Plot prediction interval
        if lower is not None and upper is not None:
            ax.fill_between(time_index, lower, upper,
                           alpha=0.3, color=CMAP_CATEGORICAL[0],
                           label='95% CI')
            
        # Plot observed
        ax.scatter(time_index, observed, s=10, c='black', alpha=0.5,
                  label='Observed', zorder=3)
        
        # Plot predicted
        ax.plot(time_index, predicted, color=CMAP_CATEGORICAL[0],
               linewidth=self.style.line_width, label='Predicted', zorder=2)
        
        ax.set_xlabel(xlabel, fontsize=self.style.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.style.label_fontsize)
        
        if title:
            ax.set_title(title, fontsize=self.style.title_fontsize)
            
        if show_legend:
            ax.legend(loc='upper right')
            
        # Format x-axis for datetime
        if isinstance(time_index[0], (pd.Timestamp, np.datetime64)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
        return ax
        
    def plot_multi_location_timeseries(
        self,
        observed: NDArray,
        predicted: NDArray,
        lower: Optional[NDArray] = None,
        upper: Optional[NDArray] = None,
        location_ids: Optional[List[int]] = None,
        n_locations: int = 6,
        time_index: Optional[NDArray] = None,
        suptitle: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot time series for multiple locations.
        
        Parameters
        ----------
        observed : NDArray
            Observed values, shape (n_times, n_locations)
        predicted : NDArray
            Predicted values, shape (n_times, n_locations)
        lower, upper : NDArray, optional
            Prediction interval bounds
        location_ids : list of int, optional
            Specific location indices to plot
        n_locations : int
            Number of locations to plot if location_ids not provided
        time_index : NDArray, optional
            Time values for x-axis
        suptitle : str, optional
            Overall title
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        if location_ids is None:
            # Select evenly spaced locations
            n_total = observed.shape[1]
            location_ids = np.linspace(0, n_total-1, n_locations, dtype=int)
            
        n_locs = len(location_ids)
        ncols = min(3, n_locs)
        nrows = (n_locs + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows))
        axes = np.atleast_2d(axes).flatten()
        
        for i, loc_id in enumerate(location_ids):
            panel_label = chr(ord('a') + i)
            
            loc_lower = lower[:, loc_id] if lower is not None else None
            loc_upper = upper[:, loc_id] if upper is not None else None
            
            self.plot_time_series(
                observed[:, loc_id],
                predicted[:, loc_id],
                lower=loc_lower,
                upper=loc_upper,
                time_index=time_index,
                ax=axes[i],
                title=f'({panel_label}) Location {loc_id}',
                show_legend=(i == 0)
            )
            
        # Hide unused axes
        for i in range(n_locs, len(axes)):
            axes[i].axis('off')
            
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_variance_evolution(
        self,
        observed: NDArray,
        smoothed: NDArray,
        time_index: Optional[NDArray] = None,
        title: str = 'Temporal Variance Comparison',
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot temporal evolution of variance (observed vs smoothed).
        
        Parameters
        ----------
        observed : NDArray
            Observed values, shape (n_times, n_locations)
        smoothed : NDArray
            Smoothed values, shape (n_times, n_locations)
        time_index : NDArray, optional
            Time values
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        if time_index is None:
            time_index = np.arange(observed.shape[0])
            
        # Compute variance across space at each time
        obs_var = np.var(observed, axis=1)
        smooth_var = np.var(smoothed, axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(time_index, obs_var, 'k-', linewidth=1.5,
               label='Observed variance', alpha=0.7)
        ax.plot(time_index, smooth_var, color=CMAP_CATEGORICAL[0],
               linewidth=2, label='Smoothed variance')
        
        ax.fill_between(time_index, smooth_var, obs_var,
                       alpha=0.2, color=CMAP_CATEGORICAL[0],
                       label='Variance reduction')
        
        ax.set_xlabel('Time', fontsize=self.style.label_fontsize)
        ax.set_ylabel('Spatial Variance', fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)
        ax.legend()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class ModelComparisonVisualizer:
    """Visualize model comparisons between GAM-only and GAM-SSM.
    
    Creates comprehensive comparison plots including:
    - Observed vs predicted scatter plots
    - Performance metric comparisons
    - Residual distributions
    - Spatial and temporal error patterns
    """
    
    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = style or PlotStyle()
        
    def plot_observed_vs_predicted(
        self,
        observed: NDArray,
        gam_predicted: NDArray,
        hybrid_predicted: NDArray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Create observed vs predicted comparison for both models.
        
        Parameters
        ----------
        observed : NDArray
            Observed values (flattened)
        gam_predicted : NDArray
            GAM-only predictions (flattened)
        hybrid_predicted : NDArray
            Hybrid model predictions (flattened)
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        observed = observed.flatten()
        gam_predicted = gam_predicted.flatten()
        hybrid_predicted = hybrid_predicted.flatten()
        
        # Compute metrics
        def compute_metrics(y_true, y_pred):
            residuals = y_true - y_pred
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - ss_res / ss_tot
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            return {'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr}
            
        gam_metrics = compute_metrics(observed, gam_predicted)
        hybrid_metrics = compute_metrics(observed, hybrid_predicted)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Common limits
        all_vals = np.concatenate([observed, gam_predicted, hybrid_predicted])
        vmin, vmax = all_vals.min(), all_vals.max()
        
        # GAM-only
        ax = axes[0]
        ax.scatter(observed, gam_predicted, alpha=0.1, s=3, c=CMAP_CATEGORICAL[1])
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='1:1 line')
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('Observed NO₂ (µg/m³)')
        ax.set_ylabel('Predicted NO₂ (µg/m³)')
        ax.set_title(f"(a) GAM-LUR Only\nR² = {gam_metrics['r2']:.3f}, RMSE = {gam_metrics['rmse']:.2f}")
        ax.set_aspect('equal')
        ax.legend(loc='lower right')
        
        # Hybrid
        ax = axes[1]
        ax.scatter(observed, hybrid_predicted, alpha=0.1, s=3, c=CMAP_CATEGORICAL[0])
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='1:1 line')
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('Observed NO₂ (µg/m³)')
        ax.set_ylabel('Predicted NO₂ (µg/m³)')
        ax.set_title(f"(b) Hybrid GAM-SSM\nR² = {hybrid_metrics['r2']:.3f}, RMSE = {hybrid_metrics['rmse']:.2f}")
        ax.set_aspect('equal')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_metric_comparison(
        self,
        gam_metrics: Dict[str, float],
        hybrid_metrics: Dict[str, float],
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Create bar chart comparing model metrics.
        
        Parameters
        ----------
        gam_metrics : dict
            Metrics for GAM-only model
        hybrid_metrics : dict
            Metrics for hybrid model
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        metrics = ['RMSE', 'MAE', 'R²', 'Correlation']
        gam_values = [gam_metrics['rmse'], gam_metrics['mae'],
                     gam_metrics['r2'], gam_metrics.get('correlation', gam_metrics.get('corr', 0))]
        hybrid_values = [hybrid_metrics['rmse'], hybrid_metrics['mae'],
                        hybrid_metrics['r2'], hybrid_metrics.get('correlation', hybrid_metrics.get('corr', 0))]
        
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        
        for i, (metric, gam_val, hybrid_val) in enumerate(zip(metrics, gam_values, hybrid_values)):
            ax = axes[i]
            x = [0, 1]
            vals = [gam_val, hybrid_val]
            colors = [CMAP_CATEGORICAL[1], CMAP_CATEGORICAL[0]]
            
            bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(['GAM-LUR', 'GAM-SSM'])
            ax.set_title(metric, fontsize=12, fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(vals),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
                
            # Add improvement annotation for RMSE and MAE
            if metric in ['RMSE', 'MAE'] and gam_val > 0:
                improvement = (gam_val - hybrid_val) / gam_val * 100
                ax.annotate(f'{improvement:.1f}%↓', 
                           xy=(0.5, max(vals)*0.5), fontsize=11,
                           color='green', fontweight='bold', ha='center')
                
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_residual_comparison(
        self,
        observed: NDArray,
        gam_predicted: NDArray,
        hybrid_predicted: NDArray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Compare residual distributions between models.
        
        Parameters
        ----------
        observed : NDArray
            Observed values
        gam_predicted : NDArray
            GAM-only predictions
        hybrid_predicted : NDArray
            Hybrid predictions
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        gam_residuals = (observed - gam_predicted).flatten()
        hybrid_residuals = (observed - hybrid_predicted).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histograms
        ax = axes[0, 0]
        ax.hist(gam_residuals, bins=50, alpha=0.6, color=CMAP_CATEGORICAL[1],
               label=f'GAM-LUR (SD={gam_residuals.std():.2f})', density=True)
        ax.hist(hybrid_residuals, bins=50, alpha=0.6, color=CMAP_CATEGORICAL[0],
               label=f'GAM-SSM (SD={hybrid_residuals.std():.2f})', density=True)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Residual (µg/m³)')
        ax.set_ylabel('Density')
        ax.set_title('(a) Residual Distributions')
        ax.legend()
        
        # Q-Q plots
        from scipy import stats
        
        ax = axes[0, 1]
        stats.probplot(gam_residuals, dist="norm", plot=ax)
        ax.get_lines()[0].set_color(CMAP_CATEGORICAL[1])
        ax.get_lines()[0].set_alpha(0.5)
        ax.get_lines()[1].set_color('red')
        ax.set_title('(b) Q-Q Plot: GAM-LUR Residuals')
        
        ax = axes[1, 0]
        stats.probplot(hybrid_residuals, dist="norm", plot=ax)
        ax.get_lines()[0].set_color(CMAP_CATEGORICAL[0])
        ax.get_lines()[0].set_alpha(0.5)
        ax.get_lines()[1].set_color('red')
        ax.set_title('(c) Q-Q Plot: GAM-SSM Residuals')
        
        # Residuals vs fitted
        ax = axes[1, 1]
        ax.scatter(gam_predicted.flatten(), gam_residuals, alpha=0.1, s=3,
                  c=CMAP_CATEGORICAL[1], label='GAM-LUR')
        ax.scatter(hybrid_predicted.flatten(), hybrid_residuals, alpha=0.1, s=3,
                  c=CMAP_CATEGORICAL[0], label='GAM-SSM')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Fitted values (µg/m³)')
        ax.set_ylabel('Residual (µg/m³)')
        ax.set_title('(d) Residuals vs Fitted')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class DiagnosticsVisualizer:
    """Visualize model diagnostics and convergence."""
    
    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = style or PlotStyle()
        
    def plot_em_convergence(
        self,
        log_likelihoods: List[float],
        param_traces: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot EM algorithm convergence diagnostics.
        
        Parameters
        ----------
        log_likelihoods : list of float
            Log-likelihood values at each iteration
        param_traces : dict, optional
            Parameter traces (e.g., {'tr_T': [...], 'tr_Q': [...], 'tr_H': [...]})
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        iterations = range(len(log_likelihoods))
        
        # Log-likelihood
        ax = axes[0, 0]
        ax.plot(iterations, log_likelihoods, 'b-o', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')
        ax.set_title('(a) Log-likelihood')
        ax.grid(True, alpha=0.3)
        
        # Parameter traces
        ax = axes[0, 1]
        if param_traces:
            for name, values in param_traces.items():
                ax.plot(iterations[:len(values)], values, '-', label=name, linewidth=1.5)
            ax.legend()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter trace')
        ax.set_title('(b) Parameter Traces')
        ax.grid(True, alpha=0.3)
        
        # Log-likelihood increment
        ax = axes[1, 0]
        ll_change = np.abs(np.diff(log_likelihoods))
        ax.semilogy(range(1, len(log_likelihoods)), ll_change, 'b-o', markersize=4)
        ax.axhline(y=1e-6, color='r', linestyle='--', label='Convergence threshold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('|ΔLL|')
        ax.set_title('(c) Log-likelihood Increment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = (
            f"EM Convergence Summary\n"
            f"{'='*30}\n\n"
            f"Iterations: {len(log_likelihoods)}\n"
            f"Initial LL: {log_likelihoods[0]:.4e}\n"
            f"Final LL: {log_likelihoods[-1]:.4e}\n"
            f"Total ΔLL: {log_likelihoods[-1] - log_likelihoods[0]:.4e}\n"
            f"Final |ΔLL|: {ll_change[-1]:.4e}\n"
        )
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.set_title('(d) Summary')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_residual_diagnostics(
        self,
        residuals: NDArray,
        fitted: NDArray,
        time_index: Optional[NDArray] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Create comprehensive residual diagnostics plot.
        
        Parameters
        ----------
        residuals : NDArray
            Model residuals
        fitted : NDArray
            Fitted values
        time_index : NDArray, optional
            Time index for temporal pattern
        save_path : str or Path, optional
            Path to save figure
            
        Returns
        -------
        fig : Figure
        """
        from scipy import stats
        
        residuals = residuals.flatten()
        fitted = fitted.flatten()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Residuals vs fitted
        ax = axes[0, 0]
        ax.scatter(fitted, residuals, alpha=0.1, s=3)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        ax.set_title('(a) Residuals vs Fitted')
        
        # Q-Q plot
        ax = axes[0, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('(b) Q-Q Plot')
        
        # Histogram
        ax = axes[0, 2]
        ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', lw=2)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('(c) Residual Distribution')
        
        # Temporal pattern (if residuals are 2D or time index provided)
        ax = axes[1, 0]
        if time_index is not None and len(time_index) < len(residuals):
            # Reshape to (n_times, n_locations)
            n_times = len(time_index)
            n_locs = len(residuals) // n_times
            resid_matrix = residuals.reshape(n_times, n_locs)
            daily_mean = resid_matrix.mean(axis=1)
            daily_std = resid_matrix.std(axis=1)
            
            ax.plot(time_index, daily_mean, 'b-', linewidth=1)
            ax.fill_between(time_index, daily_mean - daily_std, daily_mean + daily_std,
                           alpha=0.3, color='blue')
            ax.axhline(y=0, color='r', linestyle='--')
        else:
            # Just plot all residuals in order
            ax.plot(residuals[:min(1000, len(residuals))], 'b-', alpha=0.5, linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Residual')
        ax.set_title('(d) Temporal Pattern')
        
        # Scale-location plot
        ax = axes[1, 1]
        sqrt_abs_resid = np.sqrt(np.abs(residuals))
        ax.scatter(fitted, sqrt_abs_resid, alpha=0.1, s=3)
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('√|Residuals|')
        ax.set_title('(e) Scale-Location')
        
        # ACF
        ax = axes[1, 2]
        nlags = min(50, len(residuals) // 4)
        acf_vals = np.correlate(residuals - residuals.mean(), residuals - residuals.mean(), mode='full')
        acf_vals = acf_vals[len(acf_vals)//2:]
        acf_vals = acf_vals[:nlags+1] / acf_vals[0]
        
        ax.bar(range(nlags+1), acf_vals, width=0.3, color='blue', alpha=0.7)
        ax.axhline(y=0, color='black')
        ax.axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
        ax.axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title('(f) Residual ACF')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def create_publication_figure_set(
    model,
    output_dir: Union[str, Path],
    coordinates: Optional[NDArray] = None,
) -> Dict[str, Path]:
    """Generate complete set of publication-ready figures.
    
    Creates all figures needed for a typical air quality modeling paper:
    - Figure 1: Study area and spatial discretization
    - Figure 2: Feature importance (SHAP)
    - Figure 3: Temporal evolution at sample locations
    - Figure 4: Spatial NO₂ distribution (observed vs smoothed)
    - Figure 5: EM convergence diagnostics
    - Figure 6: Model comparison (GAM vs GAM-SSM)
    - Figure 7: Residual diagnostics
    
    Parameters
    ----------
    model : HybridGAMSSM
        Fitted hybrid model
    output_dir : str or Path
        Directory to save figures
    coordinates : NDArray, optional
        Spatial coordinates for maps
        
    Returns
    -------
    dict
        Mapping of figure names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figure_paths = {}
    
    # Get model data
    predictions = model.predict()
    y_matrix = model._y_matrix
    
    if coordinates is None:
        # Create dummy coordinates if not provided
        n_locs = model.n_locations_
        coordinates = np.column_stack([
            np.random.randn(n_locs) * 0.1 - 6.26,  # Dublin-ish longitude
            np.random.randn(n_locs) * 0.05 + 53.35  # Dublin-ish latitude
        ])
        
    # Initialize visualizers
    spatial_viz = SpatialVisualizer(coordinates)
    temporal_viz = TemporalVisualizer()
    comparison_viz = ModelComparisonVisualizer()
    diagnostics_viz = DiagnosticsVisualizer()
    
    # Figure 4: Temporal evolution
    fig_path = output_dir / 'fig4_temporal_evolution.png'
    temporal_viz.plot_multi_location_timeseries(
        observed=y_matrix,
        predicted=predictions.total,
        lower=predictions.lower,
        upper=predictions.upper,
        n_locations=6,
        suptitle='Figure 4: Temporal Evolution at Representative Locations',
        save_path=fig_path
    )
    figure_paths['temporal_evolution'] = fig_path
    
    # Figure 5: Spatial distribution
    fig_path = output_dir / 'fig5_spatial_distribution.png'
    spatial_viz.plot_temporal_snapshots(
        values=predictions.total,
        time_indices=[0, model.n_times_//3, 2*model.n_times_//3],
        time_labels=['Day 0', f'Day {model.n_times_//3}', f'Day {2*model.n_times_//3}'],
        title='Figure 5: Spatial NO₂ Distribution Over Time',
        save_path=fig_path
    )
    figure_paths['spatial_distribution'] = fig_path
    
    # Figure 6: EM Convergence
    fig_path = output_dir / 'fig6_em_convergence.png'
    em_history = model.get_em_convergence()
    diagnostics_viz.plot_em_convergence(
        log_likelihoods=em_history['log_likelihood'].tolist(),
        param_traces={
            'tr(T)': em_history['tr_T'].tolist(),
            'tr(Q)': em_history['tr_Q'].tolist(),
            'tr(H)': em_history['tr_H'].tolist(),
        },
        save_path=fig_path
    )
    figure_paths['em_convergence'] = fig_path
    
    # Figure 7: Model comparison
    # First get GAM-only predictions
    gam_pred = model.gam_.predict(model._X_train)
    gam_pred_matrix = gam_pred.reshape(model.n_times_, model.n_locations_)
    
    fig_path = output_dir / 'fig7_model_comparison.png'
    comparison_viz.plot_observed_vs_predicted(
        observed=y_matrix,
        gam_predicted=gam_pred_matrix,
        hybrid_predicted=predictions.total,
        save_path=fig_path
    )
    figure_paths['model_comparison'] = fig_path
    
    # Figure 8: Residual diagnostics
    fig_path = output_dir / 'fig8_residual_diagnostics.png'
    residuals = y_matrix - predictions.total
    diagnostics_viz.plot_residual_diagnostics(
        residuals=residuals,
        fitted=predictions.total,
        time_index=np.arange(model.n_times_),
        save_path=fig_path
    )
    figure_paths['residual_diagnostics'] = fig_path
    
    # Additional: RMSE spatial map
    fig_path = output_dir / 'fig_rmse_map.png'
    spatial_viz.plot_rmse_map(
        observed=y_matrix,
        predicted=predictions.total,
        title='Spatial RMSE Distribution',
        save_path=fig_path
    )
    figure_paths['rmse_map'] = fig_path
    
    # Additional: Variance evolution
    fig_path = output_dir / 'fig_variance_evolution.png'
    temporal_viz.plot_variance_evolution(
        observed=y_matrix,
        smoothed=predictions.total,
        title='Temporal Variance: Observed vs Smoothed',
        save_path=fig_path
    )
    figure_paths['variance_evolution'] = fig_path
    
    # Additional: Full spatial comparison grid
    fig_path = output_dir / 'fig_spatial_comparison.png'
    spatial_viz.plot_comparison_grid(
        observed=y_matrix,
        gam_predicted=gam_pred_matrix,
        hybrid_predicted=predictions.total,
        time_idx=model.n_times_//2,
        suptitle=f'Spatial Comparison at Day {model.n_times_//2}',
        save_path=fig_path
    )
    figure_paths['spatial_comparison'] = fig_path
    
    logger.info(f"Generated {len(figure_paths)} publication figures in {output_dir}")
    
    return figure_paths
