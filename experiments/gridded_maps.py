#!/usr/bin/env python
"""
Gridded Surface Maps for LUR Visualization.

Creates publication-ready interpolated raster maps showing:
- Continuous NO₂ concentration surfaces
- Model comparison as gridded maps
- Difference maps (hybrid - baseline)
- Uncertainty surfaces

These are the standard presentation format for LUR studies.

NOTE: This script now uses the unified visualization module (mapping_utils.py)
to avoid code duplication across examples.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from output_utils import make_experiment_dirs
from mapping_utils import (
    create_interpolation_grid,
    interpolate_to_grid,
    plot_gridded_surface,
    create_gridded_comparison,
    create_gridded_residual_map,
    create_uncertainty_surface,
    create_temporal_gridded_sequence,
)

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_ROOT, OUTPUT_GROUPS = make_experiment_dirs(
    base="outputs",
    groups=["gridded_maps"],
)
OUTPUT_DIR = OUTPUT_GROUPS["gridded_maps"]

# Dublin bounding box (approximate)
DUBLIN_BOUNDS = {
    'lon_min': -6.45,
    'lon_max': -6.05,
    'lat_min': 53.25,
    'lat_max': 53.45,
}

# Grid resolution
GRID_RESOLUTION = 200  # 200x200 grid cells


def generate_realistic_data(n_locations=300, n_days=50, seed=42):
    """Generate realistic spatiotemporal NO₂ data for Dublin."""
    np.random.seed(seed)
    
    # Dublin coordinates - more locations for better interpolation
    lon = np.random.uniform(DUBLIN_BOUNDS['lon_min'] + 0.05, 
                           DUBLIN_BOUNDS['lon_max'] - 0.05, n_locations)
    lat = np.random.uniform(DUBLIN_BOUNDS['lat_min'] + 0.02,
                           DUBLIN_BOUNDS['lat_max'] - 0.02, n_locations)
    coordinates = np.column_stack([lon, lat])
    
    # Create spatial pattern based on Dublin geography
    # Higher pollution near city center (~-6.26, 53.35) and along M50 corridor
    city_center_lon, city_center_lat = -6.26, 53.35
    
    # Distance from city center
    dist_center = np.sqrt((lon - city_center_lon)**2 + (lat - city_center_lat)**2)
    
    # M50 corridor effect (rough arc west of city)
    m50_effect = np.exp(-((lon - (-6.35))**2) / 0.01) * 5
    
    # Port area effect (east)
    port_effect = np.exp(-((lon - (-6.15))**2 + (lat - 53.35)**2) / 0.005) * 8
    
    # TRUE SPATIAL COMPONENT
    true_spatial = (
        35 +                                    # baseline
        -15 * dist_center +                     # decay from center
        m50_effect +                            # M50 motorway
        port_effect +                           # Dublin Port
        3 * np.sin(lon * 30) * np.cos(lat * 40) +  # fine-scale variation
        np.random.randn(n_locations) * 2        # local noise
    )
    true_spatial = np.clip(true_spatial, 8, 55)
    
    # TRUE TEMPORAL COMPONENT
    t = np.arange(n_days)
    weekly_cycle = 4 * np.sin(2 * np.pi * t / 7)
    ar_process = np.zeros(n_days)
    ar_process[0] = np.random.randn() * 2
    for i in range(1, n_days):
        ar_process[i] = 0.85 * ar_process[i-1] + np.random.randn() * 1.5
    true_temporal = weekly_cycle + ar_process
    
    # Generate observations
    observed = np.zeros((n_days, n_locations))
    for t_idx in range(n_days):
        observed[t_idx] = true_spatial + true_temporal[t_idx] + np.random.randn(n_locations) * 3
    observed = np.clip(observed, 0, 70)
    
    # GAM-LUR predictions (spatial only, misses temporal)
    gam_spatial = true_spatial + np.random.randn(n_locations) * 2
    gam_predictions = np.tile(gam_spatial, (n_days, 1))
    gam_predictions += np.random.randn(n_days, n_locations) * 1.5
    
    # Hybrid GAM-SSM predictions (captures both)
    hybrid_spatial = true_spatial + np.random.randn(n_locations) * 1.5
    hybrid_temporal = np.zeros(n_days)
    for t_idx in range(n_days):
        if t_idx == 0:
            hybrid_temporal[t_idx] = true_temporal[t_idx] + np.random.randn() * 0.8
        else:
            pred = 0.85 * hybrid_temporal[t_idx-1]
            obs_mean = (observed[t_idx] - hybrid_spatial).mean()
            hybrid_temporal[t_idx] = 0.6 * pred + 0.4 * obs_mean
    
    hybrid_predictions = np.zeros((n_days, n_locations))
    hybrid_std = np.zeros((n_days, n_locations))
    for t_idx in range(n_days):
        hybrid_predictions[t_idx] = hybrid_spatial + hybrid_temporal[t_idx]
        hybrid_std[t_idx] = 2.0 + np.random.rand(n_locations) * 1.0
    
    return {
        'coordinates': coordinates,
        'observed': observed,
        'gam_predictions': gam_predictions,
        'hybrid_predictions': hybrid_predictions,
        'hybrid_std': hybrid_std,
        'true_spatial': true_spatial,
        'true_temporal': true_temporal,
        'n_days': n_days,
        'n_locations': n_locations,
    }


# Note: create_interpolation_grid, interpolate_to_grid, and plot_gridded_surface
# are now imported from mapping_utils.py to avoid code duplication


# =============================================================================
# FIGURE 1: Three-panel comparison (Observed, GAM-LUR, GAM-SSM)
# =============================================================================
def create_three_panel_comparison(data, time_idx=25):
    """Create main comparison figure with gridded surfaces."""
    print(f"Creating Figure 1: Three-panel gridded comparison (Day {time_idx})...")

    coords = data['coordinates']

    # Use the unified module function
    create_gridded_comparison(
        coordinates=coords,
        observed=data['observed'][time_idx],
        baseline_pred=data['gam_predictions'][time_idx],
        hybrid_pred=data['hybrid_predictions'][time_idx],
        output_path=OUTPUT_DIR / 'fig1_three_panel_comparison.png',
        lon_bounds=(DUBLIN_BOUNDS['lon_min'], DUBLIN_BOUNDS['lon_max']),
        lat_bounds=(DUBLIN_BOUNDS['lat_min'], DUBLIN_BOUNDS['lat_max']),
        resolution=GRID_RESOLUTION,
        title_suffix=f' - Day {time_idx}',
    )

    print(f"  Saved: {OUTPUT_DIR / 'fig1_three_panel_comparison.png'}")


# =============================================================================
# FIGURE 2: Six-panel with residuals
# =============================================================================
def create_six_panel_with_residuals(data, time_idx=25):
    """Create 2x3 panel figure with predictions and residuals."""
    print(f"Creating Figure 2: Six-panel with residuals (Day {time_idx})...")

    coords = data['coordinates']

    # Use the unified module function
    create_gridded_residual_map(
        coordinates=coords,
        observed=data['observed'][time_idx],
        baseline_pred=data['gam_predictions'][time_idx],
        hybrid_pred=data['hybrid_predictions'][time_idx],
        output_path=OUTPUT_DIR / 'fig2_six_panel_residuals.png',
        lon_bounds=(DUBLIN_BOUNDS['lon_min'], DUBLIN_BOUNDS['lon_max']),
        lat_bounds=(DUBLIN_BOUNDS['lat_min'], DUBLIN_BOUNDS['lat_max']),
        resolution=GRID_RESOLUTION,
    )

    print(f"  Saved: {OUTPUT_DIR / 'fig2_six_panel_residuals.png'}")


# =============================================================================
# FIGURE 3: Difference map (GAM-SSM minus GAM-LUR)
# =============================================================================
def create_difference_map(data, time_idx=25):
    """Create map showing improvement of hybrid over baseline."""
    print(f"Creating Figure 3: Difference map (Day {time_idx})...")
    
    Lon, Lat, _, _ = create_interpolation_grid()
    coords = data['coordinates']
    
    # Compute absolute errors at each point
    obs = data['observed'][time_idx]
    gam_pred = data['gam_predictions'][time_idx]
    hybrid_pred = data['hybrid_predictions'][time_idx]
    
    gam_abs_error = np.abs(obs - gam_pred)
    hybrid_abs_error = np.abs(obs - hybrid_pred)
    
    # Improvement = GAM error - Hybrid error (positive = hybrid is better)
    improvement = gam_abs_error - hybrid_abs_error
    
    # Interpolate
    improvement_grid = interpolate_to_grid(coords, improvement, Lon, Lat)
    gam_error_grid = interpolate_to_grid(coords, gam_abs_error, Lon, Lat)
    hybrid_error_grid = interpolate_to_grid(coords, hybrid_abs_error, Lon, Lat)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # GAM absolute error
    vmax_err = max(gam_error_grid.max(), hybrid_error_grid.max())
    plot_gridded_surface(gam_error_grid, Lon, Lat, ax=axes[0],
                        title='(a) GAM-LUR Absolute Error',
                        cmap='YlOrRd', vmin=0, vmax=vmax_err,
                        colorbar_label='|Error| (µg/m³)')
    
    # Hybrid absolute error
    plot_gridded_surface(hybrid_error_grid, Lon, Lat, ax=axes[1],
                        title='(b) GAM-SSM Absolute Error',
                        cmap='YlOrRd', vmin=0, vmax=vmax_err,
                        colorbar_label='|Error| (µg/m³)')
    
    # Improvement map
    vmax_imp = np.abs(improvement_grid).max()
    plot_gridded_surface(improvement_grid, Lon, Lat, ax=axes[2],
                        title='(c) Improvement (GAM-LUR error − GAM-SSM error)',
                        cmap='RdYlGn', vmin=-vmax_imp, vmax=vmax_imp,
                        colorbar_label='Error Reduction (µg/m³)',
                        contours=True, n_contours=6)
    
    # Add annotation
    pct_improved = (improvement > 0).mean() * 100
    axes[2].text(0.02, 0.98, f'{pct_improved:.0f}% of area improved',
                transform=axes[2].transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(f'Prediction Error Comparison - Day {time_idx}',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_difference_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig3_difference_map.png'}")


# =============================================================================
# FIGURE 4: Temporal sequence of gridded maps
# =============================================================================
def create_temporal_sequence(data, time_points=None):
    """Create temporal sequence showing evolution of NO₂ surface."""
    if time_points is None:
        time_points = [0, 12, 25, 37, 49]
    
    print(f"Creating Figure 4: Temporal sequence (Days {time_points})...")
    
    Lon, Lat, _, _ = create_interpolation_grid()
    coords = data['coordinates']
    
    n_times = len(time_points)
    fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 8))
    
    # Compute global color limits
    all_obs = [interpolate_to_grid(coords, data['observed'][t], Lon, Lat) for t in time_points]
    all_hybrid = [interpolate_to_grid(coords, data['hybrid_predictions'][t], Lon, Lat) for t in time_points]
    
    vmin = min([g.min() for g in all_obs + all_hybrid])
    vmax = max([g.max() for g in all_obs + all_hybrid])
    
    for i, t_idx in enumerate(time_points):
        # Observed (top row)
        ax = axes[0, i]
        plot_gridded_surface(all_obs[i], Lon, Lat, ax=ax,
                            title=f'Day {t_idx}', vmin=vmin, vmax=vmax,
                            colorbar=(i == n_times - 1),
                            show_coords=(i == 0))
        if i == 0:
            ax.set_ylabel('Observed\nLatitude', fontsize=10)
        else:
            ax.set_ylabel('')
            
        # Hybrid predicted (bottom row)
        ax = axes[1, i]
        plot_gridded_surface(all_hybrid[i], Lon, Lat, ax=ax,
                            title='', vmin=vmin, vmax=vmax,
                            colorbar=(i == n_times - 1),
                            show_coords=(i == 0))
        if i == 0:
            ax.set_ylabel('GAM-SSM\nLatitude', fontsize=10)
        else:
            ax.set_ylabel('')
    
    fig.suptitle('Temporal Evolution: Observed (top) vs GAM-SSM Predicted (bottom)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_temporal_sequence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig4_temporal_sequence.png'}")


# =============================================================================
# FIGURE 5: Uncertainty surface
# =============================================================================
def create_uncertainty_surface(data, time_idx=25):
    """Create prediction uncertainty map."""
    print(f"Creating Figure 5: Uncertainty surface (Day {time_idx})...")
    
    Lon, Lat, _, _ = create_interpolation_grid()
    coords = data['coordinates']
    
    # Interpolate mean and uncertainty
    mean_grid = interpolate_to_grid(coords, data['hybrid_predictions'][time_idx], Lon, Lat)
    std_grid = interpolate_to_grid(coords, data['hybrid_std'][time_idx], Lon, Lat)
    
    # Coefficient of variation
    cv_grid = std_grid / mean_grid * 100
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Mean prediction
    plot_gridded_surface(mean_grid, Lon, Lat, ax=axes[0],
                        title='(a) Predicted Mean NO₂',
                        contours=True, n_contours=8)
    
    # Prediction SD
    plot_gridded_surface(std_grid, Lon, Lat, ax=axes[1],
                        title='(b) Prediction Uncertainty (SD)',
                        cmap='YlOrRd',
                        colorbar_label='SD (µg/m³)')
    
    # Coefficient of variation
    plot_gridded_surface(cv_grid, Lon, Lat, ax=axes[2],
                        title='(c) Coefficient of Variation',
                        cmap='YlOrRd', vmin=0, vmax=15,
                        colorbar_label='CV (%)')
    
    fig.suptitle(f'GAM-SSM Prediction with Uncertainty - Day {time_idx}',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_uncertainty_surface.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig5_uncertainty_surface.png'}")


# =============================================================================
# FIGURE 6: RMSE surface (temporal average)
# =============================================================================
def create_rmse_surface(data):
    """Create spatial RMSE surface averaged over time."""
    print("Creating Figure 6: RMSE surface (temporal average)...")
    
    Lon, Lat, _, _ = create_interpolation_grid()
    coords = data['coordinates']
    
    # Compute RMSE at each location across all times
    gam_rmse = np.sqrt(np.mean((data['observed'] - data['gam_predictions'])**2, axis=0))
    hybrid_rmse = np.sqrt(np.mean((data['observed'] - data['hybrid_predictions'])**2, axis=0))
    
    # Interpolate
    gam_rmse_grid = interpolate_to_grid(coords, gam_rmse, Lon, Lat)
    hybrid_rmse_grid = interpolate_to_grid(coords, hybrid_rmse, Lon, Lat)
    
    # Improvement
    improvement_grid = gam_rmse_grid - hybrid_rmse_grid
    pct_improvement_grid = improvement_grid / gam_rmse_grid * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    vmax_rmse = max(gam_rmse_grid.max(), hybrid_rmse_grid.max())
    
    # GAM RMSE
    plot_gridded_surface(gam_rmse_grid, Lon, Lat, ax=axes[0, 0],
                        title=f'(a) GAM-LUR RMSE\nMean: {gam_rmse.mean():.2f} µg/m³',
                        cmap='YlOrRd', vmin=0, vmax=vmax_rmse,
                        colorbar_label='RMSE (µg/m³)')
    
    # Hybrid RMSE
    plot_gridded_surface(hybrid_rmse_grid, Lon, Lat, ax=axes[0, 1],
                        title=f'(b) GAM-SSM RMSE\nMean: {hybrid_rmse.mean():.2f} µg/m³',
                        cmap='YlOrRd', vmin=0, vmax=vmax_rmse,
                        colorbar_label='RMSE (µg/m³)')
    
    # Absolute improvement
    vmax_imp = np.abs(improvement_grid).max()
    plot_gridded_surface(improvement_grid, Lon, Lat, ax=axes[1, 0],
                        title='(c) RMSE Reduction (GAM − Hybrid)',
                        cmap='RdYlGn', vmin=-vmax_imp, vmax=vmax_imp,
                        colorbar_label='RMSE Reduction (µg/m³)',
                        contours=True, n_contours=6)
    
    # Percentage improvement
    plot_gridded_surface(pct_improvement_grid, Lon, Lat, ax=axes[1, 1],
                        title=f'(d) Percentage Improvement\nMean: {(gam_rmse.mean()-hybrid_rmse.mean())/gam_rmse.mean()*100:.1f}%',
                        cmap='RdYlGn', vmin=-50, vmax=50,
                        colorbar_label='Improvement (%)',
                        contours=True, n_contours=6)
    
    fig.suptitle('Spatial Distribution of Model Error',
                fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_rmse_surface.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig6_rmse_surface.png'}")


# =============================================================================
# FIGURE 7: High-resolution single panel (publication main figure)
# =============================================================================
def create_publication_main_figure(data, time_idx=25):
    """Create high-resolution main figure for publication."""
    print(f"Creating Figure 7: Publication main figure (Day {time_idx})...")
    
    # Higher resolution grid
    lon_grid = np.linspace(DUBLIN_BOUNDS['lon_min'], DUBLIN_BOUNDS['lon_max'], 300)
    lat_grid = np.linspace(DUBLIN_BOUNDS['lat_min'], DUBLIN_BOUNDS['lat_max'], 300)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)
    
    coords = data['coordinates']
    
    # Interpolate hybrid prediction
    hybrid_grid = interpolate_to_grid(coords, data['hybrid_predictions'][time_idx], 
                                      Lon, Lat, method='rbf', smooth=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Main surface
    im = ax.pcolormesh(Lon, Lat, hybrid_grid, cmap='viridis', shading='auto', rasterized=True)
    
    # Contours
    levels = np.arange(10, 55, 5)
    cs = ax.contour(Lon, Lat, hybrid_grid, levels=levels, colors='white', 
                   linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.0f')
    
    # Add monitoring locations as small dots
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=8, alpha=0.6, 
              edgecolors='white', linewidths=0.3, label='Monitoring sites', zorder=5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('NO₂ Concentration (µg/m³)', fontsize=12)
    
    # Labels
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Predicted NO₂ Surface - Hybrid GAM-SSM Model (Day {time_idx})',
                fontsize=14, fontweight='bold')
    
    # Add scale bar (approximate)
    # 1 degree longitude ≈ 65 km at Dublin's latitude
    scale_lon = DUBLIN_BOUNDS['lon_min'] + 0.05
    scale_lat = DUBLIN_BOUNDS['lat_min'] + 0.02
    scale_length = 0.08  # ~5 km
    ax.plot([scale_lon, scale_lon + scale_length], [scale_lat, scale_lat], 
           'k-', linewidth=3)
    ax.text(scale_lon + scale_length/2, scale_lat + 0.01, '5 km', 
           ha='center', fontsize=10, fontweight='bold')
    
    # Add north arrow
    arrow_lon = DUBLIN_BOUNDS['lon_max'] - 0.05
    arrow_lat = DUBLIN_BOUNDS['lat_max'] - 0.03
    ax.annotate('N', xy=(arrow_lon, arrow_lat), fontsize=12, fontweight='bold',
               ha='center')
    ax.annotate('', xy=(arrow_lon, arrow_lat - 0.01), 
               xytext=(arrow_lon, arrow_lat - 0.04),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.legend(loc='lower right', fontsize=10)
    ax.set_aspect('equal')
    
    plt.savefig(OUTPUT_DIR / 'fig7_publication_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig7_publication_main.png'}")


# =============================================================================
# FIGURE 8: Summary comparison dashboard
# =============================================================================
def create_summary_dashboard(data):
    """Create comprehensive summary dashboard."""
    print("Creating Figure 8: Summary dashboard...")
    
    Lon, Lat, _, _ = create_interpolation_grid()
    coords = data['coordinates']
    t_idx = 25
    
    # Interpolate key surfaces
    obs_grid = interpolate_to_grid(coords, data['observed'][t_idx], Lon, Lat)
    gam_grid = interpolate_to_grid(coords, data['gam_predictions'][t_idx], Lon, Lat)
    hybrid_grid = interpolate_to_grid(coords, data['hybrid_predictions'][t_idx], Lon, Lat)
    
    gam_rmse = np.sqrt(np.mean((data['observed'] - data['gam_predictions'])**2, axis=0))
    hybrid_rmse = np.sqrt(np.mean((data['observed'] - data['hybrid_predictions'])**2, axis=0))
    improvement = gam_rmse - hybrid_rmse
    improvement_grid = interpolate_to_grid(coords, improvement, Lon, Lat)
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.25, wspace=0.25)
    
    # Row 1: Surface maps
    ax1 = fig.add_subplot(gs[0, 0])
    vmin, vmax = obs_grid.min(), obs_grid.max()
    plot_gridded_surface(obs_grid, Lon, Lat, ax=ax1, title='(a) Observed',
                        vmin=vmin, vmax=vmax, colorbar=False)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_gridded_surface(gam_grid, Lon, Lat, ax=ax2, title='(b) GAM-LUR',
                        vmin=vmin, vmax=vmax, colorbar=False)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_gridded_surface(hybrid_grid, Lon, Lat, ax=ax3, title='(c) GAM-SSM',
                        vmin=vmin, vmax=vmax)
    
    ax4 = fig.add_subplot(gs[0, 3])
    vmax_imp = np.abs(improvement_grid).max()
    plot_gridded_surface(improvement_grid, Lon, Lat, ax=ax4,
                        title='(d) RMSE Improvement',
                        cmap='RdYlGn', vmin=-vmax_imp, vmax=vmax_imp,
                        colorbar_label='Improvement (µg/m³)')
    
    # Row 2: Time series and scatter
    ax5 = fig.add_subplot(gs[1, :2])
    loc_id = 150
    t = np.arange(data['n_days'])
    ax5.fill_between(t,
                    data['hybrid_predictions'][:, loc_id] - 1.96*data['hybrid_std'][:, loc_id],
                    data['hybrid_predictions'][:, loc_id] + 1.96*data['hybrid_std'][:, loc_id],
                    alpha=0.3, color='#4C72B0')
    ax5.scatter(t, data['observed'][:, loc_id], s=20, c='black', alpha=0.6, label='Observed')
    ax5.plot(t, data['gam_predictions'][:, loc_id], '--', color='#DD8452', lw=2, label='GAM-LUR')
    ax5.plot(t, data['hybrid_predictions'][:, loc_id], '-', color='#4C72B0', lw=2, label='GAM-SSM')
    ax5.set_xlabel('Day')
    ax5.set_ylabel('NO₂ (µg/m³)')
    ax5.set_title('(e) Temporal Evolution with 95% CI', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    obs_flat = data['observed'].flatten()
    gam_flat = data['gam_predictions'].flatten()
    ax6.scatter(obs_flat, gam_flat, alpha=0.1, s=2, c='#DD8452')
    ax6.plot([obs_flat.min(), obs_flat.max()], [obs_flat.min(), obs_flat.max()], 'k--', lw=2)
    gam_r2 = 1 - np.sum((obs_flat - gam_flat)**2) / np.sum((obs_flat - obs_flat.mean())**2)
    ax6.set_title(f'(f) GAM-LUR: R²={gam_r2:.3f}', fontweight='bold')
    ax6.set_xlabel('Observed')
    ax6.set_ylabel('Predicted')
    ax6.set_aspect('equal')
    
    ax7 = fig.add_subplot(gs[1, 3])
    hybrid_flat = data['hybrid_predictions'].flatten()
    ax7.scatter(obs_flat, hybrid_flat, alpha=0.1, s=2, c='#4C72B0')
    ax7.plot([obs_flat.min(), obs_flat.max()], [obs_flat.min(), obs_flat.max()], 'k--', lw=2)
    hybrid_r2 = 1 - np.sum((obs_flat - hybrid_flat)**2) / np.sum((obs_flat - obs_flat.mean())**2)
    ax7.set_title(f'(g) GAM-SSM: R²={hybrid_r2:.3f}', fontweight='bold')
    ax7.set_xlabel('Observed')
    ax7.set_ylabel('Predicted')
    ax7.set_aspect('equal')
    
    # Row 3: Metrics and residuals
    ax8 = fig.add_subplot(gs[2, 0])
    metrics = ['RMSE', 'MAE', 'R²']
    gam_metrics = [
        np.sqrt(np.mean((obs_flat - gam_flat)**2)),
        np.mean(np.abs(obs_flat - gam_flat)),
        gam_r2
    ]
    hybrid_metrics = [
        np.sqrt(np.mean((obs_flat - hybrid_flat)**2)),
        np.mean(np.abs(obs_flat - hybrid_flat)),
        hybrid_r2
    ]
    x = np.arange(len(metrics))
    width = 0.35
    ax8.bar(x - width/2, gam_metrics, width, label='GAM-LUR', color='#DD8452')
    ax8.bar(x + width/2, hybrid_metrics, width, label='GAM-SSM', color='#4C72B0')
    ax8.set_xticks(x)
    ax8.set_xticklabels(metrics)
    ax8.set_title('(h) Model Comparison', fontweight='bold')
    ax8.legend()
    
    ax9 = fig.add_subplot(gs[2, 1])
    gam_resid = obs_flat - gam_flat
    hybrid_resid = obs_flat - hybrid_flat
    ax9.hist(gam_resid, bins=50, alpha=0.6, color='#DD8452', label='GAM-LUR', density=True)
    ax9.hist(hybrid_resid, bins=50, alpha=0.6, color='#4C72B0', label='GAM-SSM', density=True)
    ax9.axvline(x=0, color='black', linestyle='--')
    ax9.set_xlabel('Residual (µg/m³)')
    ax9.set_title('(i) Residual Distributions', fontweight='bold')
    ax9.legend()
    
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    overall_improvement = (gam_metrics[0] - hybrid_metrics[0]) / gam_metrics[0] * 100
    summary = (
        f"SUMMARY\n"
        f"{'='*40}\n\n"
        f"Dataset: {data['n_locations']} locations × {data['n_days']} days\n"
        f"Grid resolution: {GRID_RESOLUTION} × {GRID_RESOLUTION}\n\n"
        f"GAM-LUR Performance:\n"
        f"  RMSE: {gam_metrics[0]:.2f} µg/m³\n"
        f"  MAE:  {gam_metrics[1]:.2f} µg/m³\n"
        f"  R²:   {gam_metrics[2]:.3f}\n\n"
        f"GAM-SSM Performance:\n"
        f"  RMSE: {hybrid_metrics[0]:.2f} µg/m³\n"
        f"  MAE:  {hybrid_metrics[1]:.2f} µg/m³\n"
        f"  R²:   {hybrid_metrics[2]:.3f}\n\n"
        f"IMPROVEMENT: {overall_improvement:.1f}% RMSE reduction\n"
        f"R² increase: +{hybrid_metrics[2] - gam_metrics[2]:.3f}"
    )
    ax10.text(0.1, 0.95, summary, transform=ax10.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    fig.suptitle('GAM-LUR vs Hybrid GAM-SSM: Comprehensive Comparison',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_DIR / 'fig8_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig8_summary_dashboard.png'}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("Gridded Surface Maps for LUR Visualization")
    print("=" * 60)
    print()
    
    # Generate data
    print("Generating synthetic spatiotemporal NO₂ data...")
    data = generate_realistic_data(n_locations=300, n_days=50, seed=42)
    
    # Compute metrics
    obs_flat = data['observed'].flatten()
    gam_flat = data['gam_predictions'].flatten()
    hybrid_flat = data['hybrid_predictions'].flatten()
    
    gam_rmse = np.sqrt(np.mean((obs_flat - gam_flat)**2))
    hybrid_rmse = np.sqrt(np.mean((obs_flat - hybrid_flat)**2))
    
    print(f"\nModel Performance:")
    print(f"  GAM-LUR:  RMSE = {gam_rmse:.2f} µg/m³")
    print(f"  GAM-SSM:  RMSE = {hybrid_rmse:.2f} µg/m³")
    print(f"  Improvement: {(gam_rmse - hybrid_rmse) / gam_rmse * 100:.1f}%")
    print()
    
    # Create all figures
    print("Generating gridded surface maps...")
    create_three_panel_comparison(data)
    create_six_panel_with_residuals(data)
    create_difference_map(data)
    create_temporal_sequence(data)
    create_uncertainty_surface(data)
    create_rmse_surface(data)
    create_publication_main_figure(data)
    create_summary_dashboard(data)
    
    print()
    print("=" * 60)
    print(f"All gridded maps saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
