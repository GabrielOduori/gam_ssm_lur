#!/usr/bin/env python
"""
Standalone Visualization Demo: GAM-LUR vs Hybrid GAM-SSM-LUR.

This script generates comprehensive publication-ready figures comparing
classic LUR with the hybrid GAM-SSM approach.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.interpolate import griddata

from output_utils import make_experiment_dirs

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

# Colorblind-friendly colors
COLORS = {
    'gam': '#DD8452',      # Orange for GAM-LUR
    'hybrid': '#4C72B0',   # Blue for GAM-SSM
    'observed': '#2d2d2d', # Dark gray for observed
}

OUTPUT_ROOT, OUTPUT_GROUPS = make_experiment_dirs(
    base="outputs",
    groups=["figures"],
)
OUTPUT_DIR = OUTPUT_GROUPS["figures"]


def generate_realistic_data(n_locations=200, n_days=50, seed=42):
    """Generate realistic spatiotemporal NO2 data."""
    np.random.seed(seed)
    
    # Dublin-like coordinates
    lon = -6.26 + np.random.randn(n_locations) * 0.08
    lat = 53.35 + np.random.randn(n_locations) * 0.04
    coordinates = np.column_stack([lon, lat])
    
    # Spatial features (simplified)
    motorway_dist = np.random.exponential(1000, n_locations)
    industrial = np.random.exponential(500, n_locations)
    traffic = 5000 + 10000 * np.exp(-motorway_dist/500) + np.random.randn(n_locations) * 500
    
    # TRUE SPATIAL COMPONENT (what GAM should capture)
    true_spatial = (
        30 +  # baseline
        -0.005 * motorway_dist +  # closer to motorway = higher
        0.01 * industrial +  # more industrial = higher
        0.001 * traffic +  # more traffic = higher
        3 * np.sin(lon * 50) +  # non-linear spatial pattern
        2 * np.cos(lat * 80)
    )
    true_spatial = np.clip(true_spatial, 5, 60)
    
    # TRUE TEMPORAL COMPONENT (what SSM should capture)
    t = np.arange(n_days)
    diurnal_like = 5 * np.sin(2 * np.pi * t / 7)  # weekly cycle
    ar_process = np.zeros(n_days)
    ar_process[0] = np.random.randn() * 2
    for i in range(1, n_days):
        ar_process[i] = 0.85 * ar_process[i-1] + np.random.randn() * 1.5
    true_temporal = diurnal_like + ar_process
    
    # Generate observations: spatial + temporal + noise
    observed = np.zeros((n_days, n_locations))
    for t_idx in range(n_days):
        for s_idx in range(n_locations):
            observed[t_idx, s_idx] = (
                true_spatial[s_idx] + 
                true_temporal[t_idx] + 
                np.random.randn() * 3  # measurement noise
            )
    observed = np.clip(observed, 0, 80)
    
    # SIMULATE GAM-LUR PREDICTIONS (captures spatial, misses temporal)
    # GAM captures spatial pattern but treats temporal as noise
    gam_spatial_fit = true_spatial + np.random.randn(n_locations) * 1.5  # some fitting error
    gam_predictions = np.tile(gam_spatial_fit, (n_days, 1))  # same prediction each day
    # Add small random variation (as if time-varying covariates helped slightly)
    gam_predictions += np.random.randn(n_days, n_locations) * 2
    
    # SIMULATE HYBRID GAM-SSM PREDICTIONS (captures both)
    # GAM component
    hybrid_spatial = true_spatial + np.random.randn(n_locations) * 1.0
    # SSM component (captures temporal dynamics)
    hybrid_temporal = np.zeros(n_days)
    # Kalman-filtered temporal estimate
    for t_idx in range(n_days):
        if t_idx == 0:
            hybrid_temporal[t_idx] = true_temporal[t_idx] + np.random.randn() * 1
        else:
            # Smooth estimate combining prediction and observation
            pred = 0.85 * hybrid_temporal[t_idx-1]
            obs_mean = (observed[t_idx] - hybrid_spatial).mean()
            hybrid_temporal[t_idx] = 0.7 * pred + 0.3 * obs_mean
    
    hybrid_predictions = np.zeros((n_days, n_locations))
    hybrid_std = np.zeros((n_days, n_locations))
    for t_idx in range(n_days):
        hybrid_predictions[t_idx] = hybrid_spatial + hybrid_temporal[t_idx]
        # Uncertainty (higher at edges, lower in center)
        base_std = 1.5 + 0.5 * np.abs(hybrid_temporal[t_idx] - hybrid_temporal.mean()) / hybrid_temporal.std()
        hybrid_std[t_idx] = base_std + np.random.rand(n_locations) * 0.5
    
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


def compute_metrics(observed, predicted):
    """Compute evaluation metrics."""
    obs = observed.flatten()
    pred = predicted.flatten()
    residuals = obs - pred
    
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    r2 = 1 - ss_res / ss_tot
    corr = np.corrcoef(obs, pred)[0, 1]
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr}


# =============================================================================
# FIGURE 1: Spatial Maps Comparison
# =============================================================================
def create_spatial_comparison(data):
    """Create side-by-side spatial maps."""
    print("Creating Figure 1: Spatial Maps Comparison...")
    
    coords = data['coordinates']
    t_idx = data['n_days'] // 2  # Middle time point
    
    observed = data['observed'][t_idx]
    gam_pred = data['gam_predictions'][t_idx]
    hybrid_pred = data['hybrid_predictions'][t_idx]
    
    # Residuals
    gam_resid = observed - gam_pred
    hybrid_resid = observed - hybrid_pred
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.25)
    
    # Common color limits
    vmin_conc = min(observed.min(), gam_pred.min(), hybrid_pred.min())
    vmax_conc = max(observed.max(), gam_pred.max(), hybrid_pred.max())
    vmax_res = max(np.abs(gam_resid).max(), np.abs(hybrid_resid).max())
    
    # Row 1: Concentrations
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=observed, cmap='viridis',
                      s=30, vmin=vmin_conc, vmax=vmax_conc, alpha=0.8)
    ax1.set_title('(a) Observed NO₂', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(sc1, ax=ax1, label='NO₂ (µg/m³)', shrink=0.8)
    
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(coords[:, 0], coords[:, 1], c=gam_pred, cmap='viridis',
                      s=30, vmin=vmin_conc, vmax=vmax_conc, alpha=0.8)
    ax2.set_title('(b) GAM-LUR Predicted', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(sc2, ax=ax2, label='NO₂ (µg/m³)', shrink=0.8)
    
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(coords[:, 0], coords[:, 1], c=hybrid_pred, cmap='viridis',
                      s=30, vmin=vmin_conc, vmax=vmax_conc, alpha=0.8)
    ax3.set_title('(c) GAM-SSM Predicted', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    plt.colorbar(sc3, ax=ax3, label='NO₂ (µg/m³)', shrink=0.8)
    
    # Row 2: Residuals
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    # Add text summary
    gam_metrics = compute_metrics(data['observed'], data['gam_predictions'])
    hybrid_metrics = compute_metrics(data['observed'], data['hybrid_predictions'])
    summary = (
        f"Model Performance Summary (Day {t_idx})\n"
        f"{'='*35}\n\n"
        f"GAM-LUR:\n"
        f"  RMSE: {gam_metrics['rmse']:.2f} µg/m³\n"
        f"  R²:   {gam_metrics['r2']:.3f}\n\n"
        f"GAM-SSM (Hybrid):\n"
        f"  RMSE: {hybrid_metrics['rmse']:.2f} µg/m³\n"
        f"  R²:   {hybrid_metrics['r2']:.3f}\n\n"
        f"Improvement: {(gam_metrics['rmse']-hybrid_metrics['rmse'])/gam_metrics['rmse']*100:.1f}%"
    )
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax5 = fig.add_subplot(gs[1, 1])
    sc5 = ax5.scatter(coords[:, 0], coords[:, 1], c=gam_resid, cmap='RdBu_r',
                      s=30, vmin=-vmax_res, vmax=vmax_res, alpha=0.8)
    ax5.set_title('(d) GAM-LUR Residuals', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    plt.colorbar(sc5, ax=ax5, label='Residual (µg/m³)', shrink=0.8)
    
    ax6 = fig.add_subplot(gs[1, 2])
    sc6 = ax6.scatter(coords[:, 0], coords[:, 1], c=hybrid_resid, cmap='RdBu_r',
                      s=30, vmin=-vmax_res, vmax=vmax_res, alpha=0.8)
    ax6.set_title('(e) GAM-SSM Residuals', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    plt.colorbar(sc6, ax=ax6, label='Residual (µg/m³)', shrink=0.8)
    
    fig.suptitle(f'Spatial Comparison: GAM-LUR vs Hybrid GAM-SSM (Day {t_idx})', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(OUTPUT_DIR / 'fig1_spatial_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig1_spatial_comparison.png'}")


# =============================================================================
# FIGURE 2: Observed vs Predicted Scatter Plots
# =============================================================================
def create_scatter_comparison(data):
    """Create observed vs predicted scatter plots."""
    print("Creating Figure 2: Observed vs Predicted...")
    
    obs = data['observed'].flatten()
    gam_pred = data['gam_predictions'].flatten()
    hybrid_pred = data['hybrid_predictions'].flatten()
    
    gam_metrics = compute_metrics(data['observed'], data['gam_predictions'])
    hybrid_metrics = compute_metrics(data['observed'], data['hybrid_predictions'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    vmin, vmax = min(obs.min(), gam_pred.min(), hybrid_pred.min()), max(obs.max(), gam_pred.max(), hybrid_pred.max())
    
    # GAM-LUR
    ax = axes[0]
    ax.scatter(obs, gam_pred, alpha=0.15, s=5, c=COLORS['gam'], edgecolors='none')
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=2, label='1:1 line')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel('Observed NO₂ (µg/m³)', fontsize=11)
    ax.set_ylabel('Predicted NO₂ (µg/m³)', fontsize=11)
    ax.set_title(f"(a) GAM-LUR Only\nR² = {gam_metrics['r2']:.3f}, RMSE = {gam_metrics['rmse']:.2f} µg/m³",
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Hybrid GAM-SSM
    ax = axes[1]
    ax.scatter(obs, hybrid_pred, alpha=0.15, s=5, c=COLORS['hybrid'], edgecolors='none')
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=2, label='1:1 line')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel('Observed NO₂ (µg/m³)', fontsize=11)
    ax.set_ylabel('Predicted NO₂ (µg/m³)', fontsize=11)
    ax.set_title(f"(b) Hybrid GAM-SSM\nR² = {hybrid_metrics['r2']:.3f}, RMSE = {hybrid_metrics['rmse']:.2f} µg/m³",
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_observed_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig2_observed_vs_predicted.png'}")


# =============================================================================
# FIGURE 3: Model Performance Metrics Comparison
# =============================================================================
def create_metrics_comparison(data):
    """Create bar chart comparing metrics."""
    print("Creating Figure 3: Metrics Comparison...")
    
    gam_metrics = compute_metrics(data['observed'], data['gam_predictions'])
    hybrid_metrics = compute_metrics(data['observed'], data['hybrid_predictions'])
    
    metrics = ['RMSE\n(µg/m³)', 'MAE\n(µg/m³)', 'R²', 'Correlation']
    gam_vals = [gam_metrics['rmse'], gam_metrics['mae'], gam_metrics['r2'], gam_metrics['corr']]
    hybrid_vals = [hybrid_metrics['rmse'], hybrid_metrics['mae'], hybrid_metrics['r2'], hybrid_metrics['corr']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, gam_vals, width, label='GAM-LUR', color=COLORS['gam'], edgecolor='black')
    bars2 = ax.bar(x + width/2, hybrid_vals, width, label='GAM-SSM', color=COLORS['hybrid'], edgecolor='black')
    
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Add improvement annotations
    for i, (g, h) in enumerate(zip(gam_vals, hybrid_vals)):
        if i < 2:  # RMSE and MAE - lower is better
            improvement = (g - h) / g * 100
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'{improvement:+.1f}%', xy=(i, max(g, h) * 1.1),
                       ha='center', fontsize=10, fontweight='bold', color=color)
        else:  # R² and Correlation - higher is better
            improvement = h - g
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'{improvement:+.3f}', xy=(i, max(g, h) * 1.1),
                       ha='center', fontsize=10, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig3_metrics_comparison.png'}")


# =============================================================================
# FIGURE 4: Temporal Evolution at Multiple Locations
# =============================================================================
def create_temporal_evolution(data):
    """Create temporal evolution plots with uncertainty."""
    print("Creating Figure 4: Temporal Evolution...")
    
    n_locs = 6
    loc_ids = np.linspace(0, data['n_locations']-1, n_locs, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    t = np.arange(data['n_days'])
    
    for i, loc_id in enumerate(loc_ids):
        ax = axes[i]
        
        obs = data['observed'][:, loc_id]
        gam_pred = data['gam_predictions'][:, loc_id]
        hybrid_pred = data['hybrid_predictions'][:, loc_id]
        hybrid_std = data['hybrid_std'][:, loc_id]
        
        # 95% CI
        lower = hybrid_pred - 1.96 * hybrid_std
        upper = hybrid_pred + 1.96 * hybrid_std
        
        # Plot
        ax.fill_between(t, lower, upper, alpha=0.3, color=COLORS['hybrid'], label='95% CI')
        ax.scatter(t, obs, s=15, c=COLORS['observed'], alpha=0.6, label='Observed', zorder=3)
        ax.plot(t, gam_pred, '--', color=COLORS['gam'], linewidth=1.5, label='GAM-LUR', alpha=0.8)
        ax.plot(t, hybrid_pred, '-', color=COLORS['hybrid'], linewidth=2, label='GAM-SSM')
        
        ax.set_xlabel('Day')
        ax.set_ylabel('NO₂ (µg/m³)')
        ax.set_title(f'({chr(97+i)}) Location {loc_id}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    fig.suptitle('Temporal Evolution: GAM-LUR vs Hybrid GAM-SSM with Uncertainty', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_temporal_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig4_temporal_evolution.png'}")


# =============================================================================
# FIGURE 5: Residual Distributions Comparison
# =============================================================================
def create_residual_comparison(data):
    """Compare residual distributions."""
    print("Creating Figure 5: Residual Comparison...")
    
    gam_resid = (data['observed'] - data['gam_predictions']).flatten()
    hybrid_resid = (data['observed'] - data['hybrid_predictions']).flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histograms
    ax = axes[0, 0]
    bins = np.linspace(min(gam_resid.min(), hybrid_resid.min()),
                      max(gam_resid.max(), hybrid_resid.max()), 50)
    ax.hist(gam_resid, bins=bins, alpha=0.6, color=COLORS['gam'], 
           label=f'GAM-LUR (SD={gam_resid.std():.2f})', density=True)
    ax.hist(hybrid_resid, bins=bins, alpha=0.6, color=COLORS['hybrid'],
           label=f'GAM-SSM (SD={hybrid_resid.std():.2f})', density=True)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Residual (µg/m³)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Residual Distributions', fontweight='bold')
    ax.legend()
    
    # Q-Q plot for GAM
    ax = axes[0, 1]
    stats.probplot(gam_resid, dist="norm", plot=ax)
    ax.get_lines()[0].set_color(COLORS['gam'])
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color('red')
    ax.set_title('(b) Q-Q Plot: GAM-LUR', fontweight='bold')
    
    # Q-Q plot for Hybrid
    ax = axes[1, 0]
    stats.probplot(hybrid_resid, dist="norm", plot=ax)
    ax.get_lines()[0].set_color(COLORS['hybrid'])
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color('red')
    ax.set_title('(c) Q-Q Plot: GAM-SSM', fontweight='bold')
    
    # Residuals vs fitted
    ax = axes[1, 1]
    ax.scatter(data['gam_predictions'].flatten(), gam_resid, alpha=0.1, s=3,
              c=COLORS['gam'], label='GAM-LUR')
    ax.scatter(data['hybrid_predictions'].flatten(), hybrid_resid, alpha=0.1, s=3,
              c=COLORS['hybrid'], label='GAM-SSM')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted values (µg/m³)')
    ax.set_ylabel('Residual (µg/m³)')
    ax.set_title('(d) Residuals vs Fitted', fontweight='bold')
    ax.legend(markerscale=5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_residual_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig5_residual_comparison.png'}")


# =============================================================================
# FIGURE 6: Spatial RMSE Map
# =============================================================================
def create_rmse_map(data):
    """Create spatial RMSE distribution map."""
    print("Creating Figure 6: Spatial RMSE Map...")
    
    coords = data['coordinates']
    
    # RMSE at each location
    gam_rmse_per_loc = np.sqrt(np.mean((data['observed'] - data['gam_predictions'])**2, axis=0))
    hybrid_rmse_per_loc = np.sqrt(np.mean((data['observed'] - data['hybrid_predictions'])**2, axis=0))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    vmin = min(gam_rmse_per_loc.min(), hybrid_rmse_per_loc.min())
    vmax = max(gam_rmse_per_loc.max(), hybrid_rmse_per_loc.max())
    
    ax = axes[0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=gam_rmse_per_loc, cmap='YlOrRd',
                   s=40, vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='gray', linewidths=0.5)
    ax.set_title(f'(a) GAM-LUR RMSE\nMean: {gam_rmse_per_loc.mean():.2f} µg/m³', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(sc, ax=ax, label='RMSE (µg/m³)', shrink=0.8)
    
    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=hybrid_rmse_per_loc, cmap='YlOrRd',
                   s=40, vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='gray', linewidths=0.5)
    ax.set_title(f'(b) GAM-SSM RMSE\nMean: {hybrid_rmse_per_loc.mean():.2f} µg/m³', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(sc, ax=ax, label='RMSE (µg/m³)', shrink=0.8)
    
    fig.suptitle('Spatial Distribution of Prediction Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_rmse_spatial_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig6_rmse_spatial_map.png'}")


# =============================================================================
# FIGURE 7: Temporal Variance Evolution
# =============================================================================
def create_variance_evolution(data):
    """Show how variance is reduced by smoothing."""
    print("Creating Figure 7: Variance Evolution...")
    
    # Spatial variance at each time point
    obs_var = np.var(data['observed'], axis=1)
    gam_var = np.var(data['gam_predictions'], axis=1)
    hybrid_var = np.var(data['hybrid_predictions'], axis=1)
    
    # Residual variance
    gam_resid_var = np.var(data['observed'] - data['gam_predictions'], axis=1)
    hybrid_resid_var = np.var(data['observed'] - data['hybrid_predictions'], axis=1)
    
    t = np.arange(data['n_days'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prediction variance
    ax = axes[0]
    ax.plot(t, obs_var, 'k-', linewidth=2, label='Observed', alpha=0.7)
    ax.plot(t, gam_var, '--', color=COLORS['gam'], linewidth=2, label='GAM-LUR')
    ax.plot(t, hybrid_var, '-', color=COLORS['hybrid'], linewidth=2, label='GAM-SSM')
    ax.set_xlabel('Day')
    ax.set_ylabel('Spatial Variance')
    ax.set_title('(a) Spatial Variance Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual variance
    ax = axes[1]
    ax.fill_between(t, 0, gam_resid_var, alpha=0.4, color=COLORS['gam'], label='GAM-LUR residual var')
    ax.fill_between(t, 0, hybrid_resid_var, alpha=0.4, color=COLORS['hybrid'], label='GAM-SSM residual var')
    ax.plot(t, gam_resid_var, '--', color=COLORS['gam'], linewidth=2)
    ax.plot(t, hybrid_resid_var, '-', color=COLORS['hybrid'], linewidth=2)
    ax.set_xlabel('Day')
    ax.set_ylabel('Residual Variance')
    ax.set_title('(b) Residual Variance Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation for variance reduction
    avg_reduction = (gam_resid_var.mean() - hybrid_resid_var.mean()) / gam_resid_var.mean() * 100
    ax.annotate(f'Avg. reduction: {avg_reduction:.1f}%', 
               xy=(0.7, 0.9), xycoords='axes fraction',
               fontsize=11, fontweight='bold', color='green',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_variance_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig7_variance_evolution.png'}")


# =============================================================================
# FIGURE 8: Temporal Snapshots (Animated-style)
# =============================================================================
def create_temporal_snapshots(data):
    """Create spatial snapshots at multiple time points."""
    print("Creating Figure 8: Temporal Snapshots...")
    
    coords = data['coordinates']
    time_points = [0, data['n_days']//4, data['n_days']//2, 3*data['n_days']//4, data['n_days']-1]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Common color limits
    vmin = data['observed'].min()
    vmax = data['observed'].max()
    
    for i, t_idx in enumerate(time_points):
        # Observed (top row)
        ax = axes[0, i]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=data['observed'][t_idx], 
                       cmap='viridis', s=20, vmin=vmin, vmax=vmax, alpha=0.8)
        ax.set_title(f'Day {t_idx}', fontsize=11, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Observed\nLatitude', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Hybrid predicted (bottom row)
        ax = axes[1, i]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=data['hybrid_predictions'][t_idx],
                       cmap='viridis', s=20, vmin=vmin, vmax=vmax, alpha=0.8)
        if i == 0:
            ax.set_ylabel('GAM-SSM\nLatitude', fontsize=10)
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='NO₂ (µg/m³)')
    
    fig.suptitle('Temporal Evolution: Observed (top) vs GAM-SSM Predictions (bottom)', 
                fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_DIR / 'fig8_temporal_snapshots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig8_temporal_snapshots.png'}")


# =============================================================================
# FIGURE 9: Interpolated Surface Maps
# =============================================================================
def create_interpolated_surfaces(data):
    """Create smooth interpolated surface maps."""
    print("Creating Figure 9: Interpolated Surfaces...")
    
    coords = data['coordinates']
    t_idx = data['n_days'] // 2
    
    # Create grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    obs_interp = griddata(coords, data['observed'][t_idx], (Xi, Yi), method='cubic')
    gam_interp = griddata(coords, data['gam_predictions'][t_idx], (Xi, Yi), method='cubic')
    hybrid_interp = griddata(coords, data['hybrid_predictions'][t_idx], (Xi, Yi), method='cubic')
    
    vmin = min(np.nanmin(obs_interp), np.nanmin(gam_interp), np.nanmin(hybrid_interp))
    vmax = max(np.nanmax(obs_interp), np.nanmax(gam_interp), np.nanmax(hybrid_interp))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    ax = axes[0]
    im = ax.pcolormesh(Xi, Yi, obs_interp, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    ax.scatter(coords[:, 0], coords[:, 1], c='white', s=5, alpha=0.3, edgecolors='none')
    ax.set_title('(a) Observed NO₂', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='NO₂ (µg/m³)', shrink=0.8)
    
    ax = axes[1]
    im = ax.pcolormesh(Xi, Yi, gam_interp, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    ax.scatter(coords[:, 0], coords[:, 1], c='white', s=5, alpha=0.3, edgecolors='none')
    ax.set_title('(b) GAM-LUR Predicted', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='NO₂ (µg/m³)', shrink=0.8)
    
    ax = axes[2]
    im = ax.pcolormesh(Xi, Yi, hybrid_interp, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    ax.scatter(coords[:, 0], coords[:, 1], c='white', s=5, alpha=0.3, edgecolors='none')
    ax.set_title('(c) GAM-SSM Predicted', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='NO₂ (µg/m³)', shrink=0.8)
    
    fig.suptitle(f'Interpolated NO₂ Concentration Surfaces (Day {t_idx})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_interpolated_surfaces.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig9_interpolated_surfaces.png'}")


# =============================================================================
# FIGURE 10: Summary Dashboard
# =============================================================================
def create_summary_dashboard(data):
    """Create a comprehensive summary dashboard."""
    print("Creating Figure 10: Summary Dashboard...")
    
    gam_metrics = compute_metrics(data['observed'], data['gam_predictions'])
    hybrid_metrics = compute_metrics(data['observed'], data['hybrid_predictions'])
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    coords = data['coordinates']
    t_idx = data['n_days'] // 2
    
    # Panel A: Spatial map comparison
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(coords[:, 0], coords[:, 1], c=data['observed'][t_idx],
                    cmap='viridis', s=15, alpha=0.8)
    ax1.set_title('(a) Observed', fontweight='bold')
    ax1.set_xlabel('Lon')
    ax1.set_ylabel('Lat')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(coords[:, 0], coords[:, 1], c=data['hybrid_predictions'][t_idx],
               cmap='viridis', s=15, alpha=0.8)
    ax2.set_title('(b) GAM-SSM Predicted', fontweight='bold')
    ax2.set_xlabel('Lon')
    
    # Panel B: Scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    obs_flat = data['observed'].flatten()
    hybrid_flat = data['hybrid_predictions'].flatten()
    ax3.scatter(obs_flat, hybrid_flat, alpha=0.1, s=3, c=COLORS['hybrid'])
    ax3.plot([obs_flat.min(), obs_flat.max()], [obs_flat.min(), obs_flat.max()], 'k--', lw=2)
    ax3.set_xlabel('Observed')
    ax3.set_ylabel('Predicted')
    ax3.set_title(f"(c) Observed vs Predicted\nR²={hybrid_metrics['r2']:.3f}", fontweight='bold')
    ax3.set_aspect('equal')
    
    # Panel C: Time series
    ax4 = fig.add_subplot(gs[1, :2])
    loc_id = data['n_locations'] // 2
    t = np.arange(data['n_days'])
    ax4.fill_between(t, 
                    data['hybrid_predictions'][:, loc_id] - 1.96*data['hybrid_std'][:, loc_id],
                    data['hybrid_predictions'][:, loc_id] + 1.96*data['hybrid_std'][:, loc_id],
                    alpha=0.3, color=COLORS['hybrid'])
    ax4.scatter(t, data['observed'][:, loc_id], s=20, c='black', alpha=0.6, label='Observed')
    ax4.plot(t, data['gam_predictions'][:, loc_id], '--', color=COLORS['gam'], lw=2, label='GAM-LUR')
    ax4.plot(t, data['hybrid_predictions'][:, loc_id], '-', color=COLORS['hybrid'], lw=2, label='GAM-SSM')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('NO₂ (µg/m³)')
    ax4.set_title(f'(d) Temporal Evolution (Location {loc_id})', fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Panel D: Metrics comparison
    ax5 = fig.add_subplot(gs[1, 2])
    metrics = ['RMSE', 'MAE', 'R²']
    gam_vals = [gam_metrics['rmse'], gam_metrics['mae'], gam_metrics['r2']]
    hybrid_vals = [hybrid_metrics['rmse'], hybrid_metrics['mae'], hybrid_metrics['r2']]
    x = np.arange(len(metrics))
    width = 0.35
    ax5.bar(x - width/2, gam_vals, width, label='GAM-LUR', color=COLORS['gam'])
    ax5.bar(x + width/2, hybrid_vals, width, label='GAM-SSM', color=COLORS['hybrid'])
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.set_title('(e) Model Comparison', fontweight='bold')
    ax5.legend()
    
    # Panel E: Residual histograms
    ax6 = fig.add_subplot(gs[2, 0])
    gam_resid = (data['observed'] - data['gam_predictions']).flatten()
    hybrid_resid = (data['observed'] - data['hybrid_predictions']).flatten()
    ax6.hist(gam_resid, bins=40, alpha=0.6, color=COLORS['gam'], label='GAM-LUR', density=True)
    ax6.hist(hybrid_resid, bins=40, alpha=0.6, color=COLORS['hybrid'], label='GAM-SSM', density=True)
    ax6.axvline(x=0, color='black', linestyle='--')
    ax6.set_xlabel('Residual (µg/m³)')
    ax6.set_title('(f) Residual Distributions', fontweight='bold')
    ax6.legend()
    
    # Panel F: RMSE map
    ax7 = fig.add_subplot(gs[2, 1])
    hybrid_rmse = np.sqrt(np.mean((data['observed'] - data['hybrid_predictions'])**2, axis=0))
    sc = ax7.scatter(coords[:, 0], coords[:, 1], c=hybrid_rmse, cmap='YlOrRd', s=20, alpha=0.8)
    ax7.set_title('(g) Spatial RMSE (GAM-SSM)', fontweight='bold')
    ax7.set_xlabel('Lon')
    ax7.set_ylabel('Lat')
    plt.colorbar(sc, ax=ax7, label='RMSE', shrink=0.8)
    
    # Panel G: Summary text
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    improvement = (gam_metrics['rmse'] - hybrid_metrics['rmse']) / gam_metrics['rmse'] * 100
    summary_text = (
        f"SUMMARY\n"
        f"{'='*25}\n\n"
        f"Locations: {data['n_locations']}\n"
        f"Time steps: {data['n_days']}\n\n"
        f"GAM-LUR:\n"
        f"  RMSE: {gam_metrics['rmse']:.2f} µg/m³\n"
        f"  R²:   {gam_metrics['r2']:.3f}\n\n"
        f"GAM-SSM:\n"
        f"  RMSE: {hybrid_metrics['rmse']:.2f} µg/m³\n"
        f"  R²:   {hybrid_metrics['r2']:.3f}\n\n"
        f"RMSE Improvement:\n"
        f"  {improvement:.1f}%"
    )
    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    fig.suptitle('GAM-LUR vs Hybrid GAM-SSM: Comprehensive Comparison', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(OUTPUT_DIR / 'fig10_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig10_summary_dashboard.png'}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("GAM-LUR vs Hybrid GAM-SSM Visualization Demo")
    print("=" * 60)
    print()
    
    # Generate data
    print("Generating synthetic spatiotemporal NO₂ data...")
    data = generate_realistic_data(n_locations=200, n_days=50, seed=42)
    
    # Compute and print metrics
    gam_metrics = compute_metrics(data['observed'], data['gam_predictions'])
    hybrid_metrics = compute_metrics(data['observed'], data['hybrid_predictions'])
    
    print(f"\nModel Performance:")
    print(f"  GAM-LUR:  RMSE={gam_metrics['rmse']:.2f}, R²={gam_metrics['r2']:.3f}")
    print(f"  GAM-SSM:  RMSE={hybrid_metrics['rmse']:.2f}, R²={hybrid_metrics['r2']:.3f}")
    print(f"  Improvement: {(gam_metrics['rmse']-hybrid_metrics['rmse'])/gam_metrics['rmse']*100:.1f}%")
    print()
    
    # Create all figures
    print("Generating figures...")
    create_spatial_comparison(data)
    create_scatter_comparison(data)
    create_metrics_comparison(data)
    create_temporal_evolution(data)
    create_residual_comparison(data)
    create_rmse_map(data)
    create_variance_evolution(data)
    create_temporal_snapshots(data)
    create_interpolated_surfaces(data)
    create_summary_dashboard(data)
    
    print()
    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
