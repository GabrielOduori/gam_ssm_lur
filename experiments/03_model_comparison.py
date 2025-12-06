#!/usr/bin/env python
"""
Model Comparison Example: GAM-LUR vs Hybrid GAM-SSM-LUR.

This script demonstrates a comprehensive comparison between:
1. Static GAM-LUR (baseline)
2. Hybrid GAM-SSM-LUR (proposed method)

Generates publication-ready figures showing:
- Spatial maps of predictions and residuals
- Temporal evolution with uncertainty
- Model performance comparisons
- Convergence diagnostics
- Residual analysis

Run with:
    python experiments/03_model_comparison.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from output_utils import make_experiment_dirs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_dublin_like_data(
    n_locations: int = 200,
    n_days: int = 50,
    random_state: int = 42,
):
    """Generate synthetic spatiotemporal NO₂ data resembling Dublin.
    
    Creates realistic patterns including:
    - Spatial heterogeneity from land use (motorways, industrial areas)
    - Temporal dynamics (diurnal, weekly cycles, AR process)
    - Measurement noise
    
    Returns
    -------
    dict with:
        - coordinates: (n_locations, 2) array of (lon, lat)
        - features: DataFrame of spatial features
        - observations: DataFrame with timestamp, location_id, no2, etc.
        - true_spatial: true spatial component
        - true_temporal: true temporal component
    """
    np.random.seed(random_state)
    
    logger.info(f"Generating synthetic data: {n_locations} locations × {n_days} days")
    
    # Dublin-like coordinates (centered around Dublin)
    lon_center, lat_center = -6.26, 53.35
    coordinates = np.column_stack([
        lon_center + np.random.randn(n_locations) * 0.08,
        lat_center + np.random.randn(n_locations) * 0.04,
    ])
    
    # Generate spatial features
    n_features = 20
    feature_names = [
        'motorway_50m', 'motorway_100m', 'motorway_500m', 'motorway_1000m',
        'primary_50m', 'primary_100m', 'primary_500m', 'primary_1000m',
        'secondary_100m', 'secondary_500m',
        'industrial_100m', 'industrial_500m', 'industrial_1000m',
        'residential_100m', 'residential_500m',
        'commercial_100m', 'commercial_500m',
        'motorway_distance', 'traffic_volume', 'tropomi_no2'
    ]
    
    # Create correlated features (as in real LUR data)
    features = pd.DataFrame(index=range(n_locations))
    
    # Road features (exponentially distributed, spatially correlated)
    for name in feature_names[:8]:
        base = np.random.exponential(scale=300, size=n_locations)
        features[name] = base + np.random.randn(n_locations) * 50
        features[name] = features[name].clip(lower=0)
        
    # Other features
    for name in feature_names[8:17]:
        features[name] = np.random.exponential(scale=500, size=n_locations)
        
    # Distance features
    features['motorway_distance'] = np.random.exponential(scale=1000, size=n_locations)
    
    # Traffic volume (higher near motorways)
    features['traffic_volume'] = 5000 + 10000 * np.exp(-features['motorway_distance']/500) + np.random.randn(n_locations) * 500
    
    # TROPOMI (satellite)
    features['tropomi_no2'] = 20 + np.random.randn(n_locations) * 5
    
    features['location_id'] = range(n_locations)
    
    # TRUE SPATIAL COMPONENT
    # Realistic LUR-style relationships
    true_spatial = (
        0.02 * features['motorway_50m'].values +
        0.015 * features['motorway_100m'].values +
        0.01 * features['primary_50m'].values +
        0.008 * features['industrial_100m'].values +
        -0.002 * features['motorway_distance'].values +
        0.0005 * features['traffic_volume'].values +
        0.3 * features['tropomi_no2'].values
    )
    
    # Add non-linear effects
    true_spatial += 2 * np.sin(features['motorway_50m'].values / 200)
    true_spatial += -0.00001 * features['motorway_distance'].values**2
    
    # Normalize to realistic NO2 range (10-50 µg/m³)
    true_spatial = (true_spatial - true_spatial.mean()) / true_spatial.std() * 8 + 25
    
    # TRUE TEMPORAL COMPONENT
    n_hours = n_days * 24
    t = np.arange(n_hours)
    
    # Diurnal cycle (peak at 8am and 6pm for traffic)
    diurnal = 5 * np.sin(2 * np.pi * t / 24 - np.pi/3) + 3 * np.sin(4 * np.pi * t / 24)
    
    # Weekly cycle (lower on weekends)
    day_of_week = (t // 24) % 7
    weekly = np.where(day_of_week >= 5, -3, 0)  # Weekend reduction
    
    # AR(1) process for weather-driven variability
    ar_coef = 0.85
    ar_process = np.zeros(n_hours)
    ar_process[0] = np.random.randn()
    for i in range(1, n_hours):
        ar_process[i] = ar_coef * ar_process[i-1] + np.random.randn() * 2
        
    true_temporal = diurnal + weekly + ar_process
    
    # Subsample to daily for manageable computation
    # Take daily mean
    true_temporal_daily = true_temporal.reshape(n_days, 24).mean(axis=1)
    
    # GENERATE OBSERVATIONS
    timestamps = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    obs_list = []
    for t_idx, ts in enumerate(timestamps):
        for s_idx in range(n_locations):
            # True signal
            signal = true_spatial[s_idx] + true_temporal_daily[t_idx]
            
            # Add measurement noise
            noise = np.random.randn() * 2.5
            
            no2 = max(0, signal + noise)  # Non-negative
            
            obs_list.append({
                'timestamp': ts,
                'time_idx': t_idx,
                'location_id': s_idx,
                'no2': no2,
                'lon': coordinates[s_idx, 0],
                'lat': coordinates[s_idx, 1],
            })
            
    observations = pd.DataFrame(obs_list)
    
    logger.info(f"Generated {len(observations)} observations")
    logger.info(f"NO2 range: {observations['no2'].min():.1f} - {observations['no2'].max():.1f} µg/m³")
    
    return {
        'coordinates': coordinates,
        'features': features,
        'observations': observations,
        'true_spatial': true_spatial,
        'true_temporal': true_temporal_daily,
        'n_locations': n_locations,
        'n_days': n_days,
    }


def fit_and_compare_models(data: dict, output_dir: Path):
    """Fit both models and generate comprehensive comparison.
    
    Parameters
    ----------
    data : dict
        Output from generate_dublin_like_data()
    output_dir : Path
        Directory to save outputs
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from gam_ssm_lur import HybridGAMSSM
    from gam_ssm_lur.spatial_gam import SpatialGAM
    from gam_ssm_lur.features import FeatureSelector
    from gam_ssm_lur.visualization import (
        SpatialVisualizer,
        TemporalVisualizer,
        ModelComparisonVisualizer,
        DiagnosticsVisualizer,
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    obs = data['observations']
    features = data['features']
    coordinates = data['coordinates']
    
    # Prepare feature matrix
    feature_cols = [c for c in features.columns if c != 'location_id']
    
    # Merge features with observations
    df = obs.merge(features, on='location_id')
    
    X = df[feature_cols].values
    y = df['no2'].values
    time_idx = df['time_idx'].values
    loc_idx = df['location_id'].values
    
    # =========================================================================
    # Feature Selection
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Feature Selection")
    logger.info("=" * 60)
    
    selector = FeatureSelector(
        correlation_threshold=0.85,
        vif_threshold=10.0,
        n_top_features=12,
        force_keep=['traffic_volume', 'tropomi_no2', 'motorway_distance'],
        random_state=42,
    )
    
    X_df = pd.DataFrame(X, columns=feature_cols)
    X_selected = selector.fit_transform(X_df, y)
    selected_features = list(X_selected.columns)
    
    logger.info(f"Selected {len(selected_features)} features:")
    for f in selected_features:
        logger.info(f"  - {f}")
        
    # =========================================================================
    # Fit GAM-only Model (Baseline)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Fitting GAM-LUR Baseline")
    logger.info("=" * 60)
    
    gam_model = SpatialGAM(n_splines=8, lam='auto')
    gam_model.fit(X_selected, y, feature_names=selected_features)
    
    gam_pred = gam_model.predict(X_selected)
    gam_residuals = y - gam_pred
    
    # Compute GAM metrics
    gam_rmse = np.sqrt(np.mean(gam_residuals**2))
    gam_mae = np.mean(np.abs(gam_residuals))
    gam_ss_res = np.sum(gam_residuals**2)
    gam_ss_tot = np.sum((y - np.mean(y))**2)
    gam_r2 = 1 - gam_ss_res / gam_ss_tot
    gam_corr = np.corrcoef(y, gam_pred)[0, 1]
    
    gam_metrics = {
        'rmse': gam_rmse,
        'mae': gam_mae,
        'r2': gam_r2,
        'corr': gam_corr,
    }
    
    logger.info(f"GAM-LUR Results:")
    logger.info(f"  RMSE: {gam_rmse:.4f} µg/m³")
    logger.info(f"  MAE:  {gam_mae:.4f} µg/m³")
    logger.info(f"  R²:   {gam_r2:.4f}")
    logger.info(f"  Corr: {gam_corr:.4f}")
    
    # =========================================================================
    # Fit Hybrid GAM-SSM Model
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Fitting Hybrid GAM-SSM Model")
    logger.info("=" * 60)
    
    hybrid_model = HybridGAMSSM(
        n_splines=8,
        gam_lam='auto',
        em_max_iter=30,
        em_tol=1e-5,
        scalability_mode='auto',
        confidence_level=0.95,
        random_state=42,
    )
    
    hybrid_model.fit(
        X=X_selected,
        y=y,
        time_index=time_idx,
        location_index=loc_idx,
    )
    
    hybrid_pred = hybrid_model.predict()
    hybrid_metrics = hybrid_model.evaluate(
        y_true=hybrid_model._y_matrix.flatten(),
        y_pred=hybrid_pred.total.flatten(),
        y_lower=hybrid_pred.lower.flatten(),
        y_upper=hybrid_pred.upper.flatten(),
    )
    
    logger.info(f"GAM-SSM Results:")
    logger.info(f"  RMSE:     {hybrid_metrics['rmse']:.4f} µg/m³")
    logger.info(f"  MAE:      {hybrid_metrics['mae']:.4f} µg/m³")
    logger.info(f"  R²:       {hybrid_metrics['r2']:.4f}")
    logger.info(f"  Corr:     {hybrid_metrics['correlation']:.4f}")
    logger.info(f"  Coverage: {hybrid_metrics['coverage_95']:.1%}")
    
    # Improvement
    rmse_improvement = (gam_rmse - hybrid_metrics['rmse']) / gam_rmse * 100
    logger.info(f"\nRMSE Improvement: {rmse_improvement:.1f}%")
    
    # =========================================================================
    # Reshape predictions to matrices for visualization
    # =========================================================================
    n_times = data['n_days']
    n_locs = data['n_locations']
    
    y_matrix = y.reshape(n_times, n_locs)
    gam_pred_matrix = gam_pred.reshape(n_times, n_locs)
    
    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Generating Visualizations")
    logger.info("=" * 60)
    
    # Initialize visualizers
    spatial_viz = SpatialVisualizer(coordinates, coordinate_system='latlon')
    temporal_viz = TemporalVisualizer()
    comparison_viz = ModelComparisonVisualizer()
    diagnostics_viz = DiagnosticsVisualizer()
    
    # Figure 1: Spatial comparison at a single time point
    logger.info("Creating spatial comparison figure...")
    fig = spatial_viz.plot_comparison_grid(
        observed=y_matrix,
        gam_predicted=gam_pred_matrix,
        hybrid_predicted=hybrid_pred.total,
        time_idx=n_times // 2,
        suptitle=f'Spatial Comparison (Day {n_times//2})',
        save_path=output_dir / 'fig1_spatial_comparison.png'
    )
    plt.close(fig)
    
    # Figure 2: Observed vs Predicted scatter plots
    logger.info("Creating observed vs predicted figure...")
    fig = comparison_viz.plot_observed_vs_predicted(
        observed=y_matrix,
        gam_predicted=gam_pred_matrix,
        hybrid_predicted=hybrid_pred.total,
        save_path=output_dir / 'fig2_observed_vs_predicted.png'
    )
    plt.close(fig)
    
    # Figure 3: Metric comparison bar chart
    logger.info("Creating metrics comparison figure...")
    fig = comparison_viz.plot_metric_comparison(
        gam_metrics=gam_metrics,
        hybrid_metrics=hybrid_metrics,
        save_path=output_dir / 'fig3_metric_comparison.png'
    )
    plt.close(fig)
    
    # Figure 4: Residual comparison
    logger.info("Creating residual comparison figure...")
    fig = comparison_viz.plot_residual_comparison(
        observed=y_matrix,
        gam_predicted=gam_pred_matrix,
        hybrid_predicted=hybrid_pred.total,
        save_path=output_dir / 'fig4_residual_comparison.png'
    )
    plt.close(fig)
    
    # Figure 5: Temporal evolution at multiple locations
    logger.info("Creating temporal evolution figure...")
    fig = temporal_viz.plot_multi_location_timeseries(
        observed=y_matrix,
        predicted=hybrid_pred.total,
        lower=hybrid_pred.lower,
        upper=hybrid_pred.upper,
        n_locations=6,
        suptitle='Temporal Evolution with 95% Prediction Intervals',
        save_path=output_dir / 'fig5_temporal_evolution.png'
    )
    plt.close(fig)
    
    # Figure 6: EM convergence
    logger.info("Creating EM convergence figure...")
    em_history = hybrid_model.get_em_convergence()
    fig = diagnostics_viz.plot_em_convergence(
        log_likelihoods=em_history['log_likelihood'].tolist(),
        param_traces={
            'tr(T)': em_history['tr_T'].tolist(),
            'tr(Q)': em_history['tr_Q'].tolist(),
            'tr(H)': em_history['tr_H'].tolist(),
        },
        save_path=output_dir / 'fig6_em_convergence.png'
    )
    plt.close(fig)
    
    # Figure 7: Full residual diagnostics for hybrid model
    logger.info("Creating residual diagnostics figure...")
    hybrid_residuals = y_matrix - hybrid_pred.total
    fig = diagnostics_viz.plot_residual_diagnostics(
        residuals=hybrid_residuals,
        fitted=hybrid_pred.total,
        time_index=np.arange(n_times),
        save_path=output_dir / 'fig7_residual_diagnostics.png'
    )
    plt.close(fig)
    
    # Figure 8: Variance evolution
    logger.info("Creating variance evolution figure...")
    fig = temporal_viz.plot_variance_evolution(
        observed=y_matrix,
        smoothed=hybrid_pred.total,
        title='Temporal Variance: Observed vs Kalman-Smoothed',
        save_path=output_dir / 'fig8_variance_evolution.png'
    )
    plt.close(fig)
    
    # Figure 9: RMSE spatial map
    logger.info("Creating RMSE map figure...")
    fig = spatial_viz.plot_rmse_map(
        observed=y_matrix,
        predicted=hybrid_pred.total,
        title='Spatial RMSE Distribution (GAM-SSM)',
        save_path=output_dir / 'fig9_rmse_map.png'
    )
    plt.close(fig)
    
    # Figure 10: Uncertainty map
    logger.info("Creating uncertainty map figure...")
    mean_pred = hybrid_pred.total.mean(axis=0)
    mean_std = hybrid_pred.std.mean(axis=0)
    fig = spatial_viz.plot_uncertainty_map(
        mean=mean_pred,
        std=mean_std,
        title='Prediction Uncertainty Map',
        save_path=output_dir / 'fig10_uncertainty_map.png'
    )
    plt.close(fig)
    
    # Figure 11: Temporal snapshots
    logger.info("Creating temporal snapshots figure...")
    fig = spatial_viz.plot_temporal_snapshots(
        values=hybrid_pred.total,
        time_indices=[0, n_times//4, n_times//2, 3*n_times//4, n_times-1],
        time_labels=['Day 1', f'Day {n_times//4}', f'Day {n_times//2}', 
                    f'Day {3*n_times//4}', f'Day {n_times}'],
        title='NO₂ Predictions Over Time',
        save_path=output_dir / 'fig11_temporal_snapshots.png'
    )
    plt.close(fig)
    
    # =========================================================================
    # Save Results Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Saving Results Summary")
    logger.info("=" * 60)
    
    results_df = pd.DataFrame({
        'Metric': ['RMSE (µg/m³)', 'MAE (µg/m³)', 'R²', 'Correlation', '95% Coverage'],
        'GAM-LUR': [f"{gam_rmse:.3f}", f"{gam_mae:.3f}", f"{gam_r2:.3f}", 
                   f"{gam_corr:.3f}", "N/A"],
        'GAM-SSM': [f"{hybrid_metrics['rmse']:.3f}", f"{hybrid_metrics['mae']:.3f}",
                   f"{hybrid_metrics['r2']:.3f}", f"{hybrid_metrics['correlation']:.3f}",
                   f"{hybrid_metrics['coverage_95']:.1%}"],
        'Improvement': [f"{rmse_improvement:.1f}%", 
                       f"{(gam_mae - hybrid_metrics['mae'])/gam_mae*100:.1f}%",
                       f"+{hybrid_metrics['r2'] - gam_r2:.3f}", 
                       f"+{hybrid_metrics['correlation'] - gam_corr:.3f}",
                       "—"]
    })
    
    results_df.to_csv(output_dir / 'model_comparison_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"Figures saved to: {output_dir}")
    print("=" * 60)
    
    return {
        'gam_metrics': gam_metrics,
        'hybrid_metrics': hybrid_metrics,
        'improvement': rmse_improvement,
    }


def main():
    """Run the model comparison example."""
    
    _, output_groups = make_experiment_dirs(base="outputs", groups=["model_comparison"])
    output_dir = output_groups["model_comparison"]
    
    logger.info("=" * 70)
    logger.info("GAM-LUR vs Hybrid GAM-SSM-LUR Comparison")
    logger.info("=" * 70)
    
    # Generate synthetic data
    data = generate_dublin_like_data(
        n_locations=150,
        n_days=50,
        random_state=42,
    )
    
    # Fit models and generate comparisons
    results = fit_and_compare_models(data, output_dir)
    
    logger.info("=" * 70)
    logger.info("Comparison completed!")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
