#!/usr/bin/env python
"""
Reproduce Paper Results: GAM-SSM-LUR for Dublin NO₂ Prediction.

This script reproduces the main results from:
    "Hybrid Generalized Additive-State Space Modelling for Urban NO₂ Prediction:
     Integrating Spatial and Temporal Dynamics"
    
Usage:
    python examples/reproduce_paper.py --data-dir data/ --output-dir results/

Data Requirements:
    Download from Zenodo: https://zenodo.org/uploads/16534138
    Expected files:
        - dublin_no2_2023.csv: Hourly NO₂ observations
        - spatial_features.csv: Land use and road network features
        - epa_stations.csv: Validation station locations
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from output_utils import make_experiment_dirs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default prefixes for spatial/traffic feature columns in merged datasets.
# This list includes the provided columns plus a few common variants to stay backwards compatible.
BASE_FEATURE_PREFIXES = [
    "traffic_volume",
    "scats_distance",
    "motorway",
    "industrial",
    "industria",       # handle possible shortened column name
    "traffic_signals", # if present
    # Legacy/extra prefixes retained for compatibility
    "commercial",
    "residential",
    "primary",
    "secondary",
    "trunk",
    "tertiary",
]


def setup_paths(data_dir: Path, output_base: Path, run_name: str | None) -> dict:
    """Set up file paths with a unique experiment folder to avoid overwriting."""
    output_root, groups = make_experiment_dirs(
        base=output_base,
        groups=['figures', 'models', 'tables'],
        run_name=run_name,
    )
    paths = {
        'data_dir': data_dir,
        'output_dir': output_root,
        'figures_dir': groups['figures'],
        'models_dir': groups['models'],
        'tables_dir': groups['tables'],
    }
    return paths


def _infer_feature_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    """Infer feature columns based on known prefixes."""
    candidates = {col for prefix in prefixes for col in df.columns if col.startswith(prefix)}
    return sorted(candidates)


def _clean_merged_dataframe(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    fallback_target_col: str | None,
    location_col: str,
    lat_col: str,
    lon_col: str,
    feature_prefixes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Clean merged dataset to handle missing/inf values and infer features."""
    required = {timestamp_col, target_col, location_col, lat_col, lon_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in merged data: {', '.join(sorted(missing))}")
    
    df = df.copy()
    
    # Clean target, optionally filling missing values from fallback column
    if fallback_target_col and fallback_target_col in df.columns:
        df[target_col] = df[target_col].fillna(df[fallback_target_col])
    
    df = df.dropna(subset=[target_col])
    df = df[~np.isinf(df[target_col])]
    df.loc[df[target_col] < 0, target_col] = 0
    if df.empty:
        raise ValueError(
            f"No rows remaining after cleaning target column '{target_col}'. "
            "Check that the column exists and has non-null values, or set --fallback-target-col."
        )
    
    # Clean coordinates
    for coord in [lon_col, lat_col]:
        if coord in df.columns:
            df = df.dropna(subset=[coord])
            df = df[~np.isinf(df[coord])]
    
    # Feature list (based on prefixes)
    feature_cols = _infer_feature_columns(df, feature_prefixes)
    # Optional satellite NO2 as feature if not the target column
    if "no2_values" in df.columns and target_col != "no2_values":
        feature_cols.append("no2_values")
    if not feature_cols:
        raise ValueError(
            "No feature columns found using provided prefixes. "
            "Check column names or extend BASE_FEATURE_PREFIXES."
        )
    
    # Clean features (replace inf/NaN with median or 0)
    clean_features = []
    for f in feature_cols:
        if f not in df.columns:
            continue
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
        median = df[f].median()
        df[f] = df[f].fillna(0 if pd.isna(median) else median)
        clean_features.append(f)
    
    return df, clean_features


def load_data(
    data_dir: Path,
    data_file: Path | None,
    timestamp_col: str,
    target_col: str,
    fallback_target_col: str | None,
    location_col: str,
    lat_col: str,
    lon_col: str,
) -> tuple:
    """Load Dublin NO₂ dataset, supporting both merged and separate inputs.
    
    Returns
    -------
    observations : pd.DataFrame
        NO₂ observations with columns: timestamp, location_id, no2, lat, lon
    features : pd.DataFrame
        Spatial features for each location
    validation_stations : pd.DataFrame
        EPA validation station information (None if not provided)
    """
    logger.info("Loading data...")
    
    # Path 1: user-provided merged table
    if data_file:
        if not data_file.exists():
            raise FileNotFoundError(f"Merged data file not found: {data_file}")
        
        logger.info(f"Using merged data file: {data_file}")
        
        df = pd.read_csv(data_file, parse_dates=[timestamp_col])
        df, feature_cols = _clean_merged_dataframe(
            df=df,
            timestamp_col=timestamp_col,
            target_col=target_col,
            fallback_target_col=fallback_target_col,
            location_col=location_col,
            lat_col=lat_col,
            lon_col=lon_col,
            feature_prefixes=BASE_FEATURE_PREFIXES,
        )
        
        # Normalize column names for downstream code
        df = df.rename(columns={
            timestamp_col: "timestamp",
            target_col: "no2",
            location_col: "location_id",
            lat_col: "lat",
            lon_col: "lon",
        })
        
        observations = df[["timestamp", "location_id", "no2", "lat", "lon"]].copy()
        
        # Keep only the inferred/cleaned feature columns
        selected_feature_cols = [c for c in feature_cols if c in df.columns and c not in {"timestamp", "no2"}]
        features = df[["location_id"] + selected_feature_cols].drop_duplicates(subset=["location_id"]).reset_index(drop=True)
        
        logger.info(f"Loaded merged data: {len(observations)} observations across {len(features)} locations")
        validation_stations = None
        return observations, features, validation_stations
    
    # Path 2: separate files (default demo setup)
    obs_path = data_dir / "dublin_no2_2023.csv"
    feat_path = data_dir / "spatial_features.csv"
    val_path = data_dir / "epa_stations.csv"
    
    if obs_path.exists() and feat_path.exists():
        observations = pd.read_csv(obs_path, parse_dates=['timestamp'])
        features = pd.read_csv(feat_path)
        validation_stations = pd.read_csv(val_path) if val_path.exists() else None
        logger.info(f"Loaded {len(observations)} observations, {len(features)} feature sets")
    else:
        logger.warning("Real data not found. Generating synthetic data for demonstration.")
        observations, features, validation_stations = generate_demo_data()
        
    return observations, features, validation_stations


def generate_demo_data() -> tuple:
    """Generate synthetic data matching paper characteristics."""
    np.random.seed(42)
    
    n_locations = 100  # Subset for demo
    n_days = 50
    n_hours = n_days * 24
    
    # Generate location grid
    locations = pd.DataFrame({
        'location_id': range(n_locations),
        'lat': 53.3 + np.random.randn(n_locations) * 0.05,
        'lon': -6.26 + np.random.randn(n_locations) * 0.08,
    })
    
    # Generate spatial features (simplified)
    n_features = 56
    feature_names = (
        [f'motorway_{d}m' for d in [50, 100, 200, 500, 1000]] +
        [f'primary_{d}m' for d in [50, 100, 200, 500, 1000]] +
        [f'secondary_{d}m' for d in [50, 100, 200, 500, 1000]] +
        [f'tertiary_{d}m' for d in [50, 100, 200, 500, 1000]] +
        [f'residential_{d}m' for d in [50, 100, 200, 500, 1000]] +
        [f'industrial_{d}m' for d in [100, 200, 500, 1000]] +
        [f'commercial_{d}m' for d in [100, 200, 500, 1000]] +
        ['motorway_distance', 'primary_distance', 'industrial_distance'] +
        ['traffic_volume', 'traffic_distance'] +
        ['tropomi_no2'] +
        [f'extra_feat_{i}' for i in range(n_features - 43)]
    )
    
    features = pd.DataFrame(
        np.random.exponential(scale=500, size=(n_locations, len(feature_names))),
        columns=feature_names
    )
    features['location_id'] = range(n_locations)
    
    # Generate observations with realistic patterns
    timestamps = pd.date_range('2023-01-01', periods=n_hours, freq='H')
    
    # Spatial baseline
    spatial_effect = (
        0.02 * features['motorway_50m'].values +
        0.01 * features['industrial_100m'].values -
        0.005 * features['motorway_distance'].values
    )
    spatial_effect = (spatial_effect - spatial_effect.mean()) * 5 + 20
    
    # Temporal pattern (diurnal + weekly + trend)
    hour_effect = 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24 - np.pi/2)  # Peak at 8am
    day_effect = 2 * np.sin(2 * np.pi * np.arange(n_hours) / (24*7))  # Weekly cycle
    ar_process = np.zeros(n_hours)
    ar_process[0] = np.random.randn()
    for t in range(1, n_hours):
        ar_process[t] = 0.9 * ar_process[t-1] + np.random.randn() * 0.5
    temporal_effect = hour_effect + day_effect + ar_process
    
    # Generate observations
    obs_list = []
    for t, ts in enumerate(timestamps):
        for s in range(n_locations):
            no2 = spatial_effect[s] + temporal_effect[t] + np.random.randn() * 2
            no2 = max(0, no2)  # Non-negative
            obs_list.append({
                'timestamp': ts,
                'location_id': s,
                'no2': no2,
                'lat': locations.loc[s, 'lat'],
                'lon': locations.loc[s, 'lon'],
            })
    
    observations = pd.DataFrame(obs_list)
    
    # Validation stations (8 EPA sites)
    val_idx = np.random.choice(n_locations, 8, replace=False)
    validation_stations = locations.iloc[val_idx].copy()
    validation_stations['station_name'] = [f'EPA_Station_{i}' for i in range(8)]
    
    return observations, features, validation_stations


def fit_baseline_gam(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """Fit static GAM-LUR baseline model."""
    from gam_ssm_lur.spatial_gam import SpatialGAM
    
    logger.info("Fitting baseline GAM-LUR model...")
    
    gam = SpatialGAM(n_splines=10, lam='auto')
    gam.fit(X, y, feature_names=feature_names)
    
    y_pred = gam.predict(X)
    
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    corr = np.corrcoef(y, y_pred)[0, 1]
    
    return {
        'model': gam,
        'predictions': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': corr,
    }


def fit_hybrid_model(
    X: np.ndarray,
    y: np.ndarray,
    time_idx: np.ndarray,
    loc_idx: np.ndarray,
    feature_names: list,
) -> dict:
    """Fit hybrid GAM-SSM model."""
    from gam_ssm_lur import HybridGAMSSM
    
    logger.info("Fitting hybrid GAM-SSM model...")
    
    model = HybridGAMSSM(
        n_splines=10,
        gam_lam='auto',
        em_max_iter=50,
        em_tol=1e-6,
        scalability_mode='auto',
        confidence_level=0.95,
        random_state=42,
    )
    
    model.fit(
        X=pd.DataFrame(X, columns=feature_names),
        y=y,
        time_index=time_idx,
        location_index=loc_idx,
    )
    
    predictions = model.predict()
    metrics = model.evaluate(
        y_true=model._y_matrix.flatten(),
        y_pred=predictions.total.flatten(),
        y_lower=predictions.lower.flatten(),
        y_upper=predictions.upper.flatten(),
    )
    
    return {
        'model': model,
        'predictions': predictions,
        'metrics': metrics,
    }


def create_comparison_table(baseline: dict, hybrid: dict, output_path: Path) -> pd.DataFrame:
    """Create Table 2: Model comparison (GAM-only vs GAM-SSM)."""
    
    table = pd.DataFrame({
        'Metric': ['RMSE (µg/m³)', 'MAE (µg/m³)', 'R²', 'Correlation'],
        'GAM-only': [
            f"{baseline['rmse']:.3f}",
            f"{baseline['mae']:.3f}",
            f"{baseline['r2']:.3f}",
            f"{baseline['correlation']:.3f}",
        ],
        'GAM-SSM': [
            f"{hybrid['metrics']['rmse']:.3f}",
            f"{hybrid['metrics']['mae']:.3f}",
            f"{hybrid['metrics']['r2']:.3f}",
            f"{hybrid['metrics']['correlation']:.3f}",
        ],
    })
    
    # Calculate improvement
    rmse_improvement = (baseline['rmse'] - hybrid['metrics']['rmse']) / baseline['rmse'] * 100
    table['Improvement'] = [f"{rmse_improvement:.1f}%", '', '', '']
    
    table.to_csv(output_path, index=False)
    logger.info(f"Saved comparison table to {output_path}")
    
    return table


def create_convergence_plot(hybrid_result: dict, output_path: Path):
    """Create Figure 6: EM convergence diagnostics."""
    
    model = hybrid_result['model']
    em_history = model.get_em_convergence()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Log-likelihood
    ax = axes[0, 0]
    ax.plot(em_history['iteration'], em_history['log_likelihood'], 'b-o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-likelihood')
    ax.set_title('(a) Log-likelihood')
    ax.grid(True, alpha=0.3)
    
    # Parameter traces
    ax = axes[0, 1]
    ax.plot(em_history['iteration'], em_history['tr_T'], 'b-', label='tr(T)')
    ax.plot(em_history['iteration'], em_history['tr_Q'], 'r-', label='tr(Q)')
    ax.plot(em_history['iteration'], em_history['tr_H'], 'g-', label='tr(H)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter trace')
    ax.set_title('(b) Parameter traces')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log-likelihood increment
    ax = axes[1, 0]
    ll = em_history['log_likelihood'].values
    ll_change = np.abs(np.diff(ll))
    ax.semilogy(range(1, len(ll)), ll_change, 'b-o', markersize=4)
    ax.axhline(y=1e-6, color='r', linestyle='--', label='Convergence threshold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|ΔLL|')
    ax.set_title('(c) Log-likelihood increment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final diagnostics text
    ax = axes[1, 1]
    ax.axis('off')
    diagnostics = model.ssm_.get_diagnostics()
    text = (
        f"EM Convergence Summary\n"
        f"{'='*30}\n\n"
        f"Converged: {diagnostics.em_converged}\n"
        f"Iterations: {diagnostics.em_iterations}\n"
        f"Final log-likelihood: {diagnostics.log_likelihood:.2e}\n"
        f"AIC: {diagnostics.aic:.2e}\n"
        f"BIC: {diagnostics.bic:.2e}\n\n"
        f"Process noise var: {diagnostics.process_noise_variance:.4f}\n"
        f"Observation noise var: {diagnostics.observation_noise_variance:.4f}\n"
    )
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    ax.set_title('(d) Summary')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved convergence plot to {output_path}")
    plt.close()


def create_residual_diagnostics(hybrid_result: dict, output_path: Path):
    """Create Figure 8: Residual diagnostics."""
    from scipy import stats
    
    model = hybrid_result['model']
    predictions = hybrid_result['predictions']
    
    y_true = model._y_matrix.flatten()
    y_pred = predictions.total.flatten()
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Residuals vs fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.1, s=1)
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
    
    # Temporal residual pattern
    ax = axes[1, 0]
    residual_matrix = residuals.reshape(model.n_times_, model.n_locations_)
    daily_mean = residual_matrix.mean(axis=1)
    daily_std = residual_matrix.std(axis=1)
    t = np.arange(len(daily_mean))
    ax.plot(t, daily_mean, 'b-', lw=1)
    ax.fill_between(t, daily_mean - daily_std, daily_mean + daily_std, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean residual')
    ax.set_title('(d) Temporal Residual Pattern')
    
    # Spatial residual pattern
    ax = axes[1, 1]
    spatial_rmse = np.sqrt((residual_matrix**2).mean(axis=0))
    ax.hist(spatial_rmse, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=spatial_rmse.mean(), color='r', linestyle='--', label=f'Mean: {spatial_rmse.mean():.3f}')
    ax.set_xlabel('Location RMSE')
    ax.set_ylabel('Frequency')
    ax.set_title('(e) Spatial RMSE Distribution')
    ax.legend()
    
    # ACF of residuals
    ax = axes[1, 2]
    from statsmodels.graphics.tsaplots import plot_acf
    # Use mean residuals across space
    try:
        plot_acf(daily_mean, lags=30, ax=ax, alpha=0.05)
        ax.set_title('(f) Residual ACF')
    except:
        # Fallback if statsmodels not available
        nlags = 30
        acf = np.correlate(daily_mean - daily_mean.mean(), daily_mean - daily_mean.mean(), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf[:nlags+1] / acf[0]
        ax.bar(range(nlags+1), acf, width=0.3, color='blue', alpha=0.7)
        ax.axhline(y=0, color='black')
        ax.axhline(y=1.96/np.sqrt(len(daily_mean)), color='r', linestyle='--')
        ax.axhline(y=-1.96/np.sqrt(len(daily_mean)), color='r', linestyle='--')
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title('(f) Residual ACF')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved residual diagnostics to {output_path}")
    plt.close()


def create_observed_vs_predicted_plot(baseline: dict, hybrid_result: dict, output_path: Path):
    """Create Figure 7: Observed vs predicted comparison."""
    
    model = hybrid_result['model']
    predictions = hybrid_result['predictions']
    
    y_true = model._y_matrix.flatten()
    y_pred_hybrid = predictions.total.flatten()
    y_pred_baseline = baseline['predictions']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline
    ax = axes[0]
    ax.scatter(y_true, y_pred_baseline, alpha=0.1, s=1)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Observed NO₂ (µg/m³)')
    ax.set_ylabel('Predicted NO₂ (µg/m³)')
    ax.set_title(f"(a) GAM-only (R² = {baseline['r2']:.3f})")
    ax.set_aspect('equal')
    
    # Hybrid
    ax = axes[1]
    ax.scatter(y_true, y_pred_hybrid, alpha=0.1, s=1)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    corr = np.corrcoef(y_true, y_pred_hybrid)[0, 1]
    ax.set_xlabel('Observed NO₂ (µg/m³)')
    ax.set_ylabel('Predicted NO₂ (µg/m³)')
    ax.set_title(f"(b) GAM-SSM (R² = {hybrid_result['metrics']['r2']:.3f}, r = {corr:.3f})")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved observed vs predicted plot to {output_path}")
    plt.close()


def main():
    """Reproduce paper results."""
    
    parser = argparse.ArgumentParser(description='Reproduce GAM-SSM-LUR paper results')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Directory containing input data')
    parser.add_argument('--data-file', type=Path, default=None,
                        help='Path to merged CSV containing timestamp, target, location, coords, and all spatial features')
    parser.add_argument('--timestamp-col', type=str, default='timestamp',
                        help='Timestamp column in merged data')
    parser.add_argument('--target-col', type=str, default='epa_no2',
                        help='Target column (e.g., gold-standard EPA NO₂) in merged data')
    parser.add_argument('--location-col', type=str, default='grid_id',
                        help='Location ID column in merged data')
    parser.add_argument('--lat-col', type=str, default='latitude',
                        help='Latitude column in merged data')
    parser.add_argument('--lon-col', type=str, default='longitude',
                        help='Longitude column in merged data')
    parser.add_argument('--fallback-target-col', type=str, default='no2_values',
                        help='Optional column to fill missing target values (e.g., satellite NO₂)')
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                        help='Base directory for output files (each run gets a timestamped subfolder)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional custom experiment folder name (defaults to experiment_<timestamp>)')
    args = parser.parse_args()
    
    # Setup
    paths = setup_paths(args.data_dir, args.output_dir, args.run_name)
    
    logger.info("=" * 70)
    logger.info("GAM-SSM-LUR Paper Reproduction")
    logger.info("=" * 70)
    
    # Add src to path for development
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    # Load data
    observations, features, validation_stations = load_data(
        data_dir=args.data_dir,
        data_file=args.data_file,
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        fallback_target_col=args.fallback_target_col,
        location_col=args.location_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
    )
    
    # Prepare arrays
    logger.info("Preparing data arrays...")
    
    # Merge observations with features
    df = observations.merge(features, on='location_id')
    
    feature_cols = [c for c in features.columns if c != 'location_id']
    X = df[feature_cols].values
    y = df['no2'].values
    time_idx = df['timestamp'].factorize()[0]  # Convert to integer indices
    loc_idx = df['location_id'].values
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Feature selection
    logger.info("=" * 70)
    logger.info("Feature Selection")
    logger.info("=" * 70)
    
    from gam_ssm_lur.features import FeatureSelector
    
    selector = FeatureSelector(
        correlation_threshold=0.8,
        vif_threshold=10.0,
        n_top_features=30,
        force_keep=['traffic_volume', 'tropomi_no2'] if 'traffic_volume' in feature_cols else None,
        random_state=42,
    )
    
    X_df = pd.DataFrame(X, columns=feature_cols)
    X_selected = selector.fit_transform(X_df, y)
    selected_features = list(X_selected.columns)
    
    logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)}")
    
    # Fit baseline
    logger.info("=" * 70)
    logger.info("Fitting Baseline GAM-LUR")
    logger.info("=" * 70)
    
    baseline = fit_baseline_gam(X_selected.values, y, selected_features)
    logger.info(f"Baseline RMSE: {baseline['rmse']:.4f}")
    logger.info(f"Baseline R²: {baseline['r2']:.4f}")
    
    # Fit hybrid
    logger.info("=" * 70)
    logger.info("Fitting Hybrid GAM-SSM")
    logger.info("=" * 70)
    
    hybrid = fit_hybrid_model(
        X_selected.values, y, time_idx, loc_idx, selected_features
    )
    logger.info(f"Hybrid RMSE: {hybrid['metrics']['rmse']:.4f}")
    logger.info(f"Hybrid R²: {hybrid['metrics']['r2']:.4f}")
    logger.info(f"95% Coverage: {hybrid['metrics']['coverage_95']:.1%}")
    
    # Calculate improvement
    rmse_improvement = (baseline['rmse'] - hybrid['metrics']['rmse']) / baseline['rmse'] * 100
    logger.info(f"RMSE Improvement: {rmse_improvement:.1f}%")
    
    # Create outputs
    logger.info("=" * 70)
    logger.info("Creating Outputs")
    logger.info("=" * 70)
    
    # Table 2
    table = create_comparison_table(
        baseline, hybrid,
        paths['tables_dir'] / 'table2_model_comparison.csv'
    )
    print("\nTable 2: Model Comparison")
    print(table.to_string(index=False))
    
    # Figure 6: Convergence
    create_convergence_plot(
        hybrid,
        paths['figures_dir'] / 'fig6_convergence.png'
    )
    
    # Figure 7: Observed vs Predicted
    create_observed_vs_predicted_plot(
        baseline, hybrid,
        paths['figures_dir'] / 'fig7_observed_vs_predicted.png'
    )
    
    # Figure 8: Residual diagnostics
    create_residual_diagnostics(
        hybrid,
        paths['figures_dir'] / 'fig8_residual_diagnostics.png'
    )
    
    # Save model
    hybrid['model'].save(paths['models_dir'] / 'hybrid_gam_ssm')
    
    logger.info("=" * 70)
    logger.info("Paper reproduction completed!")
    logger.info(f"Outputs saved to: {paths['output_dir']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
