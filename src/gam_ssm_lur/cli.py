"""
Command-Line Interface for GAM-SSM-LUR.

Provides CLI commands for training and prediction:
    gam-ssm-train: Train a hybrid model
    gam-ssm-predict: Generate predictions from a trained model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def train() -> None:
    """Train a hybrid GAM-SSM model."""
    parser = argparse.ArgumentParser(
        description='Train a hybrid GAM-SSM model for air pollution prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data directory (standard contract) ───────────────────────────────────
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        required=True,
        help=(
            'Root data directory containing features.csv, target.csv and a '
            'time_series/ sub-directory with temporal data files.'
        ),
    )

    # ── Column / file name overrides (all have Dublin defaults) ──────────────
    parser.add_argument('--target-col', default='atmos_no2',
                        help='Target column name in target.csv')
    parser.add_argument('--dense-obs-file', default='satellite_retreavals.csv',
                        help='Dense gridded observation file (Kalman update source)')
    parser.add_argument('--dense-obs-value-col', default='tropomi_no2',
                        help='Value column in dense obs file')
    parser.add_argument('--dense-obs-timestamp-col', default='timestamp',
                        help='Timestamp column in dense obs file')
    parser.add_argument('--point-obs-file', default='epa_timeseries.csv',
                        help='Point observation file (validation only)')
    parser.add_argument('--point-obs-value-col', default='epa_no2',
                        help='Value column in point obs file')
    parser.add_argument('--point-obs-timestamp-col', default='timestamp_utc',
                        help='Timestamp column in point obs file')
    parser.add_argument('--activity-file', default='traffic_timeseries.csv',
                        help='Activity forcing file (city-wide transition covariate)')
    parser.add_argument('--activity-value-col', default='traffic_volume',
                        help='Volume column in activity file')
    parser.add_argument('--activity-timestamp-col', default='traffic_end_time',
                        help='Timestamp column in activity file')
    parser.add_argument('--met-forcing-file',
                        default='wind_sector_features_era5land_2023-06_daily.csv',
                        help='Meteorological forcing file')
    parser.add_argument('--met-n-sectors', type=int, default=8,
                        help='Number of sectors in met forcing file')
    parser.add_argument('--grid-geojson', default='grid.geojson',
                        help='Grid polygon GeoJSON (relative to data-dir, optional)')
    
    # Model arguments
    parser.add_argument(
        '--n-splines',
        type=int,
        default=10,
        help='Number of splines for GAM',
    )
    parser.add_argument(
        '--em-max-iter',
        type=int,
        default=50,
        help='Maximum EM iterations',
    )
    parser.add_argument(
        '--em-tol',
        type=float,
        default=1e-6,
        help='EM convergence tolerance',
    )
    parser.add_argument(
        '--scalability-mode',
        type=str,
        choices=['auto', 'dense', 'diagonal', 'block'],
        default='auto',
        help='Kalman filter scalability mode',
    )
    
    # Feature selection
    parser.add_argument(
        '--corr-threshold',
        type=float,
        default=0.8,
        help='Correlation threshold for feature selection',
    )
    parser.add_argument(
        '--vif-threshold',
        type=float,
        default=10.0,
        help='VIF threshold for feature selection',
    )
    parser.add_argument(
        '--n-features',
        type=int,
        default=30,
        help='Number of features to select',
    )
    parser.add_argument(
        '--skip-feature-selection',
        action='store_true',
        help='Skip feature selection step',
    )
    
    # Output arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for model and results',
    )
    parser.add_argument(
        '--save-diagnostics',
        action='store_true',
        help='Save diagnostic plots',
    )
    
    # Other arguments
    parser.add_argument(
        '--random-state',
        type=int,
        default=None,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting GAM-SSM training pipeline")

    # Import here to avoid slow imports for --help
    from gam_ssm_lur import HybridGAMSSM, FeatureSelector, ModelEvaluator
    from gam_ssm_lur.data import SpatiotemporalDataset
    from gam_ssm_lur.features import inverse_distance_transform, filter_sparse_cells

    # ── Load data ─────────────────────────────────────────────────────────────
    ds = SpatiotemporalDataset(
        data_dir=args.data_dir,
        target_col=args.target_col,
        dense_obs_file=args.dense_obs_file,
        dense_obs_value_col=args.dense_obs_value_col,
        dense_obs_timestamp_col=args.dense_obs_timestamp_col,
        point_obs_file=args.point_obs_file,
        point_obs_value_col=args.point_obs_value_col,
        point_obs_timestamp_col=args.point_obs_timestamp_col,
        activity_file=args.activity_file,
        activity_value_col=args.activity_value_col,
        activity_timestamp_col=args.activity_timestamp_col,
        met_forcing_file=args.met_forcing_file,
        met_n_sectors=args.met_n_sectors,
        grid_geojson=args.grid_geojson,
    )

    logger.info("Loading static data...")
    static = ds.load_static()

    logger.info("Loading temporal data...")
    temporal = ds.load_temporal()

    logger.info("Calibrating dense observations against point observations...")
    calibration = ds.calibrate_dense_obs(temporal, static)

    # ── Feature engineering ───────────────────────────────────────────────────
    id_cols = ["grid_id", "latitude", "longitude"]
    features_df = inverse_distance_transform(static.features)

    target_merged = static.features[["grid_id"]].merge(
        static.target[["grid_id", args.target_col]], on="grid_id", how="left"
    )
    y_full = target_merged[args.target_col]

    feat_cols = [c for c in features_df.columns if c not in id_cols]
    X_df = features_df[feat_cols]

    if not args.skip_feature_selection:
        logger.info("Filtering sparse cells...")
        X_df, y_full = filter_sparse_cells(
            X_df.assign(**{c: features_df[c] for c in id_cols}),
            y_full,
            min_nonzero_features=args.min_nonzero_features if hasattr(args, 'min_nonzero_features') else 1,
            id_cols=id_cols,
        )
        # Re-extract after filter
        feat_cols = [c for c in X_df.columns if c not in id_cols]
        X_df = X_df[feat_cols]

        logger.info("Running feature selection...")
        selector = FeatureSelector(
            correlation_threshold=args.corr_threshold,
            vif_threshold=args.vif_threshold,
            n_top_features=args.n_features,
            random_state=args.random_state,
        )
        X_df = selector.fit_transform(X_df, y_full)
        logger.info("Selected %d features", len(X_df.columns))

    # ── Train model ───────────────────────────────────────────────────────────
    logger.info("Training hybrid GAM-SSM model...")
    model = HybridGAMSSM(
        n_splines=args.n_splines,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
        scalability_mode=args.scalability_mode,
        random_state=args.random_state,
    )

    model.fit_from_dataset(
        static=static,
        temporal=temporal,
        calibration=calibration,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "model")
    logger.info("Model saved to %s", output_dir / "model")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator = ModelEvaluator(model)

    if model._residual_matrix is not None:
        ssm_pred = model.ssm_.predict()
        lur_pred = model.gam_.predict(model._X_train)
        total_pred = lur_pred[:, np.newaxis] + ssm_pred.mean  # broadcast (n,) + (T,n)

        metrics = evaluator.compute_accuracy(
            model._y_train,
            lur_pred,
        )
        metrics_df = pd.DataFrame([metrics.to_dict()])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        logger.info("GAM metrics saved to %s", output_dir / "metrics.csv")

    if args.save_diagnostics:
        evaluator.convergence_plot(save_path=output_dir / "convergence.png")

    logger.info("Training complete!")
    

def predict() -> None:
    """Generate predictions from a trained model."""
    parser = argparse.ArgumentParser(
        description='Generate predictions from a trained GAM-SSM model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model directory',
    )
    parser.add_argument(
        '--features', '-f',
        type=str,
        required=True,
        help='Path to CSV file with features for prediction',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path for output predictions CSV',
    )
    parser.add_argument(
        '--include-intervals',
        action='store_true',
        help='Include prediction intervals in output',
    )
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for prediction intervals',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Import here to avoid slow imports for --help
    from gam_ssm_lur import HybridGAMSSM
    
    logger.info(f"Loading model from {args.model}")
    model = HybridGAMSSM.load(args.model)
    
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    
    logger.info("Generating predictions...")
    predictions = model.predict(features_df)
    
    # Build output dataframe
    output_df = pd.DataFrame({
        'prediction': predictions.total.flatten(),
    })
    
    if args.include_intervals:
        output_df['std'] = predictions.std.flatten()
        output_df['lower'] = predictions.lower.flatten()
        output_df['upper'] = predictions.upper.flatten()
        
    output_df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: gam-ssm-lur <command> [options]")
        print("\nCommands:")
        print("  train    Train a hybrid GAM-SSM model")
        print("  predict  Generate predictions from a trained model")
        sys.exit(1)
        
    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove command from argv
    
    if command == 'train':
        train()
    elif command == 'predict':
        predict()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
