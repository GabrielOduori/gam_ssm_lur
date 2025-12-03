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
    
    # Data arguments
    parser.add_argument(
        '--features', '-f',
        type=str,
        required=True,
        help='Path to CSV file with spatial features',
    )
    parser.add_argument(
        '--targets', '-t',
        type=str,
        required=True,
        help='Path to CSV file with target values (NO2 concentrations)',
    )
    parser.add_argument(
        '--time-column',
        type=str,
        default='timestamp',
        help='Name of time column in targets file',
    )
    parser.add_argument(
        '--location-column',
        type=str,
        default='location_id',
        help='Name of location column in targets file',
    )
    parser.add_argument(
        '--target-column',
        type=str,
        default='no2',
        help='Name of target column in targets file',
    )
    
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
    
    # Load data
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    
    logger.info(f"Loading targets from {args.targets}")
    targets_df = pd.read_csv(args.targets)
    
    # Extract arrays
    X = features_df.values
    feature_names = list(features_df.columns)
    y = targets_df[args.target_column].values
    time_index = targets_df[args.time_column].values
    
    location_index = None
    if args.location_column in targets_df.columns:
        location_index = targets_df[args.location_column].values
        
    logger.info(f"Data loaded: {len(y)} observations, {X.shape[1]} features")
    
    # Feature selection
    if not args.skip_feature_selection:
        logger.info("Running feature selection...")
        selector = FeatureSelector(
            correlation_threshold=args.corr_threshold,
            vif_threshold=args.vif_threshold,
            n_top_features=args.n_features,
            random_state=args.random_state,
        )
        X_selected = selector.fit_transform(X, y, feature_names=feature_names)
        feature_names = selector.selected_columns_
        X = X_selected.values
        logger.info(f"Selected {len(feature_names)} features")
    
    # Train model
    logger.info("Training hybrid GAM-SSM model...")
    model = HybridGAMSSM(
        n_splines=args.n_splines,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
        scalability_mode=args.scalability_mode,
        random_state=args.random_state,
    )
    
    model.fit(
        X=X,
        y=y,
        time_index=time_index,
        location_index=location_index,
        feature_names=feature_names,
    )
    
    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(output_dir / "model")
    logger.info(f"Model saved to {output_dir / 'model'}")
    
    # Evaluate and save results
    predictions = model.predict()
    evaluator = ModelEvaluator(model)
    
    # Save metrics
    y_matrix = model._y_matrix
    metrics = evaluator.compute_accuracy(y_matrix.flatten(), predictions.total.flatten())
    
    metrics_df = pd.DataFrame([metrics.to_dict()])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    logger.info(f"Metrics saved to {output_dir / 'metrics.csv'}")
    
    # Save diagnostics
    if args.save_diagnostics:
        evaluator.diagnostic_plots(
            y_true=y_matrix.flatten(),
            y_pred=predictions.total.flatten(),
            y_std=predictions.std.flatten(),
            save_path=output_dir / "diagnostics.png",
        )
        evaluator.convergence_plot(
            save_path=output_dir / "convergence.png",
        )
        
    # Print summary
    evaluator.summary_report(
        y_true=y_matrix.flatten(),
        y_pred=predictions.total.flatten(),
        y_std=predictions.std.flatten(),
    )
    
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
