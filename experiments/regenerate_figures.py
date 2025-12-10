#!/usr/bin/env python3
"""
Regenerate figures from saved model results without retraining.

This script loads a previously trained model and regenerates all figures
with the current plotting code. Useful for updating figure styling without
running the entire training pipeline.

Usage:
    python experiments/regenerate_figures.py --experiment-dir results/experiment_20251209_173407

    # Or use the most recent experiment automatically
    python experiments/regenerate_figures.py --latest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Setup project root and imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reproduce_paper import (
    create_convergence_plot,
    create_observed_vs_predicted_plot,
    create_residual_diagnostics,
    plot_temporal_evolution,
    plot_spatial_patterns,
    create_spatial_comparison_map,
    create_shap_importance_plot,
    create_residual_hotspot_map,
    create_morans_i_plot,
)
from mapping_utils import (
    create_gridded_comparison,
    create_gridded_residual_map,
    create_uncertainty_surface,
    create_temporal_gridded_sequence,
)
from gam_ssm_lur import HybridGAMSSM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_latest_experiment(results_dir: Path) -> Path:
    """Find the most recent experiment directory."""
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("experiment_")]
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in {results_dir}")

    # Sort by timestamp in directory name
    latest = sorted(experiment_dirs, key=lambda x: x.name)[-1]
    logger.info(f"Found latest experiment: {latest.name}")
    return latest


def load_model_and_data(experiment_dir: Path) -> dict:
    """Load saved model and associated data."""
    logger.info(f"Loading model from {experiment_dir}")

    # Load hybrid model
    model_path = experiment_dir / "models" / "hybrid_gam_ssm"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    hybrid_model = HybridGAMSSM.load(model_path)
    logger.info("Hybrid model loaded successfully")

    # Load predictions
    predictions_path = experiment_dir / "tables" / "predictions_with_intervals.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found at {predictions_path}")

    predictions_df = pd.read_csv(predictions_path)
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    logger.info(f"Loaded {len(predictions_df)} predictions")

    # Reconstruct predictions object
    n_times = hybrid_model.n_times_
    n_locs = hybrid_model.n_locations_

    # Reshape predictions to matrix form
    # Sort to ensure consistent ordering and handle any potential duplicates
    predictions_df = predictions_df.sort_values(['timestamp', 'location_id'])

    pred_matrix = predictions_df.pivot_table(
        index='timestamp', columns='location_id', values='pred_hybrid', aggfunc='first'
    ).values
    lower_matrix = predictions_df.pivot_table(
        index='timestamp', columns='location_id', values='pred_hybrid_lower', aggfunc='first'
    ).values
    upper_matrix = predictions_df.pivot_table(
        index='timestamp', columns='location_id', values='pred_hybrid_upper', aggfunc='first'
    ).values

    # Create predictions object
    from collections import namedtuple
    Predictions = namedtuple('Predictions', ['total', 'lower', 'upper', 'gam', 'ssm'])

    predictions = Predictions(
        total=pred_matrix,
        lower=lower_matrix,
        upper=upper_matrix,
        gam=None,  # Not needed for figures
        ssm=None   # Not needed for figures
    )

    # Create baseline dict
    # Baseline predictions are just a 1D array in the original order
    baseline = {
        'predictions': predictions_df['pred_baseline'].values,
        'model': None  # Not needed for most figures
    }

    # Create hybrid dict
    hybrid = {
        'model': hybrid_model,
        'predictions': predictions,
    }

    # Create observations dataframe
    observations = predictions_df[['timestamp', 'location_id', 'epa_values', 'latitude', 'longitude']].copy()
    observations.rename(columns={'epa_values': 'value', 'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    return {
        'hybrid': hybrid,
        'baseline': baseline,
        'observations': observations,
        'predictions_df': predictions_df,
    }


def regenerate_all_figures(
    data: dict,
    figures_dir: Path,
    skip_gridded: bool = False,
    skip_shap: bool = False,
):
    """Regenerate all figures using loaded model and data."""
    logger.info("Regenerating figures...")

    hybrid = data['hybrid']
    baseline = data['baseline']
    observations = data['observations']

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 6: Convergence (requires EM history, skip if not available)
    try:
        logger.info("Creating Figure 6: Convergence")
        create_convergence_plot(hybrid, figures_dir / "fig6_convergence.png")
    except (AttributeError, TypeError) as e:
        logger.warning(f"Skipping Figure 6 (Convergence) - EM history not saved with model: {e}")

    # Figure 7: Observed vs Predicted (requires training indices, skip if not available)
    try:
        logger.info("Creating Figure 7: Observed vs Predicted")
        create_observed_vs_predicted_plot(
            baseline, hybrid, figures_dir / "fig7_observed_vs_predicted.png"
        )
    except AttributeError as e:
        logger.warning(f"Skipping Figure 7 (Observed vs Predicted) - training indices not saved: {e}")

    # Figure 8: Residual diagnostics
    logger.info("Creating Figure 8: Residual Diagnostics")
    create_residual_diagnostics(hybrid, figures_dir / "fig8_residual_diagnostics.png")

    # Figure 9: Uncertainty timeseries
    logger.info("Creating Figure 9: Uncertainty Timeseries")

    # Get diverse locations (same logic as reproduce_paper.py)
    model = hybrid['model']
    predictions = hybrid['predictions']

    # Auto-select diverse locations
    residual_matrix = model._y_matrix - predictions.total
    location_variance = np.nanvar(residual_matrix, axis=0)
    location_mean = np.nanmean(model._y_matrix, axis=0)

    sorted_by_var = np.argsort(location_variance)
    n_locs = model.n_locations_
    diverse_loc_ids = [
        sorted_by_var[0],
        sorted_by_var[n_locs // 4],
        sorted_by_var[n_locs // 2],
        sorted_by_var[3 * n_locs // 4],
    ]

    # This function is defined inside main() in reproduce_paper.py, so we need to recreate it
    def plot_uncertainty_for_locations_wrapper(
        model, predictions, output_path: Path, loc_ids: list[int] = None, time_labels=None
    ):
        """Plot observed/predicted with uncertainty bands for multiple locations in a 2x2 grid."""
        import matplotlib.pyplot as plt

        # Default to first 4 locations if not specified
        if loc_ids is None:
            loc_ids = list(range(min(4, model.n_locations_)))

        # Ensure we have exactly 4 locations for 2x2 grid
        if len(loc_ids) > 4:
            loc_ids = loc_ids[:4]
            logger.warning(f"Only plotting first 4 locations from provided list: {loc_ids}")
        elif len(loc_ids) < 4:
            # Pad with additional locations if needed
            available_locs = list(range(model.n_locations_))
            for loc in available_locs:
                if loc not in loc_ids and len(loc_ids) < 4:
                    loc_ids.append(loc)

        # Setup x-axis - use time steps for cleaner display
        t_range = np.arange(model.n_times_)
        x_vals = t_range
        x_label = "Time step"

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for idx, loc_id in enumerate(loc_ids):
            ax = axes[idx]

            # Calculate statistics for this location
            obs_loc = model._y_matrix[:, loc_id]
            pred_loc = predictions.total[:, loc_id]
            mean_obs = np.nanmean(obs_loc)
            rmse_loc = np.sqrt(np.nanmean((obs_loc - pred_loc) ** 2))

            # Plot predicted line (blue, like fig8 temporal pattern)
            ax.plot(x_vals, pred_loc, "b-", lw=2, label="Predicted", color='#3498db')

            # Plot uncertainty band (blue shaded, like fig8)
            ax.fill_between(
                x_vals,
                predictions.lower[:, loc_id],
                predictions.upper[:, loc_id],
                alpha=0.3,
                label=f"{int(model.confidence_level * 100)}% CI",
                color='#3498db'
            )

            # Plot observed points (simple markers)
            ax.plot(
                x_vals,
                obs_loc,
                'o',
                color='#2c3e50',
                markersize=2.5,
                alpha=0.6,
                label="Observed"
            )

            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel("NO₂ (µg/m³)", fontsize=10)
            ax.set_title(
                f"Location {loc_id}\nMean: {mean_obs:.1f} µg/m³ | RMSE: {rmse_loc:.2f}",
                fontsize=10,
                pad=10
            )
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Calculate overall statistics for subtitle
        overall_rmse = np.sqrt(np.nanmean((model._y_matrix - predictions.total) ** 2))
        overall_r2 = 1 - (np.nansum((model._y_matrix - predictions.total) ** 2) /
                         np.nansum((model._y_matrix - np.nanmean(model._y_matrix)) ** 2))

        fig.suptitle(
            f"Prediction Intervals at Selected Locations (Overall RMSE: {overall_rmse:.2f}, R²: {overall_r2:.3f})",
            fontsize=14,
            fontweight='bold'
        )

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved uncertainty plot to {output_path}")
        plt.close()

    plot_uncertainty_for_locations_wrapper(
        model,
        predictions,
        figures_dir / "fig9_uncertainty_timeseries.png",
        loc_ids=diverse_loc_ids,
        time_labels=None,
    )

    # Figure 9b: Temporal Evolution
    logger.info("Creating Figure 9b: Temporal Evolution")
    plot_temporal_evolution(
        hybrid,
        figures_dir / "fig9b_temporal_evolution.png",
        selected_locations=None,
        time_labels=None
    )

    # Figure 9c: Spatial Patterns
    logger.info("Creating Figure 9c: Spatial Patterns")
    plot_spatial_patterns(
        hybrid,
        figures_dir / "fig9c_spatial_patterns.png",
        timesteps=None,
        observations_df=observations
    )

    # Figure 10: Spatial comparison maps
    logger.info("Creating Figure 10: Spatial Comparison")
    create_spatial_comparison_map(
        baseline,
        hybrid,
        observations,
        figures_dir / "fig10_spatial_comparison.png",
    )

    # Figure 11: SHAP feature importance (skip if requested, as it requires X data)
    if not skip_shap:
        logger.warning("Skipping Figure 11 (SHAP) - requires original feature matrix X")

    # Figure 12: Residual hotspots (requires proper index reconstruction)
    try:
        logger.info("Creating Figure 12: Residual Hotspots")

        # Reconstruct time_idx and loc_idx from observations
        time_idx = observations.groupby('timestamp').ngroup().values
        loc_map = {loc: i for i, loc in enumerate(model.location_ids_)}
        loc_idx = observations['location_id'].map(loc_map).values

        create_residual_hotspot_map(
            hybrid,
            baseline,
            time_idx,
            loc_idx,
            observations,
            figures_dir / "fig12_residual_hotspots.png",
        )
    except (KeyError, ValueError) as e:
        logger.warning(f"Skipping Figure 12 (Residual Hotspots) - index reconstruction failed: {e}")

    # Figure 13: Moran's I
    logger.info("Creating Figure 13: Moran's I")
    residual_means = (model._y_matrix - predictions.total).mean(axis=0)
    lats = observations[["location_id", "lat"]].drop_duplicates().set_index("location_id")["lat"].loc[model.location_ids_].values
    lons = observations[["location_id", "lon"]].drop_duplicates().set_index("location_id")["lon"].loc[model.location_ids_].values

    try:
        create_morans_i_plot(
            residual_means,
            lats,
            lons,
            figures_dir / "fig13_morans_i.png",
        )
    except Exception as e:
        logger.warning(f"Skipping Moran's I plot: {e}")

    # Gridded plots (Figures 15-18) - skip if requested
    if skip_gridded:
        logger.info("Skipping gridded plots (Figures 15-18) per --skip-gridded flag")
        return

    logger.info("Creating gridded plots (Figures 15-18)...")

    # Get coordinates
    coordinates = np.column_stack([lons, lats])

    # Figure 15: Gridded comparison
    logger.info("Creating Figure 15: Gridded Comparison")

    # Compute time-averaged values for comparison
    y_obs_mean = model._y_matrix.mean(axis=0)
    hybrid_mean = predictions.total.mean(axis=0)

    # Baseline predictions need to be reshaped and averaged
    # They are stored as a flat array, so we need to group by location
    baseline_df = data['predictions_df'].groupby('location_id')['pred_baseline'].mean()
    baseline_mean = baseline_df.loc[model.location_ids_].values

    create_gridded_comparison(
        coordinates=coordinates,
        observed=y_obs_mean,
        baseline_pred=baseline_mean,
        hybrid_pred=hybrid_mean,
        output_path=figures_dir / "fig15_gridded_comparison.png",
        title_suffix=" (Time-Averaged)",
    )

    # Figure 16: Gridded residuals
    logger.info("Creating Figure 16: Gridded Residuals")
    create_gridded_residual_map(
        coordinates=coordinates,
        observed=y_obs_mean,
        baseline_pred=baseline_mean,
        hybrid_pred=hybrid_mean,
        output_path=figures_dir / "fig16_gridded_residuals.png",
    )

    # Figure 17: Uncertainty surface
    logger.info("Creating Figure 17: Uncertainty Surface")
    # Compute standard deviation from confidence intervals
    # Assuming 95% CI: CI = mean ± 1.96*std, so std = (upper - lower) / (2 * 1.96)
    lower_mean = predictions.lower.mean(axis=0)
    upper_mean = predictions.upper.mean(axis=0)
    std_dev = (upper_mean - lower_mean) / (2 * 1.96)

    create_uncertainty_surface(
        coordinates=coordinates,
        predictions=hybrid_mean,
        std_dev=std_dev,
        output_path=figures_dir / "fig17_uncertainty_surface.png",
    )

    # Figure 18: Temporal sequence
    logger.info("Creating Figure 18: Temporal Sequence")
    n_times = model.n_times_
    time_points = [0, n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]

    create_temporal_gridded_sequence(
        coordinates=coordinates,
        observed_matrix=model._y_matrix,
        predicted_matrix=predictions.total,
        time_points=time_points,
        output_path=figures_dir / "fig18_temporal_sequence.png",
    )

    logger.info(f"All figures saved to {figures_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate figures from saved model results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment-dir",
        type=Path,
        help="Path to experiment directory (e.g., results/experiment_20251209_173407)",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent experiment directory in results/",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory (default: results/)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for figures (default: same as experiment)",
    )

    parser.add_argument(
        "--skip-gridded",
        action="store_true",
        help="Skip gridded plots (Figures 15-18)",
    )

    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP plot (Figure 11)",
    )

    args = parser.parse_args()

    # Determine experiment directory
    if args.latest:
        experiment_dir = find_latest_experiment(args.results_dir)
    else:
        experiment_dir = args.experiment_dir
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Determine output directory
    if args.output_dir:
        figures_dir = args.output_dir
    else:
        figures_dir = experiment_dir / "figures"

    logger.info("=" * 70)
    logger.info("REGENERATE FIGURES")
    logger.info("=" * 70)
    logger.info(f"Experiment dir: {experiment_dir}")
    logger.info(f"Output dir: {figures_dir}")
    logger.info("")

    # Load model and data
    data = load_model_and_data(experiment_dir)

    # Regenerate figures
    regenerate_all_figures(
        data,
        figures_dir,
        skip_gridded=args.skip_gridded,
        skip_shap=args.skip_shap,
    )

    logger.info("=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
