#!/usr/bin/env python
"""
Basic Usage Example for GAM-SSM-LUR.

This script demonstrates the core workflow:
1. Generate synthetic spatiotemporal pollution data
2. Fit the hybrid GAM-SSM model
3. Evaluate performance and visualize results

Run with:
    python examples/01_basic_usage.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from output_utils import make_experiment_dirs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_locations: int = 50,
    n_times: int = 100,
    n_features: int = 10,
    noise_level: float = 0.5,
    random_state: int = 42,
) -> tuple:
    """Generate synthetic spatiotemporal pollution data.
    
    The data follows the model:
        y(s,t) = μ(s) + α(t) + ε(s,t)
    
    where:
        - μ(s) is a spatial mean based on features (non-linear)
        - α(t) is an AR(1) temporal process
        - ε(s,t) is observation noise
    
    Parameters
    ----------
    n_locations : int
        Number of spatial locations
    n_times : int
        Number of time steps
    n_features : int
        Number of spatial features
    noise_level : float
        Standard deviation of observation noise
    random_state : int
        Random seed
        
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_locations * n_times, n_features)
    y : np.ndarray
        Observations (n_locations * n_times,)
    time_idx : np.ndarray
        Time indices
    loc_idx : np.ndarray
        Location indices
    true_spatial : np.ndarray
        True spatial component
    true_temporal : np.ndarray
        True temporal component
    """
    np.random.seed(random_state)
    
    logger.info(f"Generating synthetic data: {n_locations} locations × {n_times} times")
    
    # Generate spatial features (static per location)
    X_spatial = np.random.randn(n_locations, n_features)
    
    # True spatial effect (non-linear combination)
    # Simulating LUR-style relationships
    true_spatial = (
        2.0 * np.sin(X_spatial[:, 0])  # Non-linear effect of feature 0
        + 1.5 * X_spatial[:, 1]         # Linear effect of feature 1
        - 0.5 * X_spatial[:, 2]**2      # Quadratic effect of feature 2
        + 0.3 * X_spatial[:, 3] * X_spatial[:, 4]  # Interaction
    )
    
    # Normalize spatial component
    true_spatial = (true_spatial - true_spatial.mean()) / true_spatial.std() * 5
    
    # Generate temporal process (AR(1))
    ar_coef = 0.8
    temporal_noise = np.random.randn(n_times) * 2
    true_temporal = np.zeros(n_times)
    true_temporal[0] = temporal_noise[0]
    for t in range(1, n_times):
        true_temporal[t] = ar_coef * true_temporal[t-1] + temporal_noise[t]
    
    # Create full dataset
    n_obs = n_locations * n_times
    X = np.zeros((n_obs, n_features))
    y = np.zeros(n_obs)
    time_idx = np.zeros(n_obs, dtype=int)
    loc_idx = np.zeros(n_obs, dtype=int)
    
    idx = 0
    for t in range(n_times):
        for s in range(n_locations):
            X[idx] = X_spatial[s]
            y[idx] = (
                true_spatial[s] 
                + true_temporal[t] 
                + noise_level * np.random.randn()
            )
            time_idx[idx] = t
            loc_idx[idx] = s
            idx += 1
    
    # Add some feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    logger.info(f"Generated {n_obs} observations")
    
    return X, y, time_idx, loc_idx, true_spatial, true_temporal, feature_names


def main():
    """Run the basic usage example."""
    
    # Import the package
    try:
        from gam_ssm_lur import HybridGAMSSM
        from gam_ssm_lur.features import FeatureSelector
        from gam_ssm_lur.evaluation import ModelEvaluator
    except ImportError:
        logger.error("gam_ssm_lur not installed. Run: pip install -e .")
        logger.info("Attempting relative import for development...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from gam_ssm_lur import HybridGAMSSM
        from gam_ssm_lur.features import FeatureSelector
        from gam_ssm_lur.evaluation import ModelEvaluator
    
    # Create output directory (timestamped to avoid overwriting previous runs)
    _, output_groups = make_experiment_dirs(base="outputs", groups=["basic_example"])
    output_dir = output_groups["basic_example"]
    
    # =========================================================================
    # Step 1: Generate synthetic data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Generating synthetic data")
    logger.info("=" * 60)
    
    X, y, time_idx, loc_idx, true_spatial, true_temporal, feature_names = \
        generate_synthetic_data(
            n_locations=50,
            n_times=100,
            n_features=10,
            noise_level=0.5,
            random_state=42,
        )
    
    # =========================================================================
    # Step 2: Feature selection (optional but recommended)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 2: Feature selection")
    logger.info("=" * 60)
    
    selector = FeatureSelector(
        correlation_threshold=0.9,
        vif_threshold=10.0,
        n_top_features=8,
        random_state=42,
    )
    
    X_df = pd.DataFrame(X, columns=feature_names)
    X_selected = selector.fit_transform(X_df, y)
    
    logger.info(selector.get_summary())
    
    # =========================================================================
    # Step 3: Fit hybrid model
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 3: Fitting hybrid GAM-SSM model")
    logger.info("=" * 60)
    
    model = HybridGAMSSM(
        n_splines=8,
        gam_lam="auto",
        em_max_iter=30,
        em_tol=1e-5,
        scalability_mode="auto",
        confidence_level=0.95,
        random_state=42,
    )
    
    model.fit(
        X=X_selected,
        y=y,
        time_index=time_idx,
        location_index=loc_idx,
    )
    
    # =========================================================================
    # Step 4: Get predictions
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 4: Generating predictions")
    logger.info("=" * 60)
    
    predictions = model.predict()
    
    # =========================================================================
    # Step 5: Evaluate performance
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 5: Evaluating performance")
    logger.info("=" * 60)
    
    # Reshape y to matrix form for comparison
    n_locations = 50
    n_times = 100
    y_matrix = y.reshape(n_times, n_locations)
    
    metrics = model.evaluate(
        y_true=y_matrix.flatten(),
        y_pred=predictions.total.flatten(),
        y_lower=predictions.lower.flatten(),
        y_upper=predictions.upper.flatten(),
    )
    
    logger.info("Performance Metrics:")
    logger.info(f"  RMSE:        {metrics['rmse']:.4f}")
    logger.info(f"  MAE:         {metrics['mae']:.4f}")
    logger.info(f"  R²:          {metrics['r2']:.4f}")
    logger.info(f"  Correlation: {metrics['correlation']:.4f}")
    logger.info(f"  95% Coverage:{metrics['coverage_95']:.1%}")
    
    # Get model summary
    summary = model.summary()
    logger.info(f"\nGAM R²: {summary.gam_summary.r_squared:.4f}")
    logger.info(f"SSM converged: {summary.ssm_diagnostics.em_converged}")
    
    # =========================================================================
    # Step 6: Visualize results
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Creating visualizations")
    logger.info("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Observed vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_matrix.flatten(), predictions.total.flatten(), alpha=0.3, s=5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Observed vs Predicted (R² = {metrics['r2']:.3f})")
    
    # Plot 2: Temporal component comparison
    ax = axes[0, 1]
    # Average temporal component across locations
    pred_temporal_mean = predictions.temporal.mean(axis=1)
    ax.plot(true_temporal, 'b-', label='True temporal', alpha=0.7)
    ax.plot(pred_temporal_mean, 'r--', label='Estimated temporal', alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Temporal component")
    ax.set_title("Temporal Component Recovery")
    ax.legend()
    
    # Plot 3: Spatial component comparison
    ax = axes[1, 0]
    # Average spatial prediction across time
    pred_spatial_mean = predictions.spatial.mean(axis=0)
    ax.scatter(true_spatial, pred_spatial_mean, alpha=0.7)
    ax.plot([true_spatial.min(), true_spatial.max()], 
            [true_spatial.min(), true_spatial.max()], 'r--', lw=2)
    ax.set_xlabel("True spatial component")
    ax.set_ylabel("Estimated spatial component")
    ax.set_title("Spatial Component Recovery")
    
    # Plot 4: Prediction intervals for one location
    ax = axes[1, 1]
    loc_id = 0
    t_range = np.arange(n_times)
    ax.fill_between(t_range, predictions.lower[:, loc_id], predictions.upper[:, loc_id],
                    alpha=0.3, color='blue', label='95% CI')
    ax.plot(t_range, y_matrix[:, loc_id], 'k.', markersize=3, label='Observed')
    ax.plot(t_range, predictions.total[:, loc_id], 'b-', lw=1, label='Predicted')
    ax.set_xlabel("Time")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.set_title(f"Predictions with Uncertainty (Location {loc_id})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_results.png", dpi=150)
    logger.info(f"Saved figure to {output_dir / 'model_results.png'}")
    plt.close()
    
    # =========================================================================
    # Step 7: EM Convergence
    # =========================================================================
    em_history = model.get_em_convergence()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(em_history['iteration'], em_history['log_likelihood'], 'b-o')
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("EM Algorithm Convergence")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "em_convergence.png", dpi=150)
    logger.info(f"Saved figure to {output_dir / 'em_convergence.png'}")
    plt.close()
    
    # =========================================================================
    # Step 8: Save model
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 8: Saving model")
    logger.info("=" * 60)
    
    model.save(output_dir / "model")
    logger.info(f"Model saved to {output_dir / 'model'}")
    
    # Verify loading works
    loaded_model = HybridGAMSSM.load(output_dir / "model")
    logger.info("Model loaded successfully!")
    
    logger.info("=" * 60)
    logger.info("Example completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
