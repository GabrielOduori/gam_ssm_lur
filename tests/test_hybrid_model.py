"""
Tests for the Hybrid GAM-SSM model.
"""

import numpy as np
import pandas as pd
import pytest

from gam_ssm_lur.hybrid_model import HybridGAMSSM, HybridPrediction


def generate_synthetic_data(
    n_locations: int = 50,
    n_times: int = 100,
    n_features: int = 10,
    noise_level: float = 0.5,
    random_state: int = 42,
):
    """Generate synthetic spatiotemporal data for testing."""
    np.random.seed(random_state)
    
    # Generate spatial features
    X = np.random.randn(n_locations * n_times, n_features)
    
    # Generate spatial pattern (based on features)
    true_coefficients = np.random.randn(n_features) * 0.5
    spatial_effect = X @ true_coefficients
    
    # Generate temporal pattern (AR(1) process)
    temporal_effect = np.zeros(n_times)
    for t in range(1, n_times):
        temporal_effect[t] = 0.8 * temporal_effect[t-1] + np.random.randn() * 0.3
        
    # Expand temporal effect to all observations
    temporal_expanded = np.tile(temporal_effect, n_locations)
    
    # Combine effects
    y = spatial_effect + temporal_expanded + np.random.randn(n_locations * n_times) * noise_level
    
    # Create indices
    time_index = np.repeat(np.arange(n_times), n_locations)
    location_index = np.tile(np.arange(n_locations), n_times)
    
    return X, y, time_index, location_index


class TestHybridGAMSSM:
    """Tests for HybridGAMSSM class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = HybridGAMSSM(n_splines=10, em_max_iter=50)
        
        assert model.n_splines == 10
        assert model.em_max_iter == 50
        assert not model.is_fitted_
        
    @pytest.mark.slow
    def test_fit_basic(self):
        """Test basic model fitting."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=20, n_times=50, n_features=5
        )
        
        model = HybridGAMSSM(
            n_splines=5,
            em_max_iter=10,
            scalability_mode='dense',
            random_state=42,
        )
        
        model.fit(X, y, time_index=time_idx, location_index=loc_idx)
        
        assert model.is_fitted_
        assert model.gam_ is not None
        assert model.ssm_ is not None
        assert model.n_locations_ == 20
        assert model.n_times_ == 50
        
    @pytest.mark.slow
    def test_predict(self):
        """Test prediction generation."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=20, n_times=50, n_features=5
        )
        
        model = HybridGAMSSM(
            n_splines=5,
            em_max_iter=10,
            random_state=42,
        )
        model.fit(X, y, time_index=time_idx, location_index=loc_idx)
        
        predictions = model.predict()
        
        assert isinstance(predictions, HybridPrediction)
        assert predictions.total.shape == (50, 20)  # (n_times, n_locations)
        assert predictions.spatial.shape == (50, 20)
        assert predictions.temporal.shape == (50, 20)
        assert predictions.std.shape == (50, 20)
        assert predictions.lower.shape == (50, 20)
        assert predictions.upper.shape == (50, 20)
        
    @pytest.mark.slow
    def test_prediction_intervals(self):
        """Test that prediction intervals have proper coverage."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=20, n_times=100, n_features=5, noise_level=0.3
        )
        
        model = HybridGAMSSM(
            n_splines=5,
            em_max_iter=20,
            confidence_level=0.95,
            random_state=42,
        )
        model.fit(X, y, time_index=time_idx, location_index=loc_idx)
        
        predictions = model.predict()
        y_matrix = model._y_matrix
        
        # Compute coverage
        in_interval = (y_matrix >= predictions.lower) & (y_matrix <= predictions.upper)
        coverage = np.mean(in_interval)
        
        # Coverage should be approximately 95% (allow some tolerance)
        assert coverage > 0.80, f"Coverage {coverage:.1%} is too low"
        assert coverage < 1.0, f"Coverage {coverage:.1%} is suspiciously high"
        
    @pytest.mark.slow
    def test_evaluate(self):
        """Test model evaluation."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=20, n_times=50, n_features=5
        )
        
        model = HybridGAMSSM(
            n_splines=5,
            em_max_iter=10,
            random_state=42,
        )
        model.fit(X, y, time_index=time_idx, location_index=loc_idx)
        
        predictions = model.predict()
        metrics = model.evaluate(
            model._y_matrix.flatten(),
            predictions.total.flatten()
        )
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'correlation' in metrics
        
        # Basic sanity checks
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['correlation'] >= -1 and metrics['correlation'] <= 1
        
    @pytest.mark.slow
    def test_summary(self):
        """Test model summary."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=20, n_times=50, n_features=5
        )
        
        model = HybridGAMSSM(
            n_splines=5,
            em_max_iter=10,
            random_state=42,
        )
        model.fit(X, y, time_index=time_idx, location_index=loc_idx)
        
        summary = model.summary()
        
        assert hasattr(summary, 'gam_summary')
        assert hasattr(summary, 'ssm_diagnostics')
        assert hasattr(summary, 'total_rmse')
        assert hasattr(summary, 'total_r2')
        
    @pytest.mark.slow
    def test_feature_importance(self):
        """Test feature importance extraction."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=20, n_times=50, n_features=5
        )
        
        feature_names = [f'feature_{i}' for i in range(5)]
        
        model = HybridGAMSSM(
            n_splines=5,
            em_max_iter=10,
            random_state=42,
        )
        model.fit(X, y, time_index=time_idx, location_index=loc_idx, feature_names=feature_names)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == 5
        
    @pytest.mark.slow
    def test_dataframe_input(self):
        """Test fitting with DataFrame input."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=20, n_times=50, n_features=5
        )
        
        # Convert to DataFrames
        X_df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
        y_series = pd.Series(y, name='target')
        
        model = HybridGAMSSM(
            n_splines=5,
            em_max_iter=10,
            random_state=42,
        )
        model.fit(X_df, y_series, time_index=time_idx, location_index=loc_idx)
        
        assert model.is_fitted_
        
    @pytest.mark.slow
    def test_save_load(self, tmp_path):
        """Test model save and load."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=20, n_times=50, n_features=5
        )
        
        model = HybridGAMSSM(
            n_splines=5,
            em_max_iter=10,
            random_state=42,
        )
        model.fit(X, y, time_index=time_idx, location_index=loc_idx)
        
        # Save
        save_dir = tmp_path / "model"
        model.save(save_dir)
        
        # Load
        loaded_model = HybridGAMSSM.load(save_dir)
        
        assert loaded_model.is_fitted_
        assert loaded_model.n_locations_ == model.n_locations_
        assert loaded_model.n_times_ == model.n_times_
        
        # Predictions should match
        pred_original = model.predict()
        pred_loaded = loaded_model.predict()
        
        np.testing.assert_array_almost_equal(
            pred_original.total, pred_loaded.total
        )


class TestHybridModelEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_not_fitted_error(self):
        """Test that predict raises error before fitting."""
        model = HybridGAMSSM()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict()
            
    def test_single_location(self):
        """Test with single location (degenerate case)."""
        np.random.seed(42)
        
        n_times = 50
        n_features = 3
        
        X = np.random.randn(n_times, n_features)
        y = X @ np.array([1, 0.5, -0.3]) + np.cumsum(np.random.randn(n_times) * 0.1)
        time_idx = np.arange(n_times)
        loc_idx = np.zeros(n_times, dtype=int)
        
        model = HybridGAMSSM(
            n_splines=3,
            em_max_iter=5,
            random_state=42,
        )
        
        # Should handle single location
        model.fit(X, y, time_index=time_idx, location_index=loc_idx)
        predictions = model.predict()
        
        assert predictions.total.shape == (n_times, 1)
        
    def test_short_time_series(self):
        """Test with very short time series."""
        X, y, time_idx, loc_idx = generate_synthetic_data(
            n_locations=10, n_times=10, n_features=3
        )
        
        model = HybridGAMSSM(
            n_splines=3,
            em_max_iter=5,
            random_state=42,
        )
        
        model.fit(X, y, time_index=time_idx, location_index=loc_idx)
        predictions = model.predict()
        
        assert predictions.total.shape == (10, 10)
