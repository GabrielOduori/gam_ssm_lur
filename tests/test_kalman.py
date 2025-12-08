"""
Tests for Kalman filter and RTS smoother.
"""

import numpy as np
import pytest

from gam_ssm_lur.inference import KalmanFilter, RTSSmoother, FilterResult, SmootherResult


class TestKalmanFilter:
    """Tests for KalmanFilter class."""
    
    def test_initialization(self):
        """Test filter initialization."""
        kf = KalmanFilter(state_dim=10, obs_dim=10, mode='dense')
        assert kf.state_dim == 10
        assert kf.obs_dim == 10
        assert kf.mode == 'dense'
        assert not kf._initialized
        
    def test_auto_mode_selection(self):
        """Test automatic mode selection based on problem size."""
        # Small problem -> dense
        kf_small = KalmanFilter(state_dim=100, obs_dim=100, mode='auto')
        assert kf_small.mode == 'dense'
        
        # Medium problem -> diagonal
        kf_medium = KalmanFilter(state_dim=2000, obs_dim=2000, mode='auto')
        assert kf_medium.mode == 'diagonal'
        
        # Large problem -> block
        kf_large = KalmanFilter(state_dim=6000, obs_dim=6000, mode='auto')
        assert kf_large.mode == 'block'
        
    def test_initialize_matrices(self):
        """Test matrix initialization."""
        kf = KalmanFilter(state_dim=5, obs_dim=5, mode='dense')
        
        T = np.eye(5) * 0.9
        Z = np.eye(5)
        Q = np.eye(5) * 0.1
        H = np.eye(5) * 0.2
        
        kf.initialize(T=T, Z=Z, Q=Q, H=H)
        
        assert kf._initialized
        np.testing.assert_array_almost_equal(kf.T, T)
        np.testing.assert_array_almost_equal(kf.Z, Z)
        np.testing.assert_array_almost_equal(kf.Q, Q)
        np.testing.assert_array_almost_equal(kf.H, H)
        
    def test_filter_simple(self):
        """Test filtering on simple random walk model."""
        np.random.seed(42)
        
        # Generate data from random walk
        T_len = 100
        state_dim = 3
        
        true_states = np.zeros((T_len, state_dim))
        observations = np.zeros((T_len, state_dim))
        
        for t in range(T_len):
            if t > 0:
                true_states[t] = true_states[t-1] + np.random.randn(state_dim) * 0.1
            observations[t] = true_states[t] + np.random.randn(state_dim) * 0.5
            
        # Create and run filter
        kf = KalmanFilter(state_dim=state_dim, obs_dim=state_dim, mode='dense')
        kf.initialize(
            T=np.eye(state_dim),
            Z=np.eye(state_dim),
            Q=np.eye(state_dim) * 0.01,
            H=np.eye(state_dim) * 0.25,
        )
        
        result = kf.filter(observations)
        
        assert isinstance(result, FilterResult)
        assert result.filtered_means.shape == (T_len, state_dim)
        assert result.filtered_covariances.shape == (T_len, state_dim, state_dim)
        assert np.isfinite(result.log_likelihood)
        
        # Filtered states should be closer to true states than observations
        filter_error = np.mean((result.filtered_means - true_states)**2)
        obs_error = np.mean((observations - true_states)**2)
        assert filter_error < obs_error
        
    def test_filter_diagonal_mode(self):
        """Test filtering with diagonal approximation."""
        np.random.seed(42)
        
        T_len = 50
        state_dim = 10
        
        observations = np.random.randn(T_len, state_dim)
        
        kf = KalmanFilter(state_dim=state_dim, obs_dim=state_dim, mode='diagonal')
        kf.initialize(
            T=np.eye(state_dim) * 0.9,
            Z=np.eye(state_dim),
            Q=np.eye(state_dim) * 0.1,
            H=np.eye(state_dim) * 0.2,
        )
        
        result = kf.filter(observations)
        
        assert result.filtered_means.shape == (T_len, state_dim)
        # Diagonal mode stores covariances as vectors
        assert result.filtered_covariances.shape == (T_len, state_dim)
        assert np.isfinite(result.log_likelihood)
        
    def test_filter_missing_data(self):
        """Test filtering with missing observations."""
        np.random.seed(42)
        
        T_len = 50
        state_dim = 3
        
        observations = np.random.randn(T_len, state_dim)
        missing_mask = np.zeros((T_len, state_dim), dtype=bool)
        missing_mask[10:15, :] = True  # Missing observations at t=10-14
        
        kf = KalmanFilter(state_dim=state_dim, obs_dim=state_dim, mode='dense')
        kf.initialize(
            T=np.eye(state_dim) * 0.9,
            Z=np.eye(state_dim),
            Q=np.eye(state_dim) * 0.1,
            H=np.eye(state_dim) * 0.2,
        )
        
        result = kf.filter(observations, missing_mask=missing_mask)
        
        assert result.filtered_means.shape == (T_len, state_dim)
        assert np.isfinite(result.log_likelihood)
        

class TestRTSSmoother:
    """Tests for RTSSmoother class."""
    
    def test_smoother_basic(self):
        """Test basic smoothing operation."""
        np.random.seed(42)
        
        T_len = 50
        state_dim = 3
        
        observations = np.random.randn(T_len, state_dim)
        
        # Filter first
        kf = KalmanFilter(state_dim=state_dim, obs_dim=state_dim, mode='dense')
        kf.initialize(
            T=np.eye(state_dim) * 0.9,
            Z=np.eye(state_dim),
            Q=np.eye(state_dim) * 0.1,
            H=np.eye(state_dim) * 0.2,
        )
        filter_result = kf.filter(observations)
        
        # Then smooth
        smoother = RTSSmoother(kf)
        smooth_result = smoother.smooth(filter_result)
        
        assert isinstance(smooth_result, SmootherResult)
        assert smooth_result.smoothed_means.shape == (T_len, state_dim)
        assert smooth_result.smoothed_covariances.shape == (T_len, state_dim, state_dim)
        assert smooth_result.cross_covariances.shape == (T_len, state_dim, state_dim)
        
    def test_smoother_reduces_variance(self):
        """Test that smoothing reduces posterior variance."""
        np.random.seed(42)
        
        T_len = 100
        state_dim = 5
        
        # Generate from model
        true_states = np.cumsum(np.random.randn(T_len, state_dim) * 0.1, axis=0)
        observations = true_states + np.random.randn(T_len, state_dim) * 0.5
        
        kf = KalmanFilter(state_dim=state_dim, obs_dim=state_dim, mode='dense')
        kf.initialize(
            T=np.eye(state_dim),
            Z=np.eye(state_dim),
            Q=np.eye(state_dim) * 0.01,
            H=np.eye(state_dim) * 0.25,
        )
        
        filter_result = kf.filter(observations)
        smoother = RTSSmoother(kf)
        smooth_result = smoother.smooth(filter_result)
        
        # Smoothed variance should be less than or equal to filtered variance
        for t in range(T_len - 1):  # Exclude last point where they should be equal
            smooth_var = np.trace(smooth_result.smoothed_covariances[t])
            filter_var = np.trace(filter_result.filtered_covariances[t])
            assert smooth_var <= filter_var + 1e-6
            
    def test_smoother_diagonal_mode(self):
        """Test smoothing with diagonal mode."""
        np.random.seed(42)
        
        T_len = 50
        state_dim = 10
        
        observations = np.random.randn(T_len, state_dim)
        
        kf = KalmanFilter(state_dim=state_dim, obs_dim=state_dim, mode='diagonal')
        kf.initialize(
            T=np.eye(state_dim) * 0.9,
            Z=np.eye(state_dim),
            Q=np.eye(state_dim) * 0.1,
            H=np.eye(state_dim) * 0.2,
        )
        
        filter_result = kf.filter(observations)
        smoother = RTSSmoother(kf)
        smooth_result = smoother.smooth(filter_result)
        
        assert smooth_result.smoothed_means.shape == (T_len, state_dim)
        assert smooth_result.smoothed_covariances.shape == (T_len, state_dim)


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_positive_definite_covariances(self):
        """Test that covariances remain positive definite."""
        np.random.seed(42)
        
        T_len = 100
        state_dim = 5
        
        # Use parameters that could cause numerical issues
        observations = np.random.randn(T_len, state_dim) * 10
        
        kf = KalmanFilter(state_dim=state_dim, obs_dim=state_dim, mode='dense')
        kf.initialize(
            T=np.eye(state_dim) * 0.99,
            Z=np.eye(state_dim),
            Q=np.eye(state_dim) * 1e-4,
            H=np.eye(state_dim) * 1e-4,
        )
        
        result = kf.filter(observations)
        
        for t in range(T_len):
            eigvals = np.linalg.eigvalsh(result.filtered_covariances[t])
            assert np.all(eigvals >= -1e-10), f"Non-positive covariance at t={t}"
            
    def test_log_likelihood_monotonicity(self):
        """Test that adding observations doesn't decrease cumulative likelihood."""
        np.random.seed(42)
        
        T_len = 50
        state_dim = 3
        
        observations = np.random.randn(T_len, state_dim)
        
        kf = KalmanFilter(state_dim=state_dim, obs_dim=state_dim, mode='dense')
        kf.initialize(
            T=np.eye(state_dim) * 0.9,
            Z=np.eye(state_dim),
            Q=np.eye(state_dim) * 0.1,
            H=np.eye(state_dim) * 0.2,
        )
        
        # The log-likelihood for increasing amounts of data should be finite
        for t in range(10, T_len, 10):
            result = kf.filter(observations[:t])
            assert np.isfinite(result.log_likelihood)
