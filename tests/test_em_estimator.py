"""
Tests for EM parameter estimation.
"""

import numpy as np
import pytest

from gam_ssm_lur.em_estimator import EMEstimator, EMResult


class TestEMEstimator:
    """Tests for EMEstimator class."""
    
    def test_initialization(self):
        """Test EMEstimator initialization."""
        em = EMEstimator(max_iterations=50, tolerance=1e-6)
        assert em.max_iterations == 50
        assert em.tolerance == 1e-6
        
    def test_fit_simple_model(self):
        """Test EM on simple model with known parameters."""
        np.random.seed(42)
        
        # True parameters
        true_T = np.eye(3) * 0.9
        true_Q = np.eye(3) * 0.1
        true_H = np.eye(3) * 0.2
        
        # Generate data
        T_len = 200
        state_dim = 3
        
        states = np.zeros((T_len, state_dim))
        observations = np.zeros((T_len, state_dim))
        
        for t in range(T_len):
            if t > 0:
                states[t] = true_T @ states[t-1] + np.random.multivariate_normal(
                    np.zeros(state_dim), true_Q
                )
            observations[t] = states[t] + np.random.multivariate_normal(
                np.zeros(state_dim), true_H
            )
            
        # Fit model
        em = EMEstimator(max_iterations=30, tolerance=1e-6, verbose=False)
        result = em.fit(observations, state_dim=state_dim)
        
        assert isinstance(result, EMResult)
        assert result.T.shape == (state_dim, state_dim)
        assert result.Q.shape == (state_dim, state_dim)
        assert result.H.shape == (state_dim, state_dim)
        
        # Check convergence
        assert len(result.log_likelihoods) > 1
        
        # Log-likelihood should increase (or stay same) monotonically
        for i in range(1, len(result.log_likelihoods)):
            assert result.log_likelihoods[i] >= result.log_likelihoods[i-1] - 1e-4
            
    def test_parameter_recovery(self):
        """Test that EM can approximately recover true parameters."""
        np.random.seed(123)
        
        # Simple diagonal model for easier recovery
        state_dim = 2
        T_len = 500
        
        true_T = np.diag([0.8, 0.7])
        true_Q = np.diag([0.05, 0.08])
        true_H = np.diag([0.1, 0.15])
        
        # Generate data
        states = np.zeros((T_len, state_dim))
        observations = np.zeros((T_len, state_dim))
        
        for t in range(T_len):
            if t > 0:
                states[t] = true_T @ states[t-1] + np.random.multivariate_normal(
                    np.zeros(state_dim), true_Q
                )
            observations[t] = states[t] + np.random.multivariate_normal(
                np.zeros(state_dim), true_H
            )
            
        # Fit with diagonal constraint
        em = EMEstimator(
            max_iterations=50,
            tolerance=1e-7,
            diagonal_Q=True,
            diagonal_H=True,
            verbose=False,
        )
        result = em.fit(observations, state_dim=state_dim)
        
        # Check parameter recovery (allow 30% relative error)
        np.testing.assert_allclose(
            np.diag(result.T), np.diag(true_T), rtol=0.3
        )
        
    def test_convergence_flag(self):
        """Test convergence detection."""
        np.random.seed(42)
        
        observations = np.random.randn(100, 3)
        
        # With tight tolerance, should converge
        em = EMEstimator(max_iterations=100, tolerance=1e-4, verbose=False)
        result = em.fit(observations, state_dim=3)
        
        # Should either converge or hit max iterations
        assert result.n_iterations <= 100
        
    def test_diagnostics_history(self):
        """Test that diagnostics are recorded."""
        np.random.seed(42)
        
        observations = np.random.randn(50, 2)
        
        em = EMEstimator(max_iterations=20, verbose=False)
        result = em.fit(observations, state_dim=2)
        
        # Should have diagnostics
        assert len(em.diagnostics_history) == result.n_iterations
        
        # Each diagnostic should have required fields
        for diag in em.diagnostics_history:
            assert hasattr(diag, 'iteration')
            assert hasattr(diag, 'log_likelihood')
            assert hasattr(diag, 'll_change')
            assert hasattr(diag, 'param_changes')
            
    def test_fixed_parameters(self):
        """Test fixing some parameters during estimation."""
        np.random.seed(42)
        
        observations = np.random.randn(100, 3)
        fixed_T = np.eye(3) * 0.95
        
        em = EMEstimator(
            max_iterations=20,
            estimate_T=False,  # Fix T
            verbose=False,
        )
        result = em.fit(observations, state_dim=3, T_init=fixed_T)
        
        # T should remain at initial value
        np.testing.assert_array_almost_equal(result.T, fixed_T)
        
    def test_scalability_modes(self):
        """Test different scalability modes."""
        np.random.seed(42)
        
        for mode in ['dense', 'diagonal']:
            observations = np.random.randn(50, 5)
            
            em = EMEstimator(max_iterations=10, verbose=False)
            result = em.fit(observations, state_dim=5, scalability_mode=mode)
            
            assert result.T.shape == (5, 5)
            assert np.isfinite(result.log_likelihoods[-1])
            

class TestEMNumericalStability:
    """Tests for EM numerical stability."""
    
    def test_near_singular_observations(self):
        """Test handling of near-singular observation covariance."""
        np.random.seed(42)
        
        # Observations with very low variance
        observations = np.random.randn(100, 3) * 0.01
        
        em = EMEstimator(max_iterations=20, regularization=1e-6, verbose=False)
        result = em.fit(observations, state_dim=3)
        
        assert np.isfinite(result.log_likelihoods[-1])
        assert np.all(np.isfinite(result.T))
        assert np.all(np.isfinite(result.Q))
        assert np.all(np.isfinite(result.H))
        
    def test_highly_correlated_observations(self):
        """Test with highly correlated observation dimensions."""
        np.random.seed(42)
        
        # Create correlated observations
        base = np.random.randn(100, 1)
        observations = np.hstack([base, base + 0.1 * np.random.randn(100, 1)])
        observations = np.hstack([observations, base + 0.2 * np.random.randn(100, 1)])
        
        em = EMEstimator(max_iterations=20, verbose=False)
        result = em.fit(observations, state_dim=3)
        
        assert np.isfinite(result.log_likelihoods[-1])
        
    def test_long_time_series(self):
        """Test with longer time series."""
        np.random.seed(42)
        
        T_len = 1000
        state_dim = 2
        
        observations = np.random.randn(T_len, state_dim)
        
        em = EMEstimator(max_iterations=30, verbose=False)
        result = em.fit(observations, state_dim=state_dim)
        
        assert np.isfinite(result.log_likelihoods[-1])
        assert result.smoothed_states.shape == (T_len, state_dim)
