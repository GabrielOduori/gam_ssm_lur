"""
Tests for EM parameter estimation.
"""

import numpy as np
import pytest

from gam_ssm_lur.inference.em import EMEstimator, EMResult


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
                states[t] = true_T @ states[t - 1] + np.random.multivariate_normal(
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
            assert result.log_likelihoods[i] >= result.log_likelihoods[i - 1] - 1e-4

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
                states[t] = true_T @ states[t - 1] + np.random.multivariate_normal(
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
        np.testing.assert_allclose(np.diag(result.T), np.diag(true_T), rtol=0.3)

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
            assert hasattr(diag, "iteration")
            assert hasattr(diag, "log_likelihood")
            assert hasattr(diag, "ll_change")
            assert hasattr(diag, "param_changes")

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

        for mode in ["dense", "diagonal"]:
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

    def test_dynamic_dim_holds_static_block_at_zero_process_noise(self):
        """dynamic_dim=1: state[0] is a dynamic AR(1) factor, state[1] is a
        true constant (zero process noise). Verifies EM (a) recovers the
        AR(1) parameters for the dynamic block, (b) never perturbs the
        static block's T/Q away from identity/zero, and (c) the smoothed
        constant converges close to its true value (Harvey, 1989, Ch. 3.3).
        """
        np.random.seed(0)
        T_len = 300
        true_T_dyn, true_Q_dyn, true_H = 0.8, 0.05, 0.2
        true_beta = 3.0

        alpha = np.zeros(T_len)
        for t in range(1, T_len):
            alpha[t] = true_T_dyn * alpha[t - 1] + np.random.normal(
                0, np.sqrt(true_Q_dyn)
            )

        # obs_dim = state_dim = 2; EMEstimator's internal Z defaults to
        # identity, so y[:,0] observes the dynamic factor and y[:,1]
        # observes the constant beta directly.
        observations = np.column_stack(
            [
                alpha + np.random.normal(0, np.sqrt(true_H), T_len),
                np.full(T_len, true_beta) + np.random.normal(0, np.sqrt(true_H), T_len),
            ]
        )

        em = EMEstimator(
            max_iterations=50, tolerance=1e-7, verbose=False, dynamic_dim=1
        )
        result = em.fit(observations, state_dim=2, obs_dim=2)

        # Dynamic block recovered reasonably close to truth.
        assert abs(result.T[0, 0] - true_T_dyn) < 0.15
        assert abs(result.Q[0, 0] - true_Q_dyn) < 0.05

        # Static block must be untouched: identity transition, exactly zero
        # process noise, and no leakage into/out of the dynamic block.
        assert result.T[1, 1] == pytest.approx(1.0)
        assert result.T[0, 1] == pytest.approx(0.0)
        assert result.T[1, 0] == pytest.approx(0.0)
        assert result.Q[1, 1] == pytest.approx(0.0, abs=1e-12)
        assert result.Q[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert result.Q[1, 0] == pytest.approx(0.0, abs=1e-12)

        # Smoothed constant should converge close to the true beta.
        smoothed_beta = result.smoothed_states[:, 1]
        assert abs(smoothed_beta[-1] - true_beta) < 0.3
        # And should vary very little across time (it's a true constant).
        assert smoothed_beta.std() < 0.2

    def test_time_varying_Z_with_regression_effect_recovers_truth(self):
        """Full augmented model: state[0] = dynamic AR(1) factor with a
        time-varying loading g_t; state[1] = a true constant beta with
        time-varying loading u_t. y_t = g_t*alpha_t + u_t*beta + noise.

        This is the exact structure needed for the GAM/forcing joint
        estimation (Watson & Engle, 1983; Harvey, 1989, Ch. 3.3): a known,
        time-varying regression loading multiplying an unknown constant,
        estimated jointly with the dynamic factor via the same EM loop.
        Verifies the H update (now Z_t-aware) still recovers correct noise
        variance despite Z changing every timestep.
        """
        np.random.seed(1)
        T_len = 400
        true_T_dyn, true_Q_dyn, true_H = 0.7, 0.04, 0.15
        true_beta = -2.0

        alpha = np.zeros(T_len)
        for t in range(1, T_len):
            alpha[t] = true_T_dyn * alpha[t - 1] + np.random.normal(
                0, np.sqrt(true_Q_dyn)
            )

        # Known, time-varying loadings (e.g. g_t ~ GAM-projection-like signal,
        # u_t ~ a forcing variable), both bounded away from zero so beta is
        # identifiable throughout.
        g_t = 0.5 + 0.5 * np.sin(np.linspace(0, 6 * np.pi, T_len))
        u_t = 0.5 + 0.5 * np.cos(np.linspace(0, 4 * np.pi, T_len))

        y = g_t * alpha + u_t * true_beta + np.random.normal(0, np.sqrt(true_H), T_len)
        observations = y.reshape(-1, 1)

        # Z_t = [[g_t, u_t]], obs_dim=1, state_dim=2 (state[0]=alpha, state[1]=beta)
        Z_seq = np.stack([np.array([[g_t[t], u_t[t]]]) for t in range(T_len)])

        em = EMEstimator(
            max_iterations=80, tolerance=1e-8, verbose=False, dynamic_dim=1
        )
        result = em.fit(
            observations, state_dim=2, obs_dim=1, Z=Z_seq, scalability_mode="dense"
        )

        # Dynamic block recovered.
        assert abs(result.T[0, 0] - true_T_dyn) < 0.2
        assert abs(result.Q[0, 0] - true_Q_dyn) < 0.05

        # Static block (beta) untouched by T/Q.
        assert result.T[1, 1] == pytest.approx(1.0)
        assert result.Q[1, 1] == pytest.approx(0.0, abs=1e-12)

        # H recovered despite time-varying Z (this is the part that would
        # have been wrong under the old "pool S11 then apply one Z" formula).
        assert abs(result.H[0, 0] - true_H) < 0.08

        # Smoothed beta close to truth and stable across time.
        smoothed_beta = result.smoothed_states[:, 1]
        assert abs(smoothed_beta[-1] - true_beta) < 0.4
        assert smoothed_beta.std() < 0.3

    def test_stabilize_transition_clips_explosive_estimate(self):
        """If the data is generated by a non-stationary (random-walk-like)
        process, the unconstrained M-step estimate T = S10 @ S00^-1 can have
        spectral radius >= 1. Verify the EM's stability constraint (Hamilton,
        1994, Ch. 1) catches this and returns a transition matrix with
        spectral radius capped at max_eigenvalue.
        """
        np.random.seed(11)
        T_len = 30
        state_dim = 3

        # Near-random-walk generating process (true spectral radius > 1),
        # deliberately short series -- this is exactly the short-T, drifting
        # regime where an unconstrained OLS-style T estimate is prone to
        # overshoot into the unstable region.
        true_T = np.diag([1.05, 0.9, 0.5])
        state = np.zeros((T_len, state_dim))
        for t in range(1, T_len):
            state[t] = true_T @ state[t - 1] + np.random.normal(0, 1.0, state_dim)
        observations = state + np.random.normal(0, 0.5, (T_len, state_dim))

        em = EMEstimator(
            max_iterations=30, tolerance=1e-6, verbose=False, max_eigenvalue=0.98
        )
        result = em.fit(observations, state_dim=state_dim)

        spectral_radius = np.max(np.abs(np.linalg.eigvals(result.T)))
        assert spectral_radius <= 0.98 + 1e-6, (
            f"Spectral radius {spectral_radius:.4f} exceeds the configured "
            f"stability threshold; the stabilization step did not engage."
        )

    def test_stabilize_transition_leaves_stable_matrices_unchanged(self):
        """A genuinely stable transition matrix must pass through unmodified."""
        em = EMEstimator(max_eigenvalue=0.98)
        T_stable = np.diag([0.7, 0.5, -0.3])
        T_out = em._stabilize_transition(T_stable)
        np.testing.assert_allclose(T_out, T_stable)
