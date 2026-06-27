"""
Tests for the StateSpaceModel orchestration layer, specifically the
augmented joint estimation of regression effects (GAM coefficient beta,
forcing coefficients B_tilde) introduced to replace the previous
pre-estimate-via-OLS-then-subtract design (Harvey, 1989, Ch. 3.3;
Watson & Engle, 1983).
"""

import numpy as np

from gam_ssm_lur.models.state_space import StateSpaceModel


class TestStateSpaceModelPlain:
    """Backward-compatibility: no regression effects requested."""

    def test_fit_without_forcing_or_g_tilde_uses_plain_path(self):
        np.random.seed(0)
        T_len, k = 100, 2
        observations = np.random.randn(T_len, k) * 0.5

        ssm = StateSpaceModel(state_dim=k, em_max_iter=20, scalability_mode="dense")
        ssm.fit(observations)

        assert ssm.is_fitted_
        assert ssm.beta_ is None
        assert ssm.B_ is None
        assert ssm.Z_.shape == (k, k)  # fixed Z, not time-varying

        pred = ssm.predict()
        assert pred.mean.shape == (T_len, k)


class TestStateSpaceModelAugmented:
    """Joint estimation of beta (GAM) and B_tilde (forcing) via augmented EM."""

    def test_fit_recovers_known_beta_and_forcing_coefficients(self):
        """End-to-end: build a synthetic score series with known beta, B_tilde,
        and AR(1) dynamics; verify StateSpaceModel.fit(g_tilde=..., forcing_matrix=...)
        recovers them through the public API (not just the underlying EM unit).
        """
        np.random.seed(7)
        k = 2  # score/state dimension
        T_len = 350
        true_T = np.array([[0.75, 0.0], [0.0, 0.6]])
        true_Q = np.diag([0.03, 0.02])
        true_H = np.diag([0.1, 0.1])
        true_beta = 1.5
        true_B = np.array([[0.8, -0.4], [0.3, 0.2]])  # (k, n_forcing)

        g_tilde = np.array([0.6, -0.3])  # fixed, known GAM-projection regressor

        n_forcing = 2
        forcing = np.column_stack(
            [
                0.5 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, T_len)),
                0.5 + 0.5 * np.cos(np.linspace(0, 5 * np.pi, T_len)),
            ]
        )

        alpha = np.zeros((T_len, k))
        for t in range(1, T_len):
            noise = np.random.multivariate_normal(np.zeros(k), true_Q)
            alpha[t] = true_T @ alpha[t - 1] + noise

        obs_noise = np.random.multivariate_normal(np.zeros(k), true_H, size=T_len)
        scores = (
            alpha + true_beta * g_tilde[np.newaxis, :] + forcing @ true_B.T + obs_noise
        )

        ssm = StateSpaceModel(state_dim=k, em_max_iter=80, em_tol=1e-7, random_state=0)
        ssm.fit(scores, forcing_matrix=forcing, g_tilde=g_tilde)

        assert ssm.is_fitted_
        assert ssm.Z_.shape == (
            T_len,
            k,
            k + 1 + k * n_forcing,
        )  # time-varying, augmented

        # Jointly estimated regression effects close to truth.
        assert ssm.beta_ is not None
        assert abs(ssm.beta_ - true_beta) < 0.5
        assert ssm.B_.shape == (k, n_forcing)
        assert np.max(np.abs(ssm.B_ - true_B)) < 0.5

        # Dynamic block recovered too.
        assert np.max(np.abs(ssm.T_[:k, :k] - true_T)) < 0.3

        # Static block of T/Q must be exactly identity/zero (never perturbed).
        aug_dim = ssm.T_.shape[0]
        np.testing.assert_allclose(ssm.T_[k:, k:], np.eye(aug_dim - k))
        np.testing.assert_allclose(
            ssm.Q_[k:, k:], np.zeros((aug_dim - k, aug_dim - k)), atol=1e-10
        )

        # predict() returns only the alpha-block (k,), not the full augmented state.
        pred = ssm.predict()
        assert pred.mean.shape == (T_len, k)

    def test_fit_with_only_g_tilde_no_forcing(self):
        """beta alone, no forcing covariates -- smaller augmented state."""
        np.random.seed(3)
        k = 1
        T_len = 200
        true_beta = -2.0
        g_tilde = np.array([0.9])

        alpha = np.zeros((T_len, k))
        for t in range(1, T_len):
            alpha[t] = 0.5 * alpha[t - 1] + np.random.normal(0, 0.1, k)
        scores = (
            alpha
            + true_beta * g_tilde[np.newaxis, :]
            + np.random.normal(0, 0.1, (T_len, k))
        )

        ssm = StateSpaceModel(state_dim=k, em_max_iter=60, em_tol=1e-7, random_state=1)
        ssm.fit(scores, g_tilde=g_tilde)

        assert ssm.beta_ is not None
        assert abs(ssm.beta_ - true_beta) < 0.5
        assert ssm.B_ is None
        assert ssm.Z_.shape == (T_len, k, k + 1)
