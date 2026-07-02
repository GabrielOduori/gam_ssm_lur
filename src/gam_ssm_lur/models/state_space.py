"""
High-level state-space model for temporal dynamics in spatiotemporal NO2
data, wrapping Kalman filter / RTS smoother / EM (gam_ssm_lur.inference)
behind a fit/predict interface.

References
----------
.. [1] Harvey, A. C. (1989). Forecasting, structural time series models and
       the Kalman filter. Cambridge University Press.
.. [2] Durbin, J., & Koopman, S. J. (2012). Time series analysis by state
       space methods. Oxford University Press.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gam_ssm_lur.inference import (
    FilterResult,
    KalmanFilter,
    RTSSmoother,
    SmootherResult,
)
from gam_ssm_lur.inference.em import EMEstimator, EMResult

logger = logging.getLogger(__name__)


@dataclass
class SSMPrediction:
    """Smoothed/forecast arrays (T, n_locations): mean, std, lower, upper."""

    mean: NDArray
    std: NDArray
    lower: NDArray
    upper: NDArray
    smoothed_states: NDArray
    filtered_states: NDArray


@dataclass
class SSMDiagnostics:
    """Diagnostics: LL/AIC/BIC, convergence, T-eigenvalues, Q/H, B_tilde."""

    log_likelihood: float
    aic: float
    bic: float
    em_converged: bool
    em_iterations: int
    transition_eigenvalues: NDArray
    process_noise_variance: float
    observation_noise_variance: float
    forcing_coefficients: Optional[NDArray] = None


class StateSpaceModel:
    """Linear Gaussian SSM with optional external forcing, fit via EM:

        Measurement: y_t = Z alpha_t + eps_t,             eps_t ~ N(0, H)
        Transition:  alpha_{t+1} = T alpha_t + B u_t + R eta_t,  eta_t ~ N(0, Q)

    u_t is a vector of external forcing covariates at time t (e.g. traffic
    anomaly, wind). B_tilde and the GAM-regression coefficient beta are
    "fixed effect" states (identity transition, zero process noise; Harvey,
    1989, Ch. 3.3) jointly estimated alongside alpha's dynamics by an
    augmented-state EM -- not pre-estimated by OLS and subtracted, which
    would silently discard their estimation error into the dynamics. With no
    forcing_matrix/g_tilde, this reduces to the plain AR(1) SSM (B_tilde = 0).
    """

    def __init__(
        self,
        state_dim: Optional[int] = None,
        em_max_iter: int = 50,
        em_tol: float = 1e-4,
        scalability_mode: Literal["auto", "dense", "diagonal", "block"] = "auto",
        regularization: float = 1e-6,
        estimate_T: bool = True,
        estimate_Q: bool = True,
        estimate_H: bool = True,
        diagonal_covariances: bool = True,
        forcing_coefficients: Optional[NDArray] = None,
        random_state: Optional[int] = None,
        max_eigenvalue: float = 0.98,
    ):
        self.state_dim = state_dim
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.scalability_mode = scalability_mode
        self.regularization = regularization
        self.estimate_T = estimate_T
        self.estimate_Q = estimate_Q
        self.estimate_H = estimate_H
        self.diagonal_covariances = diagonal_covariances
        self.forcing_coefficients = forcing_coefficients
        self.random_state = random_state
        # Max spectral radius for the dynamic-block T (Hamilton, 1994, Ch. 1
        # stationarity condition). Lower -> more aggressive drift suppression,
        # less flexible AR fit.
        self.max_eigenvalue = max_eigenvalue

        self.T_: Optional[NDArray] = None
        self.Z_: Optional[NDArray] = None  # (obs_dim, state_dim), or with
        # regression effects (T, obs_dim, aug_dim)
        self.Q_: Optional[NDArray] = None
        self.H_: Optional[NDArray] = None
        self.B_: Optional[NDArray] = (
            None  # B_tilde, jointly estimated (state_dim, n_forcing)
        )
        self.beta_: Optional[float] = (
            None  # GAM-regression coefficient, jointly estimated
        )
        self.initial_mean_: Optional[NDArray] = None
        self.initial_cov_: Optional[NDArray] = None

        self.em_result_: Optional[EMResult] = None
        self.kf_: Optional[KalmanFilter] = None
        self.filter_result_: Optional[FilterResult] = None
        self.smoother_result_: Optional[SmootherResult] = None
        self._forcing_matrix: Optional[NDArray] = None  # stored for forecast()
        self._g_tilde: Optional[NDArray] = (
            None  # projected GAM regressor, stored for forecast()
        )
        self._dynamic_dim: Optional[int] = None  # size of the alpha (dynamic) sub-block

        self.is_fitted_ = False
        self._obs_dim: Optional[int] = None
        self._T_len: Optional[int] = None

    def _build_em_estimator(self, dynamic_dim: Optional[int] = None) -> EMEstimator:
        """Shared EMEstimator config for both the plain and augmented fit paths,
        so their hyperparameters can't drift apart."""
        return EMEstimator(
            max_iterations=self.em_max_iter,
            tolerance=self.em_tol,
            regularization=self.regularization,
            estimate_T=self.estimate_T,
            estimate_Q=self.estimate_Q,
            estimate_H=self.estimate_H,
            diagonal_Q=self.diagonal_covariances,
            diagonal_H=self.diagonal_covariances,
            verbose=True,
            dynamic_dim=dynamic_dim,
            max_eigenvalue=self.max_eigenvalue,
        )

    def fit(
        self,
        observations: Union[NDArray, pd.DataFrame],
        forcing_matrix: Optional[Union[NDArray, pd.DataFrame]] = None,
        g_tilde: Optional[NDArray] = None,
        T_init: Optional[NDArray] = None,
        Q_init: Optional[NDArray] = None,
        H_init: Optional[NDArray] = None,
    ) -> StateSpaceModel:
        """Fit via EM. observations: (T, obs_dim) score-space (e.g. top-k SVD
        scores of the calibrated satellite field). forcing_matrix: (T, n_forcing)
        external covariates entering as y_t = alpha_t + B_tilde @ u_t (+ beta *
        g_tilde) + noise. g_tilde: (obs_dim,) known time-invariant regressor
        (e.g. a static GAM baseline projected into score-space). T_init/Q_init/
        H_init apply to the dynamic (alpha) sub-block only.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if isinstance(observations, pd.DataFrame):
            observations = observations.values
        observations = np.asarray(observations)

        self._T_len, self._obs_dim = observations.shape
        state_dim = self.state_dim or self._obs_dim
        self._dynamic_dim = (
            state_dim  # stored so predict()/forecast() know the alpha-block size
        )

        has_beta = g_tilde is not None
        if forcing_matrix is not None:
            if isinstance(forcing_matrix, pd.DataFrame):
                forcing_matrix = forcing_matrix.values
            forcing_matrix = np.asarray(forcing_matrix, dtype=float)
            if forcing_matrix.shape[0] != self._T_len:
                raise ValueError(
                    f"forcing_matrix has {forcing_matrix.shape[0]} rows but "
                    f"observations has {self._T_len} time steps."
                )
            n_forcing = forcing_matrix.shape[1]
        else:
            n_forcing = 0
        self._forcing_matrix = forcing_matrix
        self._g_tilde = np.asarray(g_tilde, dtype=float) if has_beta else None

        if not has_beta and n_forcing == 0:
            # no regression effects: plain dynamics-only model, unchanged from
            # the pre-augmentation behaviour
            return self._fit_plain(observations, state_dim, T_init, Q_init, H_init)

        # Augmented-state joint estimation
        # ---------------------------------
        # State: [alpha (state_dim); beta (1 if has_beta); vec(B_tilde)
        # (state_dim * n_forcing)]. beta and vec(B_tilde) are fixed-effect
        # states (Harvey, 1989, Ch. 3.3), estimated jointly with alpha's
        # dynamics by the augmented EM.
        beta_col = state_dim if has_beta else None
        b_tilde_start = state_dim + (1 if has_beta else 0)
        aug_dim = state_dim + (1 if has_beta else 0) + state_dim * n_forcing

        Z_seq = np.zeros((self._T_len, self._obs_dim, aug_dim))
        eye_block = np.eye(self._obs_dim, state_dim)
        for t in range(self._T_len):
            Zt = Z_seq[t]
            Zt[:, :state_dim] = eye_block
            if has_beta:
                Zt[:, beta_col] = self._g_tilde
            for j in range(n_forcing):
                start = b_tilde_start + j * state_dim
                Zt[:, start : start + state_dim] = forcing_matrix[t, j] * eye_block

        logger.info(
            "Fitting SSM with joint regression effects: T=%d, obs_dim=%d, "
            "state_dim=%d, aug_dim=%d (beta=%s, n_forcing=%d), mode=dense (forced)",
            self._T_len,
            self._obs_dim,
            state_dim,
            aug_dim,
            has_beta,
            n_forcing,
        )

        # time-varying Z requires dense mode (KalmanFilter constraint); aug_dim
        # is small so this is cheap regardless of the configured scalability_mode
        em = self._build_em_estimator(dynamic_dim=state_dim)

        self.em_result_ = em.fit(
            observations=observations,
            state_dim=aug_dim,
            obs_dim=self._obs_dim,
            T_init=T_init,
            Q_init=Q_init,
            H_init=H_init,
            Z=Z_seq,
            scalability_mode="dense",
        )

        self.T_ = self.em_result_.T
        self.Z_ = Z_seq
        self.Q_ = self.em_result_.Q
        self.H_ = self.em_result_.H
        self.initial_mean_ = self.em_result_.initial_mean
        self.initial_cov_ = self.em_result_.initial_covariance

        # point estimates for the fixed-effect states: average the smoothed
        # mean across t (should be ~constant; averaging guards against small
        # numerical drift rather than relying on a single timestep)
        smoothed = self.em_result_.smoothed_states
        self.beta_ = float(smoothed[:, beta_col].mean()) if has_beta else None
        if n_forcing > 0:
            b_flat = smoothed[:, b_tilde_start:].mean(axis=0)  # (n_forcing*state_dim,)
            self.B_ = b_flat.reshape(n_forcing, state_dim).T  # (state_dim, n_forcing)
        else:
            self.B_ = None
        logger.info(
            "Jointly estimated beta=%s, B_tilde=%s",
            None if self.beta_ is None else round(self.beta_, 4),
            None if self.B_ is None else np.round(self.B_, 4).tolist(),
        )

        self.kf_ = KalmanFilter(
            state_dim=aug_dim,
            obs_dim=self._obs_dim,
            mode="dense",
            regularization=self.regularization,
        )
        self.kf_.initialize(
            T=self.T_,
            Z=self.Z_,
            Q=self.Q_,
            H=self.H_,
            initial_mean=self.initial_mean_,
            initial_covariance=self.initial_cov_,
        )
        self.filter_result_ = self.kf_.filter(observations)
        smoother = RTSSmoother(self.kf_)
        self.smoother_result_ = smoother.smooth(self.filter_result_)

        self.is_fitted_ = True
        logger.info(
            f"SSM fitted. Converged: {self.em_result_.converged}, "
            f"Iterations: {self.em_result_.n_iterations}, "
            f"Final LL: {self.em_result_.log_likelihoods[-1]:.4e}"
        )
        return self

    def _fit_plain(
        self,
        observations: NDArray,
        state_dim: int,
        T_init: Optional[NDArray],
        Q_init: Optional[NDArray],
        H_init: Optional[NDArray],
    ) -> StateSpaceModel:
        """Fit with no regression effects: dynamics-only EM (original behaviour)."""
        self._forcing_matrix = None
        self._g_tilde = None
        self.B_ = None
        self.beta_ = None

        logger.info(
            "Fitting SSM: T=%d, obs_dim=%d, state_dim=%d, mode=%s",
            self._T_len,
            self._obs_dim,
            state_dim,
            self.scalability_mode,
        )

        em = self._build_em_estimator()

        self.em_result_ = em.fit(
            observations=observations,
            state_dim=state_dim,
            obs_dim=self._obs_dim,
            T_init=T_init,
            Q_init=Q_init,
            H_init=H_init,
            scalability_mode=self.scalability_mode,
        )

        self.T_ = self.em_result_.T
        self.Z_ = np.eye(self._obs_dim, state_dim)
        self.Q_ = self.em_result_.Q
        self.H_ = self.em_result_.H
        self.initial_mean_ = self.em_result_.initial_mean
        self.initial_cov_ = self.em_result_.initial_covariance

        self.kf_ = KalmanFilter(
            state_dim=state_dim,
            obs_dim=self._obs_dim,
            mode=self.scalability_mode,
            regularization=self.regularization,
        )
        self.kf_.initialize(
            T=self.T_,
            Z=self.Z_,
            Q=self.Q_,
            H=self.H_,
            initial_mean=self.initial_mean_,
            initial_covariance=self.initial_cov_,
        )
        self.filter_result_ = self.kf_.filter(observations)
        smoother = RTSSmoother(self.kf_)
        self.smoother_result_ = smoother.smooth(self.filter_result_)

        self.is_fitted_ = True
        logger.info(
            f"SSM fitted. Converged: {self.em_result_.converged}, "
            f"Iterations: {self.em_result_.n_iterations}, "
            f"Final LL: {self.em_result_.log_likelihoods[-1]:.4e}"
        )
        return self

    def predict(self, confidence_level: float = 0.95) -> SSMPrediction:
        """Smoothed mean/std/interval for the dynamic (alpha) sub-block only.

        beta and B_tilde are separate, jointly-estimated fixed-effect states
        (self.beta_, self.B_) -- they were never subtracted from the
        observations, so there's nothing to "add back" here (unlike the old
        OLS-based design). Callers needing the full score-level reconstruction
        (alpha_t + beta*g_tilde + B_tilde@u_t) combine self.beta_/self.B_/
        self._g_tilde/self._forcing_matrix with this alpha_t themselves (e.g.
        hybrid.py reconstructs the per-location field with the unprojected
        GAM baseline rather than its score-space projection).
        """
        self._check_fitted()

        from scipy import stats

        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        d = self._dynamic_dim
        smoothed_means = self.smoother_result_.smoothed_means[:, :d].copy()
        smoothed_covs_full = self.smoother_result_.smoothed_covariances
        if smoothed_covs_full.ndim == 3:
            smoothed_covs = smoothed_covs_full[:, :d, :d]
        else:
            smoothed_covs = smoothed_covs_full[:, :d]

        if smoothed_covs.ndim == 2:
            std = np.sqrt(smoothed_covs)  # diagonal mode: covariances stored as vectors
        else:
            std = np.sqrt(np.diagonal(smoothed_covs, axis1=1, axis2=2))

        lower = smoothed_means - z_score * std
        upper = smoothed_means + z_score * std

        return SSMPrediction(
            mean=smoothed_means,
            std=std,
            lower=lower,
            upper=upper,
            smoothed_states=smoothed_means,
            filtered_states=self.filter_result_.filtered_means[:, :d],
        )

    def forecast(self, n_steps: int, confidence_level: float = 0.95) -> SSMPrediction:
        """Forecast n_steps ahead for the dynamic (alpha) sub-block only.

        beta/B_tilde are fixed-effect constants with no further dynamics to
        extrapolate, and forecasting them would require future g_tilde/forcing
        values this method doesn't take as input.
        """
        self._check_fitted()

        from scipy import stats

        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        d = self._dynamic_dim
        T_dd = self.T_[:d, :d]
        Q_dd = self.Q_[:d, :d]
        Z_dd = np.eye(self._obs_dim, d)

        current_mean = self.filter_result_.filtered_means[-1][:d]
        last_cov = self.filter_result_.filtered_covariances[-1]
        current_cov = last_cov[:d, :d] if last_cov.ndim == 2 else np.diag(last_cov[:d])

        forecast_means = np.zeros((n_steps, self._obs_dim))
        forecast_covs = np.zeros((n_steps, self._obs_dim, self._obs_dim))

        for t in range(n_steps):
            current_mean = T_dd @ current_mean
            current_cov = T_dd @ current_cov @ T_dd.T + Q_dd

            forecast_means[t] = Z_dd @ current_mean
            forecast_covs[t] = Z_dd @ current_cov @ Z_dd.T + self.H_

        std = np.sqrt(np.diagonal(forecast_covs, axis1=1, axis2=2))
        lower = forecast_means - z_score * std
        upper = forecast_means + z_score * std

        return SSMPrediction(
            mean=forecast_means,
            std=std,
            lower=lower,
            upper=upper,
            smoothed_states=forecast_means,
            filtered_states=forecast_means,
        )

    def get_diagnostics(self) -> SSMDiagnostics:
        self._check_fitted()

        ll = self.em_result_.log_likelihoods[-1]

        state_dim = self.T_.shape[0]
        n_params = state_dim**2  # T
        if not self.diagonal_covariances:
            n_params += state_dim * (state_dim + 1) // 2  # Q (symmetric)
            n_params += self._obs_dim * (self._obs_dim + 1) // 2  # H (symmetric)
        else:
            n_params += state_dim + self._obs_dim  # diagonal Q and H

        n_obs = self._T_len * self._obs_dim

        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + np.log(n_obs) * n_params

        eigenvalues = np.linalg.eigvals(self.T_)

        return SSMDiagnostics(
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            em_converged=self.em_result_.converged,
            em_iterations=self.em_result_.n_iterations,
            transition_eigenvalues=eigenvalues,
            process_noise_variance=np.trace(self.Q_),
            observation_noise_variance=np.trace(self.H_),
            forcing_coefficients=self.B_,
        )

    def get_em_history(self) -> pd.DataFrame:
        self._check_fitted()

        return pd.DataFrame(
            {
                "iteration": range(len(self.em_result_.log_likelihoods)),
                "log_likelihood": self.em_result_.log_likelihoods,
                "tr_T": [np.trace(self.T_)] * len(self.em_result_.log_likelihoods),
                "tr_Q": [np.trace(self.Q_)] * len(self.em_result_.log_likelihoods),
                "tr_H": [np.trace(self.H_)] * len(self.em_result_.log_likelihoods),
            }
        )

    def get_innovation_diagnostics(self, observations: NDArray) -> Dict[str, NDArray]:
        """Standardised innovations y_t - E[y_t | y_{1:t-1}] and per-location ACF."""
        self._check_fitted()

        innovations = observations - self.filter_result_.predicted_means @ self.Z_.T

        pred_covs = self.filter_result_.predicted_covariances
        if pred_covs.ndim == 2:
            std_innov = innovations / np.sqrt(pred_covs + self.regularization)
        else:
            obs_cov = self.Z_ @ pred_covs @ self.Z_.T + self.H_
            std_diag = np.sqrt(np.diagonal(obs_cov, axis1=1, axis2=2))
            std_innov = innovations / std_diag

        def acf(x, nlags=20):
            n = len(x)
            mean = np.mean(x)
            var = np.var(x)
            acf_vals = np.zeros(nlags)
            for lag in range(nlags):
                acf_vals[lag] = np.mean((x[: n - lag] - mean) * (x[lag:] - mean)) / var
            return acf_vals

        acf_matrix = np.array(
            [acf(innovations[:, i]) for i in range(innovations.shape[1])]
        )

        return {
            "innovations": innovations,
            "standardized_innovations": std_innov,
            "mean_innovation": np.mean(innovations, axis=0),
            "std_innovation": np.std(innovations, axis=0),
            "acf": acf_matrix,
        }

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def _restore_inference(self, observations: NDArray) -> None:
        """Recompute filter/smoother results for a loaded model."""
        obs = np.asarray(observations)
        if obs.ndim != 2:
            raise ValueError("observations must be 2D (T, obs_dim)")

        if self._obs_dim is None:
            self._obs_dim = obs.shape[1]
        if self._T_len is None:
            self._T_len = obs.shape[0]

        missing_mask = np.isnan(obs)
        obs_clean = np.where(missing_mask, 0.0, obs) if missing_mask.any() else obs
        missing_mask = missing_mask if missing_mask.any() else None

        # time-varying Z (regression effects) requires dense mode regardless of
        # the originally configured scalability_mode -- fit() forces this too;
        # mirrored here so reloaded models filter identically
        mode = "dense" if np.asarray(self.Z_).ndim == 3 else self.scalability_mode
        kf = KalmanFilter(
            state_dim=self.T_.shape[0],
            obs_dim=self._obs_dim,
            mode=mode,
            regularization=self.regularization,
        )
        kf.initialize(
            T=self.T_,
            Z=self.Z_,
            Q=self.Q_,
            H=self.H_,
            initial_mean=self.initial_mean_,
            initial_covariance=self.initial_cov_,
        )

        filter_result = kf.filter(obs_clean, missing_mask=missing_mask)
        smoother_result = RTSSmoother(kf).smooth(filter_result)

        self.kf_ = kf
        self.filter_result_ = filter_result
        self.smoother_result_ = smoother_result

    def save(self, filepath: str) -> None:
        import pickle

        self._check_fitted()

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "params": {
                        "state_dim": self.state_dim,
                        "em_max_iter": self.em_max_iter,
                        "em_tol": self.em_tol,
                        "scalability_mode": self.scalability_mode,
                        "regularization": self.regularization,
                        "estimate_T": self.estimate_T,
                        "estimate_Q": self.estimate_Q,
                        "estimate_H": self.estimate_H,
                        "diagonal_covariances": self.diagonal_covariances,
                    },
                    "fitted": {
                        "T": self.T_,
                        "Z": self.Z_,
                        "Q": self.Q_,
                        "H": self.H_,
                        "initial_mean": self.initial_mean_,
                        "initial_cov": self.initial_cov_,
                        "obs_dim": self._obs_dim,
                        "T_len": self._T_len,
                        "B": self.B_,
                        "forcing_matrix": self._forcing_matrix,
                        "beta": self.beta_,
                        "g_tilde": self._g_tilde,
                        "dynamic_dim": self._dynamic_dim,
                    },
                    "em_result": {
                        "log_likelihoods": self.em_result_.log_likelihoods,
                        "n_iterations": self.em_result_.n_iterations,
                        "converged": self.em_result_.converged,
                    },
                },
                f,
            )

        logger.info(f"SSM saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> StateSpaceModel:
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model = cls(**data["params"])
        model.T_ = data["fitted"]["T"]
        model.Z_ = data["fitted"]["Z"]
        model.Q_ = data["fitted"]["Q"]
        model.H_ = data["fitted"]["H"]
        model.initial_mean_ = data["fitted"]["initial_mean"]
        model.initial_cov_ = data["fitted"]["initial_cov"]
        model._obs_dim = data["fitted"]["obs_dim"]
        model._T_len = data["fitted"]["T_len"]
        model.B_ = data["fitted"].get("B", None)
        model._forcing_matrix = data["fitted"].get("forcing_matrix", None)
        model.beta_ = data["fitted"].get("beta", None)
        model._g_tilde = data["fitted"].get("g_tilde", None)
        model._dynamic_dim = data["fitted"].get("dynamic_dim", model._obs_dim)
        model.is_fitted_ = True

        if "em_result" in data:
            from gam_ssm_lur.inference.em import EMResult

            er = data["em_result"]
            model.em_result_ = EMResult(
                T=model.T_,
                Q=model.Q_,
                H=model.H_,
                initial_mean=model.initial_mean_,
                initial_covariance=model.initial_cov_,
                log_likelihoods=er["log_likelihoods"],
                n_iterations=er["n_iterations"],
                converged=er["converged"],
                smoothed_states=None,
                smoothed_covariances=None,
            )

        logger.info(f"SSM loaded from {filepath}")
        return model
