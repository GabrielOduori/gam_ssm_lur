"""
Kalman filter and Rauch-Tung-Striebel smoother for linear Gaussian SSMs.

Three matrix backends (dense / diagonal / block-diagonal) so the same
recursions scale from a handful of latent factors up to spatial grids with
thousands of locations.

References
----------
.. [1] Kalman, R. E. (1960). A new approach to linear filtering and prediction
       problems. Journal of Basic Engineering, 82(1), 35-45.
.. [2] Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum likelihood
       estimates of linear dynamic systems. AIAA Journal, 3(8), 1445-1450.
.. [3] Durbin, J., & Koopman, S. J. (2012). Time series analysis by state
       space methods. Oxford University Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class FilteredState:
    """Kalman filter output at a single time step."""

    mean: NDArray
    covariance: NDArray
    predicted_mean: NDArray
    predicted_covariance: NDArray
    kalman_gain: NDArray
    log_likelihood: float


@dataclass
class SmoothedState:
    """RTS smoother output at a single time step (cross_covariance feeds the EM M-step)."""

    mean: NDArray
    covariance: NDArray
    cross_covariance: Optional[NDArray] = None


@dataclass
class FilterResult:
    """Kalman filter output across all T time steps."""

    filtered_means: NDArray
    filtered_covariances: NDArray
    predicted_means: NDArray
    predicted_covariances: NDArray
    log_likelihood: float


class KalmanFilter:
    """Kalman filter with dense / diagonal / block-diagonal backends, chosen
    automatically from state_dim unless mode is set explicitly. Time-varying
    Z_t (regression effects on known, time-varying covariates; Durbin &
    Koopman, 2012, Sec. 3.1) is supported in dense mode only.
    """

    # For spatial SSMs state_dim = n_locations. Dense mode fits n^2 x 3
    # parameters via EM, underdetermined even for ~50 cells -- 'auto'
    # therefore defaults to diagonal for any realistic spatial grid.
    DENSE_THRESHOLD = 10  # only use dense for toy problems
    DIAGONAL_THRESHOLD = 5000  # diagonal -> block for very large grids

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        mode: Literal["auto", "dense", "diagonal", "block"] = "auto",
        block_size: int = 500,
        regularization: float = 1e-6,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.block_size = block_size
        self.regularization = regularization
        self.mode = self._select_mode(state_dim) if mode == "auto" else mode

        self.T: Optional[NDArray] = None
        self.Z: Optional[NDArray] = None
        self.Q: Optional[NDArray] = None
        self.H: Optional[NDArray] = None
        self.R: Optional[NDArray] = None

        # set in initialize() if Z is passed as a (T_len, obs_dim, state_dim) sequence
        self._time_varying_Z: bool = False
        self._Z_sequence: Optional[NDArray] = None

        self.initial_mean: Optional[NDArray] = None
        self.initial_covariance: Optional[NDArray] = None
        self._initialized = False

    def _select_mode(self, n: int) -> str:
        if n < self.DENSE_THRESHOLD:
            return "dense"
        elif n < self.DIAGONAL_THRESHOLD:
            return "diagonal"
        return "block"

    def initialize(
        self,
        T: NDArray,
        Z: NDArray,
        Q: NDArray,
        H: NDArray,
        R: Optional[NDArray] = None,
        initial_mean: Optional[NDArray] = None,
        initial_covariance: Optional[NDArray] = None,
    ) -> None:
        """Set system matrices. Z may be (obs_dim, state_dim) or, for
        time-varying observation loadings, (T_len, obs_dim, state_dim) --
        the latter requires mode='dense'.
        """
        self.T = self._convert_matrix(T)

        Z_arr = np.asarray(Z)
        self._time_varying_Z = Z_arr.ndim == 3
        if self._time_varying_Z:
            if self.mode != "dense":
                raise ValueError(
                    f"Time-varying Z requires mode='dense' (got mode={self.mode!r})."
                )
            self._Z_sequence = Z_arr
            self.Z = self._Z_sequence[0]  # placeholder; set per-timestep in filter()
        else:
            self._Z_sequence = None
            self.Z = self._convert_matrix(Z)

        self.Q = self._convert_matrix(Q)
        self.H = self._convert_matrix(H)
        self.R = self._convert_matrix(R) if R is not None else self._identity()
        self.initial_mean = (
            np.asarray(initial_mean)
            if initial_mean is not None
            else np.zeros(self.state_dim)
        )
        self.initial_covariance = (
            self._convert_matrix(initial_covariance)
            if initial_covariance is not None
            else self._identity()
        )
        self._initialized = True

    def _convert_matrix(self, M: NDArray) -> NDArray:
        if self.mode == "dense":
            return np.asarray(M)
        elif self.mode == "diagonal":
            if M.ndim == 2:
                return np.diag(M) if M.shape[0] == M.shape[1] else M
            return M
        else:  # block
            return M.tocsr() if sparse.issparse(M) else csr_matrix(M)

    def _identity(self) -> NDArray:
        if self.mode == "dense":
            return np.eye(self.state_dim)
        elif self.mode == "diagonal":
            return np.ones(self.state_dim)
        return sparse.eye(self.state_dim, format="csr")

    def _matrix_multiply(self, A: NDArray, B: NDArray) -> NDArray:
        if self.mode == "diagonal":
            # diagonal matrices stored as vectors -> elementwise
            if A.ndim == 1 and B.ndim == 1:
                return A * B
            elif A.ndim == 1:
                return A[:, np.newaxis] * B
            elif B.ndim == 1:
                return A * B
            return A @ B
        # sparse/block: avoid (n,k) @ (k,) shape errors
        if sparse.issparse(A) and B.ndim == 1:
            return np.asarray(A @ B.reshape(-1, 1)).ravel()
        if (
            sparse.issparse(A)
            and B.ndim == 2
            and B.shape[0] == 1
            and B.shape[1] == A.shape[1]
        ):
            return np.asarray(A @ B.T).ravel()
        return A @ B

    def _solve(self, A: NDArray, b: NDArray) -> NDArray:
        if self.mode == "diagonal":
            return b / (A + self.regularization)
        elif self.mode == "block":
            return spsolve(A + self.regularization * sparse.eye(A.shape[0]), b)
        return np.linalg.solve(A + self.regularization * np.eye(A.shape[0]), b)

    def _inverse(self, A: NDArray) -> NDArray:
        if self.mode == "diagonal":
            return 1.0 / (A + self.regularization)
        A_reg = A + self.regularization * np.eye(A.shape[0])
        try:
            return np.linalg.inv(A_reg)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A_reg)

    def _transpose(self, A: NDArray) -> NDArray:
        if self.mode == "diagonal":
            return A  # diagonal is symmetric
        return A.T

    def _add_matrices(self, A: NDArray, B: NDArray) -> NDArray:
        return A + B

    def _compute_log_likelihood(
        self, innovation: NDArray, innovation_covariance: NDArray
    ) -> float:
        n = len(innovation)
        if self.mode == "diagonal":
            log_det = np.sum(np.log(innovation_covariance + self.regularization))
            quad_form = np.sum(
                innovation**2 / (innovation_covariance + self.regularization)
            )
        else:
            try:
                L = np.linalg.cholesky(
                    innovation_covariance + self.regularization * np.eye(n)
                )
                log_det = 2 * np.sum(np.log(np.diag(L)))
                v = np.linalg.solve(L, innovation)
                quad_form = np.dot(v, v)
            except np.linalg.LinAlgError:
                eigvals = np.maximum(
                    np.linalg.eigvalsh(innovation_covariance), self.regularization
                )
                log_det = np.sum(np.log(eigvals))
                quad_form = (
                    innovation @ self._inverse(innovation_covariance) @ innovation
                )
        return -0.5 * (n * np.log(2 * np.pi) + log_det + quad_form)

    def _predict_step(
        self, filtered_mean: NDArray, filtered_covariance: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """alpha_{t|t-1} = T alpha_{t-1|t-1};  P_{t|t-1} = T P_{t-1|t-1} T' + R Q R'."""
        predicted_mean = self._matrix_multiply(self.T, filtered_mean)
        if self.mode == "diagonal":
            predicted_cov = self.T**2 * filtered_covariance + self.R**2 * self.Q
        else:
            T_P = self._matrix_multiply(self.T, filtered_covariance)
            T_P_T = self._matrix_multiply(T_P, self._transpose(self.T))
            R_Q = self._matrix_multiply(self.R, self.Q)
            R_Q_R = self._matrix_multiply(R_Q, self._transpose(self.R))
            predicted_cov = self._add_matrices(T_P_T, R_Q_R)
        return predicted_mean, predicted_cov

    def _update_step(
        self,
        observation: NDArray,
        predicted_mean: NDArray,
        predicted_covariance: NDArray,
    ) -> FilteredState:
        """v_t = y_t - Z a_{t|t-1};  F_t = Z P_{t|t-1} Z' + H;  K_t = P_{t|t-1} Z' F_t^-1;
        a_{t|t} = a_{t|t-1} + K_t v_t;  P_{t|t} = (I - K_t Z) P_{t|t-1}."""
        innovation = observation - self._matrix_multiply(self.Z, predicted_mean)

        if self.mode == "diagonal":
            innovation_cov = self.Z**2 * predicted_covariance + self.H
            kalman_gain = (predicted_covariance * self.Z) / (
                innovation_cov + self.regularization
            )
            filtered_mean = predicted_mean + self._matrix_multiply(
                kalman_gain, innovation
            )
            filtered_cov = (1 - kalman_gain * self.Z) * predicted_covariance
            return FilteredState(
                mean=filtered_mean,
                covariance=filtered_cov,
                predicted_mean=predicted_mean,
                predicted_covariance=predicted_covariance,
                kalman_gain=kalman_gain,
                log_likelihood=self._compute_log_likelihood(innovation, innovation_cov),
            )

        Z_P = self._matrix_multiply(self.Z, predicted_covariance)
        Z_P_Z = self._matrix_multiply(Z_P, self._transpose(self.Z))
        innovation_cov = self._add_matrices(Z_P_Z, self.H)
        P_Z = self._matrix_multiply(predicted_covariance, self._transpose(self.Z))

        if self.mode == "dense":
            innovation_cov_reg = innovation_cov + self.regularization * np.eye(
                self.obs_dim
            )
            try:
                # one Cholesky factor serves both the gain (solve F K' = P_Z') and the likelihood
                chol_factor, lower = cho_factor(
                    innovation_cov_reg, lower=False, check_finite=False
                )
                kalman_gain = cho_solve(
                    (chol_factor, lower), P_Z.T, check_finite=False
                ).T
                filtered_mean = predicted_mean + self._matrix_multiply(
                    kalman_gain, innovation
                )
                I_KZ = np.eye(self.state_dim) - self._matrix_multiply(
                    kalman_gain, self.Z
                )
                filtered_cov = self._matrix_multiply(I_KZ, predicted_covariance)
                filtered_cov = 0.5 * (filtered_cov + self._transpose(filtered_cov))

                log_det = 2.0 * np.sum(np.log(np.diag(chol_factor)))
                whitened_innov = solve_triangular(
                    chol_factor, innovation, lower=lower, check_finite=False
                )
                ll = -0.5 * (
                    self.obs_dim * np.log(2 * np.pi)
                    + log_det
                    + np.dot(whitened_innov, whitened_innov)
                )
            except np.linalg.LinAlgError:
                F_inv = self._inverse(innovation_cov)
                kalman_gain = self._matrix_multiply(P_Z, F_inv)
                filtered_mean = predicted_mean + self._matrix_multiply(
                    kalman_gain, innovation
                )
                I_KZ = np.eye(self.state_dim) - self._matrix_multiply(
                    kalman_gain, self.Z
                )
                filtered_cov = self._matrix_multiply(I_KZ, predicted_covariance)
                filtered_cov = 0.5 * (filtered_cov + self._transpose(filtered_cov))
                ll = self._compute_log_likelihood(innovation, innovation_cov)
        else:
            kalman_gain = self._matrix_multiply(P_Z, self._inverse(innovation_cov))
            filtered_mean = predicted_mean + self._matrix_multiply(
                kalman_gain, innovation
            )
            I_KZ = np.eye(self.state_dim) - self._matrix_multiply(kalman_gain, self.Z)
            filtered_cov = self._matrix_multiply(I_KZ, predicted_covariance)
            filtered_cov = 0.5 * (filtered_cov + self._transpose(filtered_cov))
            ll = self._compute_log_likelihood(innovation, innovation_cov)

        return FilteredState(
            mean=filtered_mean,
            covariance=filtered_cov,
            predicted_mean=predicted_mean,
            predicted_covariance=predicted_covariance,
            kalman_gain=kalman_gain,
            log_likelihood=ll,
        )

    def filter(
        self, observations: NDArray, missing_mask: Optional[NDArray] = None
    ) -> FilterResult:
        """Forward pass over observations, shape (T, obs_dim). missing_mask
        (same shape, bool) skips the update step for missing rows, propagating
        the prediction through unchanged."""
        if not self._initialized:
            raise RuntimeError("Filter not initialized. Call initialize() first.")

        T_len = observations.shape[0]
        filtered_means = np.zeros((T_len, self.state_dim))
        predicted_means = np.zeros((T_len, self.state_dim))
        if self.mode == "diagonal":
            filtered_covs = np.zeros((T_len, self.state_dim))
            predicted_covs = np.zeros((T_len, self.state_dim))
        elif self.mode == "block":
            filtered_covs = np.empty(
                (T_len,), dtype=object
            )  # sparse blocks, not castable to float array
            predicted_covs = np.empty((T_len,), dtype=object)
        else:
            filtered_covs = np.zeros((T_len, self.state_dim, self.state_dim))
            predicted_covs = np.zeros((T_len, self.state_dim, self.state_dim))

        total_ll = 0.0
        current_mean = self.initial_mean.copy()
        current_cov = self.initial_covariance.copy()

        for t in range(T_len):
            pred_mean, pred_cov = self._predict_step(current_mean, current_cov)

            if self._time_varying_Z:
                self.Z = self._Z_sequence[t]

            obs_t = observations[t]
            if missing_mask is not None and missing_mask[t].any():
                filtered_state = FilteredState(
                    mean=pred_mean,
                    covariance=pred_cov,
                    predicted_mean=pred_mean,
                    predicted_covariance=pred_cov,
                    kalman_gain=np.zeros_like(pred_mean),
                    log_likelihood=0.0,
                )
            else:
                filtered_state = self._update_step(obs_t, pred_mean, pred_cov)

            filtered_means[t] = filtered_state.mean
            filtered_covs[t] = filtered_state.covariance
            predicted_means[t] = filtered_state.predicted_mean
            predicted_covs[t] = filtered_state.predicted_covariance
            total_ll += filtered_state.log_likelihood

            current_mean = filtered_state.mean
            current_cov = filtered_state.covariance

        return FilterResult(
            filtered_means=filtered_means,
            filtered_covariances=filtered_covs,
            predicted_means=predicted_means,
            predicted_covariances=predicted_covs,
            log_likelihood=total_ll,
        )
