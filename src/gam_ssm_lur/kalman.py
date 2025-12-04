"""
Kalman Filter and Rauch-Tung-Striebel Smoother Implementation.

This module provides memory-efficient implementations of the Kalman filter
and RTS smoother for linear Gaussian state space models, with support for
different scalability modes (dense, diagonal, block-diagonal).

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

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import cho_factor, cho_solve, solve_triangular


@dataclass
class FilteredState:
    """Container for Kalman filter output at a single time step.
    
    Attributes
    ----------
    mean : NDArray
        Filtered state mean estimate, shape (state_dim,)
    covariance : NDArray
        Filtered state covariance, shape (state_dim, state_dim) or (state_dim,)
        for diagonal approximation
    predicted_mean : NDArray
        One-step-ahead predicted state mean
    predicted_covariance : NDArray
        One-step-ahead predicted state covariance
    kalman_gain : NDArray
        Kalman gain matrix used in update step
    log_likelihood : float
        Log-likelihood contribution from this time step
    """
    mean: NDArray
    covariance: NDArray
    predicted_mean: NDArray
    predicted_covariance: NDArray
    kalman_gain: NDArray
    log_likelihood: float


@dataclass
class SmoothedState:
    """Container for RTS smoother output at a single time step.
    
    Attributes
    ----------
    mean : NDArray
        Smoothed state mean estimate
    covariance : NDArray
        Smoothed state covariance
    cross_covariance : NDArray
        Cross-covariance E[α_t α_{t-1}^T | y_{1:T}] for EM algorithm
    """
    mean: NDArray
    covariance: NDArray
    cross_covariance: Optional[NDArray] = None


@dataclass
class FilterResult:
    """Complete Kalman filter results across all time steps.
    
    Attributes
    ----------
    filtered_means : NDArray
        Filtered state means, shape (T, state_dim)
    filtered_covariances : NDArray
        Filtered covariances, shape (T, state_dim, state_dim) or (T, state_dim)
    predicted_means : NDArray
        Predicted state means, shape (T, state_dim)
    predicted_covariances : NDArray
        Predicted covariances
    log_likelihood : float
        Total log-likelihood
    """
    filtered_means: NDArray
    filtered_covariances: NDArray
    predicted_means: NDArray
    predicted_covariances: NDArray
    log_likelihood: float


@dataclass
class SmootherResult:
    """Complete RTS smoother results across all time steps.
    
    Attributes
    ----------
    smoothed_means : NDArray
        Smoothed state means, shape (T, state_dim)
    smoothed_covariances : NDArray
        Smoothed covariances
    cross_covariances : NDArray
        Cross-covariances for EM algorithm
    """
    smoothed_means: NDArray
    smoothed_covariances: NDArray
    cross_covariances: NDArray


class KalmanFilter:
    """Memory-efficient Kalman filter with adaptive matrix representation.
    
    Implements the standard Kalman filter recursions with support for three
    scalability modes:
    - 'dense': Full matrix operations for small problems (n < 1000)
    - 'diagonal': Diagonal covariance approximation for medium problems
    - 'block': Block-diagonal decomposition for large problems (n > 5000)
    
    Parameters
    ----------
    state_dim : int
        Dimension of the latent state vector
    obs_dim : int
        Dimension of the observation vector
    mode : {'auto', 'dense', 'diagonal', 'block'}
        Scalability mode. 'auto' selects based on state_dim.
    block_size : int, optional
        Block size for block-diagonal mode. Default is 500.
    regularization : float
        Small constant added to matrices for numerical stability.
        
    Attributes
    ----------
    T : NDArray
        State transition matrix, shape (state_dim, state_dim)
    Z : NDArray
        Observation matrix, shape (obs_dim, state_dim)
    Q : NDArray
        Process noise covariance
    H : NDArray
        Observation noise covariance
    R : NDArray
        Noise selection matrix (defaults to identity)
        
    Examples
    --------
    >>> kf = KalmanFilter(state_dim=100, obs_dim=100, mode='auto')
    >>> kf.initialize(T=T_matrix, Z=Z_matrix, Q=Q_matrix, H=H_matrix)
    >>> result = kf.filter(observations)
    """
    
    # Thresholds for automatic mode selection
    DENSE_THRESHOLD = 1000
    DIAGONAL_THRESHOLD = 5000
    
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
        
        # Determine mode automatically if requested
        if mode == "auto":
            self.mode = self._select_mode(state_dim)
        else:
            self.mode = mode
            
        # Initialize matrices to None
        self.T: Optional[NDArray] = None
        self.Z: Optional[NDArray] = None
        self.Q: Optional[NDArray] = None
        self.H: Optional[NDArray] = None
        self.R: Optional[NDArray] = None
        
        # Initial state
        self.initial_mean: Optional[NDArray] = None
        self.initial_covariance: Optional[NDArray] = None
        
        self._initialized = False
        
    def _select_mode(self, n: int) -> str:
        """Select scalability mode based on problem size."""
        if n < self.DENSE_THRESHOLD:
            return "dense"
        elif n < self.DIAGONAL_THRESHOLD:
            return "diagonal"
        else:
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
        """Initialize filter with system matrices.
        
        Parameters
        ----------
        T : NDArray
            State transition matrix
        Z : NDArray
            Observation matrix
        Q : NDArray
            Process noise covariance
        H : NDArray
            Observation noise covariance
        R : NDArray, optional
            Noise selection matrix. Defaults to identity.
        initial_mean : NDArray, optional
            Initial state mean. Defaults to zeros.
        initial_covariance : NDArray, optional
            Initial state covariance. Defaults to identity.
        """
        self.T = self._convert_matrix(T)
        self.Z = self._convert_matrix(Z)
        self.Q = self._convert_matrix(Q)
        self.H = self._convert_matrix(H)
        self.R = self._convert_matrix(R) if R is not None else self._identity()
        
        # Initialize state
        if initial_mean is not None:
            self.initial_mean = np.asarray(initial_mean)
        else:
            self.initial_mean = np.zeros(self.state_dim)
            
        if initial_covariance is not None:
            self.initial_covariance = self._convert_matrix(initial_covariance)
        else:
            self.initial_covariance = self._identity()
            
        self._initialized = True
        
    def _convert_matrix(self, M: NDArray) -> NDArray:
        """Convert matrix to appropriate format based on mode."""
        if self.mode == "dense":
            return np.asarray(M)
        elif self.mode == "diagonal":
            # Store only diagonal
            if M.ndim == 2:
                return np.diag(M) if M.shape[0] == M.shape[1] else M
            return M
        else:  # block
            if sparse.issparse(M):
                return M.tocsr()
            return csr_matrix(M)
            
    def _identity(self) -> NDArray:
        """Create identity matrix in appropriate format."""
        if self.mode == "dense":
            return np.eye(self.state_dim)
        elif self.mode == "diagonal":
            return np.ones(self.state_dim)
        else:
            return sparse.eye(self.state_dim, format="csr")
            
    def _matrix_multiply(self, A: NDArray, B: NDArray) -> NDArray:
        """Matrix multiplication with mode-appropriate handling."""
        if self.mode == "diagonal":
            # Element-wise for diagonal matrices stored as vectors
            if A.ndim == 1 and B.ndim == 1:
                return A * B
            elif A.ndim == 1:
                return A[:, np.newaxis] * B
            elif B.ndim == 1:
                return A * B
            return A @ B
        return A @ B
        
    def _solve(self, A: NDArray, b: NDArray) -> NDArray:
        """Solve linear system Ax = b."""
        if self.mode == "diagonal":
            return b / (A + self.regularization)
        elif self.mode == "block":
            A_reg = A + self.regularization * sparse.eye(A.shape[0])
            return spsolve(A_reg, b)
        else:
            A_reg = A + self.regularization * np.eye(A.shape[0])
            return np.linalg.solve(A_reg, b)
            
    def _inverse(self, A: NDArray) -> NDArray:
        """Compute matrix inverse or pseudo-inverse."""
        if self.mode == "diagonal":
            return 1.0 / (A + self.regularization)
        else:
            A_reg = A + self.regularization * np.eye(A.shape[0])
            try:
                return np.linalg.inv(A_reg)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(A_reg)
                
    def _transpose(self, A: NDArray) -> NDArray:
        """Matrix transpose."""
        if self.mode == "diagonal":
            return A  # Diagonal is symmetric
        elif sparse.issparse(A):
            return A.T
        return A.T
        
    def _add_matrices(self, A: NDArray, B: NDArray) -> NDArray:
        """Add two matrices."""
        return A + B
        
    def _compute_log_likelihood(
        self, 
        innovation: NDArray, 
        innovation_covariance: NDArray
    ) -> float:
        """Compute log-likelihood contribution from innovation."""
        n = len(innovation)
        
        if self.mode == "diagonal":
            log_det = np.sum(np.log(innovation_covariance + self.regularization))
            quad_form = np.sum(innovation**2 / (innovation_covariance + self.regularization))
        else:
            # Use Cholesky for numerical stability
            try:
                L = np.linalg.cholesky(innovation_covariance + self.regularization * np.eye(n))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                v = np.linalg.solve(L, innovation)
                quad_form = np.dot(v, v)
            except np.linalg.LinAlgError:
                # Fallback to eigendecomposition
                eigvals = np.linalg.eigvalsh(innovation_covariance)
                eigvals = np.maximum(eigvals, self.regularization)
                log_det = np.sum(np.log(eigvals))
                quad_form = innovation @ self._inverse(innovation_covariance) @ innovation
                
        return -0.5 * (n * np.log(2 * np.pi) + log_det + quad_form)
        
    def _predict_step(
        self,
        filtered_mean: NDArray,
        filtered_covariance: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Kalman filter prediction step.
        
        Computes:
            α_{t|t-1} = T α_{t-1|t-1}
            P_{t|t-1} = T P_{t-1|t-1} T' + R Q R'
        """
        # Predicted mean
        predicted_mean = self._matrix_multiply(self.T, filtered_mean)
        
        # Predicted covariance
        if self.mode == "diagonal":
            # T P T' + R Q R' simplifies for diagonal matrices
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
        """Kalman filter update step.
        
        Computes:
            v_t = y_t - Z α_{t|t-1}              (innovation)
            F_t = Z P_{t|t-1} Z' + H             (innovation covariance)
            K_t = P_{t|t-1} Z' F_t^{-1}          (Kalman gain)
            α_{t|t} = α_{t|t-1} + K_t v_t        (filtered mean)
            P_{t|t} = (I - K_t Z) P_{t|t-1}      (filtered covariance)
        """
        # Innovation
        innovation = observation - self._matrix_multiply(self.Z, predicted_mean)
        
        # Diagonal mode stays fully element-wise
        if self.mode == "diagonal":
            innovation_cov = self.Z**2 * predicted_covariance + self.H
            kalman_gain = (predicted_covariance * self.Z) / (innovation_cov + self.regularization)
            filtered_mean = predicted_mean + self._matrix_multiply(kalman_gain, innovation)
            filtered_cov = (1 - kalman_gain * self.Z) * predicted_covariance
            ll = self._compute_log_likelihood(innovation, innovation_cov)
            
            return FilteredState(
                mean=filtered_mean,
                covariance=filtered_cov,
                predicted_mean=predicted_mean,
                predicted_covariance=predicted_covariance,
                kalman_gain=kalman_gain,
                log_likelihood=ll,
            )
        
        # Shared computations for dense/block modes
        Z_P = self._matrix_multiply(self.Z, predicted_covariance)
        Z_P_Z = self._matrix_multiply(Z_P, self._transpose(self.Z))
        innovation_cov = self._add_matrices(Z_P_Z, self.H)
        P_Z = self._matrix_multiply(predicted_covariance, self._transpose(self.Z))
        
        if self.mode == "dense":
            # Use Cholesky factorization once for both gain and likelihood
            innovation_cov_reg = innovation_cov + self.regularization * np.eye(self.obs_dim)
            try:
                chol_factor, lower = cho_factor(
                    innovation_cov_reg, lower=False, check_finite=False
                )
                # Solve F K' = P_Z' instead of forming F^{-1}
                kalman_gain = cho_solve((chol_factor, lower), P_Z.T, check_finite=False).T
                
                filtered_mean = predicted_mean + self._matrix_multiply(kalman_gain, innovation)
                I_KZ = np.eye(self.state_dim) - self._matrix_multiply(kalman_gain, self.Z)
                filtered_cov = self._matrix_multiply(I_KZ, predicted_covariance)
                filtered_cov = 0.5 * (filtered_cov + self._transpose(filtered_cov))
                
                # Log-likelihood using the same Cholesky factor
                log_det = 2.0 * np.sum(np.log(np.diag(chol_factor)))
                whitened_innov = solve_triangular(
                    chol_factor, innovation, lower=lower, check_finite=False
                )
                quad_form = np.dot(whitened_innov, whitened_innov)
                ll = -0.5 * (self.obs_dim * np.log(2 * np.pi) + log_det + quad_form)
            except np.linalg.LinAlgError:
                # Fallback to the more expensive inverse-based path
                F_inv = self._inverse(innovation_cov)
                kalman_gain = self._matrix_multiply(P_Z, F_inv)
                filtered_mean = predicted_mean + self._matrix_multiply(kalman_gain, innovation)
                I_KZ = np.eye(self.state_dim) - self._matrix_multiply(kalman_gain, self.Z)
                filtered_cov = self._matrix_multiply(I_KZ, predicted_covariance)
                filtered_cov = 0.5 * (filtered_cov + self._transpose(filtered_cov))
                ll = self._compute_log_likelihood(innovation, innovation_cov)
        else:
            # Block/sparse modes retain the existing solve strategy
            kalman_gain = self._matrix_multiply(P_Z, self._inverse(innovation_cov))
            filtered_mean = predicted_mean + self._matrix_multiply(kalman_gain, innovation)
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
        self,
        observations: NDArray,
        missing_mask: Optional[NDArray] = None,
    ) -> FilterResult:
        """Run Kalman filter on observation sequence.
        
        Parameters
        ----------
        observations : NDArray
            Observation matrix, shape (T, obs_dim)
        missing_mask : NDArray, optional
            Boolean mask indicating missing values, shape (T, obs_dim)
            
        Returns
        -------
        FilterResult
            Container with filtered states and log-likelihood
        """
        if not self._initialized:
            raise RuntimeError("Filter not initialized. Call initialize() first.")
            
        T_len = observations.shape[0]
        
        # Storage
        filtered_means = np.zeros((T_len, self.state_dim))
        predicted_means = np.zeros((T_len, self.state_dim))
        
        if self.mode == "diagonal":
            filtered_covs = np.zeros((T_len, self.state_dim))
            predicted_covs = np.zeros((T_len, self.state_dim))
        else:
            filtered_covs = np.zeros((T_len, self.state_dim, self.state_dim))
            predicted_covs = np.zeros((T_len, self.state_dim, self.state_dim))
            
        total_ll = 0.0
        
        # Initialize
        current_mean = self.initial_mean.copy()
        current_cov = self.initial_covariance.copy() if self.mode != "diagonal" else self.initial_covariance.copy()
        
        for t in range(T_len):
            # Prediction step
            pred_mean, pred_cov = self._predict_step(current_mean, current_cov)
            
            # Handle missing observations
            obs_t = observations[t]
            if missing_mask is not None and missing_mask[t].any():
                # Skip update for missing observations
                filtered_state = FilteredState(
                    mean=pred_mean,
                    covariance=pred_cov,
                    predicted_mean=pred_mean,
                    predicted_covariance=pred_cov,
                    kalman_gain=np.zeros_like(pred_mean),
                    log_likelihood=0.0,
                )
            else:
                # Update step
                filtered_state = self._update_step(obs_t, pred_mean, pred_cov)
                
            # Store results
            filtered_means[t] = filtered_state.mean
            filtered_covs[t] = filtered_state.covariance
            predicted_means[t] = filtered_state.predicted_mean
            predicted_covs[t] = filtered_state.predicted_covariance
            total_ll += filtered_state.log_likelihood
            
            # Update for next iteration
            current_mean = filtered_state.mean
            current_cov = filtered_state.covariance
            
        return FilterResult(
            filtered_means=filtered_means,
            filtered_covariances=filtered_covs,
            predicted_means=predicted_means,
            predicted_covariances=predicted_covs,
            log_likelihood=total_ll,
        )


class RTSSmoother:
    """Rauch-Tung-Striebel smoother for linear Gaussian state space models.
    
    Computes smoothed state estimates using both forward (filtered) and
    backward information. Required for EM parameter estimation.
    
    Parameters
    ----------
    kalman_filter : KalmanFilter
        Initialized Kalman filter with system matrices
        
    Examples
    --------
    >>> kf = KalmanFilter(state_dim=100, obs_dim=100)
    >>> kf.initialize(T=T, Z=Z, Q=Q, H=H)
    >>> filter_result = kf.filter(observations)
    >>> smoother = RTSSmoother(kf)
    >>> smooth_result = smoother.smooth(filter_result)
    """
    
    def __init__(self, kalman_filter: KalmanFilter):
        self.kf = kalman_filter
        self.mode = kalman_filter.mode
        
    def smooth(self, filter_result: FilterResult) -> SmootherResult:
        """Run RTS smoother on filtered results.
        
        Parameters
        ----------
        filter_result : FilterResult
            Output from KalmanFilter.filter()
            
        Returns
        -------
        SmootherResult
            Container with smoothed states and cross-covariances
        """
        T_len = filter_result.filtered_means.shape[0]
        state_dim = self.kf.state_dim
        
        # Initialize storage
        smoothed_means = np.zeros((T_len, state_dim))
        if self.mode == "diagonal":
            smoothed_covs = np.zeros((T_len, state_dim))
            cross_covs = np.zeros((T_len, state_dim))
        else:
            smoothed_covs = np.zeros((T_len, state_dim, state_dim))
            cross_covs = np.zeros((T_len, state_dim, state_dim))
            
        # Initialize at T
        smoothed_means[-1] = filter_result.filtered_means[-1]
        smoothed_covs[-1] = filter_result.filtered_covariances[-1]
        
        # Backward recursion
        for t in range(T_len - 2, -1, -1):
            # Smoother gain: J_t = P_{t|t} T' P_{t+1|t}^{-1}
            filtered_cov_t = filter_result.filtered_covariances[t]
            predicted_cov_tp1 = filter_result.predicted_covariances[t + 1]
            
            if self.mode == "diagonal":
                J_t = (filtered_cov_t * self.kf.T) / (predicted_cov_tp1 + self.kf.regularization)
            else:
                P_T = self.kf._matrix_multiply(filtered_cov_t, self.kf._transpose(self.kf.T))
                P_inv = self.kf._inverse(predicted_cov_tp1)
                J_t = self.kf._matrix_multiply(P_T, P_inv)
                
            # Smoothed mean: α_{t|T} = α_{t|t} + J_t (α_{t+1|T} - α_{t+1|t})
            mean_diff = smoothed_means[t + 1] - filter_result.predicted_means[t + 1]
            smoothed_means[t] = filter_result.filtered_means[t] + self.kf._matrix_multiply(J_t, mean_diff)
            
            # Smoothed covariance: P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t'
            if self.mode == "diagonal":
                cov_diff = smoothed_covs[t + 1] - predicted_cov_tp1
                smoothed_covs[t] = filtered_cov_t + J_t**2 * cov_diff
            else:
                cov_diff = smoothed_covs[t + 1] - predicted_cov_tp1
                J_cov = self.kf._matrix_multiply(J_t, cov_diff)
                J_cov_J = self.kf._matrix_multiply(J_cov, self.kf._transpose(J_t))
                smoothed_covs[t] = self.kf._add_matrices(filtered_cov_t, J_cov_J)
                
            # Cross-covariance for EM: E[α_t α_{t-1}' | y_{1:T}]
            # Cov(α_{t+1}, α_t | y_{1:T}) = J_t P_{t+1|T}
            if self.mode == "diagonal":
                cross_covs[t + 1] = J_t * smoothed_covs[t + 1]
            else:
                cross_covs[t + 1] = self.kf._matrix_multiply(J_t, smoothed_covs[t + 1])
                
        return SmootherResult(
            smoothed_means=smoothed_means,
            smoothed_covariances=smoothed_covs,
            cross_covariances=cross_covs,
        )
