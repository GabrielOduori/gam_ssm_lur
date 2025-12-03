"""
Expectation-Maximisation (EM) Algorithm for State Space Model Parameter Estimation.

This module implements the EM algorithm for maximum likelihood estimation of
linear Gaussian state space model parameters, following Shumway and Stoffer (1982).

References
----------
.. [1] Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
       from incomplete data via the EM algorithm. Journal of the Royal Statistical
       Society: Series B, 39(1), 1-22.
.. [2] Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series smoothing
       and forecasting using the EM algorithm. Journal of Time Series Analysis, 3(4),
       253-264.
.. [3] Ghahramani, Z., & Hinton, G. E. (1996). Parameter estimation for linear
       dynamical systems. Technical Report CRG-TR-96-2, University of Toronto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from gam_ssm_lur.kalman import KalmanFilter, RTSSmoother, FilterResult, SmootherResult


logger = logging.getLogger(__name__)


@dataclass
class EMResult:
    """Container for EM algorithm results.
    
    Attributes
    ----------
    T : NDArray
        Estimated state transition matrix
    Q : NDArray
        Estimated process noise covariance
    H : NDArray
        Estimated observation noise covariance
    initial_mean : NDArray
        Estimated initial state mean
    initial_covariance : NDArray
        Estimated initial state covariance
    log_likelihoods : List[float]
        Log-likelihood values at each iteration
    n_iterations : int
        Number of iterations until convergence
    converged : bool
        Whether the algorithm converged
    smoothed_states : NDArray
        Final smoothed state estimates
    smoothed_covariances : NDArray
        Final smoothed state covariances
    """
    T: NDArray
    Q: NDArray
    H: NDArray
    initial_mean: NDArray
    initial_covariance: NDArray
    log_likelihoods: List[float]
    n_iterations: int
    converged: bool
    smoothed_states: NDArray
    smoothed_covariances: NDArray


@dataclass
class EMDiagnostics:
    """Diagnostics for monitoring EM convergence.
    
    Attributes
    ----------
    iteration : int
        Current iteration number
    log_likelihood : float
        Current log-likelihood
    ll_change : float
        Change in log-likelihood from previous iteration
    param_changes : Dict[str, float]
        Frobenius norm of parameter changes
    """
    iteration: int
    log_likelihood: float
    ll_change: float
    param_changes: Dict[str, float] = field(default_factory=dict)


class EMEstimator:
    """Expectation-Maximisation estimator for state space model parameters.
    
    Estimates the parameters {T, Q, H, α₀, P₀} of a linear Gaussian state
    space model using the EM algorithm. The E-step computes expected sufficient
    statistics using Kalman filtering and RTS smoothing; the M-step updates
    parameters to maximize the expected complete-data log-likelihood.
    
    Parameters
    ----------
    max_iterations : int
        Maximum number of EM iterations
    tolerance : float
        Convergence tolerance for log-likelihood change
    min_iterations : int
        Minimum iterations before checking convergence
    regularization : float
        Regularization constant for numerical stability
    estimate_T : bool
        Whether to estimate the transition matrix T
    estimate_Q : bool
        Whether to estimate the process noise covariance Q
    estimate_H : bool
        Whether to estimate the observation noise covariance H
    estimate_initial : bool
        Whether to estimate initial state parameters
    diagonal_Q : bool
        Constrain Q to be diagonal
    diagonal_H : bool
        Constrain H to be diagonal
    verbose : bool
        Print convergence information
        
    Examples
    --------
    >>> em = EMEstimator(max_iterations=100, tolerance=1e-6)
    >>> result = em.fit(observations, state_dim=50, obs_dim=50)
    >>> print(f"Converged: {result.converged} in {result.n_iterations} iterations")
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        min_iterations: int = 5,
        regularization: float = 1e-6,
        estimate_T: bool = True,
        estimate_Q: bool = True,
        estimate_H: bool = True,
        estimate_initial: bool = True,
        diagonal_Q: bool = False,
        diagonal_H: bool = False,
        verbose: bool = True,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_iterations = min_iterations
        self.regularization = regularization
        self.estimate_T = estimate_T
        self.estimate_Q = estimate_Q
        self.estimate_H = estimate_H
        self.estimate_initial = estimate_initial
        self.diagonal_Q = diagonal_Q
        self.diagonal_H = diagonal_H
        self.verbose = verbose
        
        self.diagnostics_history: List[EMDiagnostics] = []
        
    def _initialize_parameters(
        self,
        observations: NDArray,
        state_dim: int,
        obs_dim: int,
        T_init: Optional[NDArray] = None,
        Q_init: Optional[NDArray] = None,
        H_init: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Initialize SSM parameters.
        
        Parameters are initialized to sensible defaults if not provided:
        - T: Identity matrix (random walk)
        - Z: Identity matrix (direct observation)
        - Q: Scaled identity based on observation variance
        - H: Scaled identity based on observation variance
        - α₀: Zero vector
        - P₀: Identity matrix
        """
        # Observation statistics for initialization
        obs_var = np.var(observations, axis=0).mean()
        
        # Transition matrix
        if T_init is not None:
            T = T_init.copy()
        else:
            # Slight shrinkage toward mean (AR(1) coefficient < 1)
            T = 0.95 * np.eye(state_dim)
            
        # Observation matrix (identity for now)
        Z = np.eye(obs_dim, state_dim)
        
        # Process noise covariance
        if Q_init is not None:
            Q = Q_init.copy()
        else:
            Q = 0.1 * obs_var * np.eye(state_dim)
            
        # Observation noise covariance
        if H_init is not None:
            H = H_init.copy()
        else:
            H = 0.5 * obs_var * np.eye(obs_dim)
            
        # Initial state
        initial_mean = np.zeros(state_dim)
        initial_cov = np.eye(state_dim)
        
        return T, Z, Q, H, initial_mean, initial_cov
        
    def _e_step(
        self,
        observations: NDArray,
        kf: KalmanFilter,
    ) -> Tuple[FilterResult, SmootherResult]:
        """E-step: Compute expected sufficient statistics.
        
        Runs Kalman filter (forward pass) and RTS smoother (backward pass)
        to compute the posterior distribution of latent states.
        """
        # Forward pass
        filter_result = kf.filter(observations)
        
        # Backward pass
        smoother = RTSSmoother(kf)
        smoother_result = smoother.smooth(filter_result)
        
        return filter_result, smoother_result
        
    def _compute_sufficient_statistics(
        self,
        observations: NDArray,
        smoother_result: SmootherResult,
    ) -> Dict[str, NDArray]:
        """Compute sufficient statistics for M-step.
        
        Returns
        -------
        Dict containing:
            - 'S11': Σ E[αₜ αₜ']
            - 'S10': Σ E[αₜ αₜ₋₁']
            - 'S00': Σ E[αₜ₋₁ αₜ₋₁']
            - 'Sy1': Σ yₜ E[αₜ']
            - 'Syy': Σ yₜ yₜ'
        """
        T_len = observations.shape[0]
        state_dim = smoother_result.smoothed_means.shape[1]
        obs_dim = observations.shape[1]
        
        # Initialize accumulators
        S11 = np.zeros((state_dim, state_dim))
        S10 = np.zeros((state_dim, state_dim))
        S00 = np.zeros((state_dim, state_dim))
        Sy1 = np.zeros((obs_dim, state_dim))
        Syy = np.zeros((obs_dim, obs_dim))
        
        for t in range(T_len):
            alpha_t = smoother_result.smoothed_means[t]
            P_t = smoother_result.smoothed_covariances[t]
            y_t = observations[t]
            
            # Handle diagonal mode
            if P_t.ndim == 1:
                P_t_full = np.diag(P_t)
            else:
                P_t_full = P_t
                
            # E[αₜ αₜ'] = P_{t|T} + α_{t|T} α_{t|T}'
            E_alpha_alpha = P_t_full + np.outer(alpha_t, alpha_t)
            S11 += E_alpha_alpha
            
            # E[yₜ αₜ'] = yₜ α_{t|T}'
            Sy1 += np.outer(y_t, alpha_t)
            
            # Σ yₜ yₜ'
            Syy += np.outer(y_t, y_t)
            
            if t > 0:
                alpha_tm1 = smoother_result.smoothed_means[t - 1]
                P_tm1 = smoother_result.smoothed_covariances[t - 1]
                cross_cov = smoother_result.cross_covariances[t]
                
                if P_tm1.ndim == 1:
                    P_tm1_full = np.diag(P_tm1)
                else:
                    P_tm1_full = P_tm1
                    
                if cross_cov.ndim == 1:
                    cross_cov_full = np.diag(cross_cov)
                else:
                    cross_cov_full = cross_cov
                    
                # E[αₜ αₜ₋₁'] = Cov(αₜ, αₜ₋₁|y_{1:T}) + α_{t|T} α_{t-1|T}'
                E_alpha_alpha_lag = cross_cov_full + np.outer(alpha_t, alpha_tm1)
                S10 += E_alpha_alpha_lag
                
                # E[αₜ₋₁ αₜ₋₁']
                E_alpha_alpha_prev = P_tm1_full + np.outer(alpha_tm1, alpha_tm1)
                S00 += E_alpha_alpha_prev
                
        return {
            'S11': S11,
            'S10': S10,
            'S00': S00,
            'Sy1': Sy1,
            'Syy': Syy,
            'T_len': T_len,
        }
        
    def _m_step(
        self,
        observations: NDArray,
        smoother_result: SmootherResult,
        current_T: NDArray,
        current_Q: NDArray,
        current_H: NDArray,
        current_Z: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """M-step: Update parameters to maximize expected log-likelihood.
        
        Updates:
            T = S10 @ S00^{-1}
            Q = (1/(T-1)) * (S11 - T @ S10')
            H = (1/T) * (Syy - 2 * Z @ Sy1' + Z @ S11 @ Z')
        """
        T_len = observations.shape[0]
        state_dim = current_T.shape[0]
        
        # Compute sufficient statistics
        stats = self._compute_sufficient_statistics(observations, smoother_result)
        
        # Update T
        if self.estimate_T:
            S00_reg = stats['S00'] + self.regularization * np.eye(state_dim)
            try:
                new_T = stats['S10'] @ np.linalg.inv(S00_reg)
            except np.linalg.LinAlgError:
                new_T = stats['S10'] @ np.linalg.pinv(S00_reg)
        else:
            new_T = current_T
            
        # Update Q
        if self.estimate_Q:
            new_Q = (stats['S11'] - new_T @ stats['S10'].T) / (T_len - 1)
            # Ensure positive definiteness
            new_Q = 0.5 * (new_Q + new_Q.T)  # Symmetrize
            eigvals, eigvecs = np.linalg.eigh(new_Q)
            eigvals = np.maximum(eigvals, self.regularization)
            new_Q = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            if self.diagonal_Q:
                new_Q = np.diag(np.diag(new_Q))
        else:
            new_Q = current_Q
            
        # Update H
        if self.estimate_H:
            Z_S11_Z = current_Z @ stats['S11'] @ current_Z.T
            Z_Sy1 = current_Z @ stats['Sy1'].T
            new_H = (stats['Syy'] - 2 * Z_Sy1 + Z_S11_Z) / T_len
            # Ensure positive definiteness
            new_H = 0.5 * (new_H + new_H.T)
            eigvals, eigvecs = np.linalg.eigh(new_H)
            eigvals = np.maximum(eigvals, self.regularization)
            new_H = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            if self.diagonal_H:
                new_H = np.diag(np.diag(new_H))
        else:
            new_H = current_H
            
        # Update initial state parameters
        if self.estimate_initial:
            new_initial_mean = smoother_result.smoothed_means[0]
            P0 = smoother_result.smoothed_covariances[0]
            if P0.ndim == 1:
                new_initial_cov = np.diag(P0)
            else:
                new_initial_cov = P0
        else:
            new_initial_mean = np.zeros(state_dim)
            new_initial_cov = np.eye(state_dim)
            
        return new_T, new_Q, new_H, new_initial_mean, new_initial_cov
        
    def _check_convergence(
        self,
        log_likelihoods: List[float],
        iteration: int,
    ) -> bool:
        """Check if EM has converged based on log-likelihood change."""
        if iteration < self.min_iterations:
            return False
            
        if len(log_likelihoods) < 2:
            return False
            
        ll_change = log_likelihoods[-1] - log_likelihoods[-2]
        
        # EM should monotonically increase log-likelihood
        # Small decreases can occur due to numerical issues
        if ll_change < -1e-4:
            logger.warning(f"Log-likelihood decreased at iteration {iteration}: {ll_change:.6e}")
            
        return abs(ll_change) < self.tolerance
        
    def _compute_param_changes(
        self,
        T_old: NDArray,
        T_new: NDArray,
        Q_old: NDArray,
        Q_new: NDArray,
        H_old: NDArray,
        H_new: NDArray,
    ) -> Dict[str, float]:
        """Compute Frobenius norm of parameter changes."""
        return {
            'T': np.linalg.norm(T_new - T_old, 'fro'),
            'Q': np.linalg.norm(Q_new - Q_old, 'fro'),
            'H': np.linalg.norm(H_new - H_old, 'fro'),
        }
        
    def fit(
        self,
        observations: NDArray,
        state_dim: Optional[int] = None,
        obs_dim: Optional[int] = None,
        T_init: Optional[NDArray] = None,
        Q_init: Optional[NDArray] = None,
        H_init: Optional[NDArray] = None,
        scalability_mode: Literal["auto", "dense", "diagonal", "block"] = "auto",
    ) -> EMResult:
        """Fit state space model parameters using EM algorithm.
        
        Parameters
        ----------
        observations : NDArray
            Observation matrix, shape (T, obs_dim)
        state_dim : int, optional
            State dimension. Defaults to obs_dim.
        obs_dim : int, optional
            Observation dimension. Inferred from data if not provided.
        T_init : NDArray, optional
            Initial transition matrix
        Q_init : NDArray, optional
            Initial process noise covariance
        H_init : NDArray, optional
            Initial observation noise covariance
        scalability_mode : str
            Mode for Kalman filter computation
            
        Returns
        -------
        EMResult
            Container with estimated parameters and diagnostics
        """
        # Infer dimensions
        T_len, obs_dim_data = observations.shape
        obs_dim = obs_dim or obs_dim_data
        state_dim = state_dim or obs_dim
        
        logger.info(f"Starting EM with T={T_len}, state_dim={state_dim}, obs_dim={obs_dim}")
        
        # Initialize parameters
        T, Z, Q, H, initial_mean, initial_cov = self._initialize_parameters(
            observations, state_dim, obs_dim, T_init, Q_init, H_init
        )
        
        # Create Kalman filter
        kf = KalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            mode=scalability_mode,
            regularization=self.regularization,
        )
        
        log_likelihoods: List[float] = []
        converged = False
        
        for iteration in range(self.max_iterations):
            # Initialize filter with current parameters
            kf.initialize(
                T=T, Z=Z, Q=Q, H=H,
                initial_mean=initial_mean,
                initial_covariance=initial_cov,
            )
            
            # E-step
            filter_result, smoother_result = self._e_step(observations, kf)
            log_likelihoods.append(filter_result.log_likelihood)
            
            # Store old parameters for convergence check
            T_old, Q_old, H_old = T.copy(), Q.copy(), H.copy()
            
            # M-step
            T, Q, H, initial_mean, initial_cov = self._m_step(
                observations, smoother_result, T, Q, H, Z
            )
            
            # Compute diagnostics
            param_changes = self._compute_param_changes(T_old, T, Q_old, Q, H_old, H)
            ll_change = log_likelihoods[-1] - log_likelihoods[-2] if len(log_likelihoods) > 1 else float('inf')
            
            diagnostics = EMDiagnostics(
                iteration=iteration,
                log_likelihood=log_likelihoods[-1],
                ll_change=ll_change,
                param_changes=param_changes,
            )
            self.diagnostics_history.append(diagnostics)
            
            if self.verbose:
                logger.info(
                    f"Iteration {iteration + 1}: "
                    f"LL={log_likelihoods[-1]:.4e}, "
                    f"ΔLL={ll_change:.4e}, "
                    f"||ΔT||={param_changes['T']:.4e}"
                )
                
            # Check convergence
            if self._check_convergence(log_likelihoods, iteration):
                converged = True
                if self.verbose:
                    logger.info(f"Converged after {iteration + 1} iterations")
                break
                
        if not converged and self.verbose:
            logger.warning(f"EM did not converge after {self.max_iterations} iterations")
            
        return EMResult(
            T=T,
            Q=Q,
            H=H,
            initial_mean=initial_mean,
            initial_covariance=initial_cov,
            log_likelihoods=log_likelihoods,
            n_iterations=len(log_likelihoods),
            converged=converged,
            smoothed_states=smoother_result.smoothed_means,
            smoothed_covariances=smoother_result.smoothed_covariances,
        )
        
    def get_diagnostics_dataframe(self):
        """Return diagnostics history as a pandas DataFrame."""
        import pandas as pd
        
        records = []
        for d in self.diagnostics_history:
            record = {
                'iteration': d.iteration,
                'log_likelihood': d.log_likelihood,
                'll_change': d.ll_change,
            }
            record.update({f'delta_{k}': v for k, v in d.param_changes.items()})
            records.append(record)
            
        return pd.DataFrame(records)
