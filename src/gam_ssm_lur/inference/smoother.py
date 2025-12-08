"""
Rauch-Tung-Striebel (RTS) smoother built on top of the Kalman filter.

The smoother reuses the same matrix operations and system matrices provided by
an initialized :class:`~gam_ssm_lur.inference.kalman.KalmanFilter`, so the
forward/backward passes stay consistent across scalability modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from gam_ssm_lur.inference.kalman import FilterResult, KalmanFilter


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
