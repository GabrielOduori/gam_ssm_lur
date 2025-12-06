"""
State Space Model for Temporal Dynamics in Air Pollution.

This module provides a high-level interface to the state space model
components (Kalman filter, RTS smoother, EM estimator) for modeling
temporal dynamics in air pollution data.

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
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from gam_ssm_lur.inference.kalman import KalmanFilter, RTSSmoother, FilterResult, SmootherResult
from gam_ssm_lur.inference.em import EMEstimator, EMResult


logger = logging.getLogger(__name__)


@dataclass
class SSMPrediction:
    """Container for state space model predictions.
    
    Attributes
    ----------
    mean : NDArray
        Predicted mean values, shape (T, n_locations)
    std : NDArray
        Prediction standard deviations
    lower : NDArray
        Lower bound of prediction interval
    upper : NDArray
        Upper bound of prediction interval
    smoothed_states : NDArray
        Smoothed latent states
    filtered_states : NDArray
        Filtered latent states
    """
    mean: NDArray
    std: NDArray
    lower: NDArray
    upper: NDArray
    smoothed_states: NDArray
    filtered_states: NDArray


@dataclass
class SSMDiagnostics:
    """Diagnostics for state space model.
    
    Attributes
    ----------
    log_likelihood : float
        Model log-likelihood
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    em_converged : bool
        Whether EM algorithm converged
    em_iterations : int
        Number of EM iterations
    transition_eigenvalues : NDArray
        Eigenvalues of transition matrix (stability check)
    process_noise_variance : float
        Trace of Q matrix
    observation_noise_variance : float
        Trace of H matrix
    """
    log_likelihood: float
    aic: float
    bic: float
    em_converged: bool
    em_iterations: int
    transition_eigenvalues: NDArray
    process_noise_variance: float
    observation_noise_variance: float


class StateSpaceModel:
    """State Space Model for temporal dynamics in spatiotemporal pollution data.
    
    Implements a linear Gaussian state space model:
    
        Measurement: y_t = Z α_t + ε_t,  ε_t ~ N(0, H)
        Transition:  α_{t+1} = T α_t + R η_t,  η_t ~ N(0, Q)
    
    Parameters are estimated via Expectation-Maximisation, and inference
    is performed using Kalman filtering (forward) and RTS smoothing (backward).
    
    Parameters
    ----------
    state_dim : int, optional
        Dimension of latent state. Defaults to observation dimension.
    em_max_iter : int
        Maximum EM iterations for parameter estimation
    em_tol : float
        Convergence tolerance for EM
    scalability_mode : {'auto', 'dense', 'diagonal', 'block'}
        Computational mode for Kalman filter
    regularization : float
        Regularization constant for numerical stability
    estimate_T : bool
        Whether to estimate transition matrix
    estimate_Q : bool
        Whether to estimate process noise covariance
    estimate_H : bool
        Whether to estimate observation noise covariance
    diagonal_covariances : bool
        Constrain Q and H to be diagonal
    random_state : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    T_ : NDArray
        Estimated transition matrix
    Z_ : NDArray
        Observation matrix
    Q_ : NDArray
        Estimated process noise covariance
    H_ : NDArray
        Estimated observation noise covariance
    initial_mean_ : NDArray
        Estimated initial state mean
    initial_cov_ : NDArray
        Estimated initial state covariance
    is_fitted_ : bool
        Whether the model has been fitted
        
    Examples
    --------
    >>> ssm = StateSpaceModel(em_max_iter=50, scalability_mode='auto')
    >>> ssm.fit(residuals)  # residuals from GAM
    >>> predictions = ssm.predict(return_intervals=True)
    """
    
    def __init__(
        self,
        state_dim: Optional[int] = None,
        em_max_iter: int = 50,
        em_tol: float = 1e-6,
        scalability_mode: Literal["auto", "dense", "diagonal", "block"] = "auto",
        regularization: float = 1e-6,
        estimate_T: bool = True,
        estimate_Q: bool = True,
        estimate_H: bool = True,
        diagonal_covariances: bool = False,
        random_state: Optional[int] = None,
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
        self.random_state = random_state
        
        # Fitted attributes
        self.T_: Optional[NDArray] = None
        self.Z_: Optional[NDArray] = None
        self.Q_: Optional[NDArray] = None
        self.H_: Optional[NDArray] = None
        self.initial_mean_: Optional[NDArray] = None
        self.initial_cov_: Optional[NDArray] = None
        
        self.em_result_: Optional[EMResult] = None
        self.kf_: Optional[KalmanFilter] = None
        self.filter_result_: Optional[FilterResult] = None
        self.smoother_result_: Optional[SmootherResult] = None
        
        self.is_fitted_ = False
        self._obs_dim: Optional[int] = None
        self._T_len: Optional[int] = None
        
    def fit(
        self,
        observations: Union[NDArray, pd.DataFrame],
        T_init: Optional[NDArray] = None,
        Q_init: Optional[NDArray] = None,
        H_init: Optional[NDArray] = None,
    ) -> "StateSpaceModel":
        """Fit state space model parameters using EM algorithm.
        
        Parameters
        ----------
        observations : array-like of shape (T, n_locations)
            Observation matrix (typically residuals from spatial model)
        T_init : NDArray, optional
            Initial transition matrix
        Q_init : NDArray, optional
            Initial process noise covariance
        H_init : NDArray, optional
            Initial observation noise covariance
            
        Returns
        -------
        self : StateSpaceModel
            Fitted model
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Convert to numpy
        if isinstance(observations, pd.DataFrame):
            observations = observations.values
        observations = np.asarray(observations)
        
        self._T_len, self._obs_dim = observations.shape
        state_dim = self.state_dim or self._obs_dim
        
        logger.info(
            f"Fitting SSM: T={self._T_len}, obs_dim={self._obs_dim}, "
            f"state_dim={state_dim}, mode={self.scalability_mode}"
        )
        
        # Create EM estimator
        em = EMEstimator(
            max_iterations=self.em_max_iter,
            tolerance=self.em_tol,
            regularization=self.regularization,
            estimate_T=self.estimate_T,
            estimate_Q=self.estimate_Q,
            estimate_H=self.estimate_H,
            diagonal_Q=self.diagonal_covariances,
            diagonal_H=self.diagonal_covariances,
            verbose=True,
        )
        
        # Fit using EM
        self.em_result_ = em.fit(
            observations=observations,
            state_dim=state_dim,
            obs_dim=self._obs_dim,
            T_init=T_init,
            Q_init=Q_init,
            H_init=H_init,
            scalability_mode=self.scalability_mode,
        )
        
        # Store estimated parameters
        self.T_ = self.em_result_.T
        self.Z_ = np.eye(self._obs_dim, state_dim)  # Identity observation matrix
        self.Q_ = self.em_result_.Q
        self.H_ = self.em_result_.H
        self.initial_mean_ = self.em_result_.initial_mean
        self.initial_cov_ = self.em_result_.initial_covariance
        
        # Create Kalman filter with estimated parameters
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
        
        # Run final filter and smoother
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
        
    def predict(
        self,
        confidence_level: float = 0.95,
    ) -> SSMPrediction:
        """Get smoothed predictions with uncertainty.
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for prediction intervals
            
        Returns
        -------
        SSMPrediction
            Container with predictions and intervals
        """
        self._check_fitted()
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Smoothed means are the predictions
        smoothed_means = self.smoother_result_.smoothed_means
        smoothed_covs = self.smoother_result_.smoothed_covariances
        
        # Compute standard deviations
        if smoothed_covs.ndim == 2:
            # Diagonal mode: covariances stored as vectors
            std = np.sqrt(smoothed_covs)
        else:
            # Full covariance mode
            std = np.sqrt(np.diagonal(smoothed_covs, axis1=1, axis2=2))
            
        # Prediction intervals
        lower = smoothed_means - z_score * std
        upper = smoothed_means + z_score * std
        
        return SSMPrediction(
            mean=smoothed_means,
            std=std,
            lower=lower,
            upper=upper,
            smoothed_states=smoothed_means,
            filtered_states=self.filter_result_.filtered_means,
        )
        
    def forecast(
        self,
        n_steps: int,
        confidence_level: float = 0.95,
    ) -> SSMPrediction:
        """Forecast future states.
        
        Parameters
        ----------
        n_steps : int
            Number of steps to forecast
        confidence_level : float
            Confidence level for prediction intervals
            
        Returns
        -------
        SSMPrediction
            Forecasted values with uncertainty
        """
        self._check_fitted()
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Start from last filtered state
        current_mean = self.filter_result_.filtered_means[-1]
        current_cov = self.filter_result_.filtered_covariances[-1]
        
        if current_cov.ndim == 1:
            current_cov = np.diag(current_cov)
            
        # Forecast storage
        forecast_means = np.zeros((n_steps, self._obs_dim))
        forecast_covs = np.zeros((n_steps, self._obs_dim, self._obs_dim))
        
        for t in range(n_steps):
            # Predict step
            current_mean = self.T_ @ current_mean
            current_cov = self.T_ @ current_cov @ self.T_.T + self.Q_
            
            forecast_means[t] = self.Z_ @ current_mean
            forecast_covs[t] = self.Z_ @ current_cov @ self.Z_.T + self.H_
            
        # Compute intervals
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
        """Get model diagnostics.
        
        Returns
        -------
        SSMDiagnostics
            Container with diagnostic statistics
        """
        self._check_fitted()
        
        ll = self.em_result_.log_likelihoods[-1]
        
        # Count parameters
        state_dim = self.T_.shape[0]
        n_params = state_dim**2  # T
        if not self.diagonal_covariances:
            n_params += state_dim * (state_dim + 1) // 2  # Q (symmetric)
            n_params += self._obs_dim * (self._obs_dim + 1) // 2  # H (symmetric)
        else:
            n_params += state_dim + self._obs_dim  # Diagonal Q and H
            
        n_obs = self._T_len * self._obs_dim
        
        # Information criteria
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + np.log(n_obs) * n_params
        
        # Transition matrix eigenvalues (stability check)
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
        )
        
    def get_em_history(self) -> pd.DataFrame:
        """Get EM algorithm convergence history.
        
        Returns
        -------
        pd.DataFrame
            Log-likelihood and parameter traces
        """
        self._check_fitted()
        
        return pd.DataFrame({
            'iteration': range(len(self.em_result_.log_likelihoods)),
            'log_likelihood': self.em_result_.log_likelihoods,
            'tr_T': [np.trace(self.T_)] * len(self.em_result_.log_likelihoods),
            'tr_Q': [np.trace(self.Q_)] * len(self.em_result_.log_likelihoods),
            'tr_H': [np.trace(self.H_)] * len(self.em_result_.log_likelihoods),
        })
        
    def get_innovation_diagnostics(
        self,
        observations: NDArray,
    ) -> Dict[str, NDArray]:
        """Compute innovation (prediction error) diagnostics.
        
        Parameters
        ----------
        observations : NDArray
            Original observations
            
        Returns
        -------
        Dict with innovation statistics
        """
        self._check_fitted()
        
        # Innovations = y_t - E[y_t | y_{1:t-1}]
        innovations = observations - self.filter_result_.predicted_means @ self.Z_.T
        
        # Standardized innovations
        pred_covs = self.filter_result_.predicted_covariances
        if pred_covs.ndim == 2:
            std_innov = innovations / np.sqrt(pred_covs + self.regularization)
        else:
            obs_cov = self.Z_ @ pred_covs @ self.Z_.T + self.H_
            std_diag = np.sqrt(np.diagonal(obs_cov, axis1=1, axis2=2))
            std_innov = innovations / std_diag
            
        # Autocorrelation of innovations
        def acf(x, nlags=20):
            n = len(x)
            mean = np.mean(x)
            var = np.var(x)
            acf_vals = np.zeros(nlags)
            for lag in range(nlags):
                acf_vals[lag] = np.mean((x[:n-lag] - mean) * (x[lag:] - mean)) / var
            return acf_vals
            
        # Compute ACF for each location
        acf_matrix = np.array([acf(innovations[:, i]) for i in range(innovations.shape[1])])
        
        return {
            'innovations': innovations,
            'standardized_innovations': std_innov,
            'mean_innovation': np.mean(innovations, axis=0),
            'std_innovation': np.std(innovations, axis=0),
            'acf': acf_matrix,
        }
        
    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
    def _restore_inference(self, observations: NDArray) -> None:
        """Recompute filter/smoother results for a loaded model."""
        obs = np.asarray(observations)
        if obs.ndim != 2:
            raise ValueError("observations must be 2D (T, obs_dim)")
        
        # Update cached dimensions if missing
        if self._obs_dim is None:
            self._obs_dim = obs.shape[1]
        if self._T_len is None:
            self._T_len = obs.shape[0]
        
        missing_mask = np.isnan(obs)
        obs_clean = np.where(missing_mask, 0.0, obs) if missing_mask.any() else obs
        missing_mask = missing_mask if missing_mask.any() else None
        
        kf = KalmanFilter(
            state_dim=self.T_.shape[0],
            obs_dim=self._obs_dim,
            mode=self.scalability_mode,
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
        """Save model to disk."""
        import pickle
        
        self._check_fitted()
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'params': {
                    'state_dim': self.state_dim,
                    'em_max_iter': self.em_max_iter,
                    'em_tol': self.em_tol,
                    'scalability_mode': self.scalability_mode,
                    'regularization': self.regularization,
                    'estimate_T': self.estimate_T,
                    'estimate_Q': self.estimate_Q,
                    'estimate_H': self.estimate_H,
                    'diagonal_covariances': self.diagonal_covariances,
                },
                'fitted': {
                    'T': self.T_,
                    'Z': self.Z_,
                    'Q': self.Q_,
                    'H': self.H_,
                    'initial_mean': self.initial_mean_,
                    'initial_cov': self.initial_cov_,
                    'obs_dim': self._obs_dim,
                    'T_len': self._T_len,
                },
                'em_result': {
                    'log_likelihoods': self.em_result_.log_likelihoods,
                    'n_iterations': self.em_result_.n_iterations,
                    'converged': self.em_result_.converged,
                },
            }, f)
            
        logger.info(f"SSM saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> "StateSpaceModel":
        """Load model from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        model = cls(**data['params'])
        model.T_ = data['fitted']['T']
        model.Z_ = data['fitted']['Z']
        model.Q_ = data['fitted']['Q']
        model.H_ = data['fitted']['H']
        model.initial_mean_ = data['fitted']['initial_mean']
        model.initial_cov_ = data['fitted']['initial_cov']
        model._obs_dim = data['fitted']['obs_dim']
        model._T_len = data['fitted']['T_len']
        model.is_fitted_ = True
        
        logger.info(f"SSM loaded from {filepath}")
        return model
