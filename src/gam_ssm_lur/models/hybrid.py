"""
Hybrid GAM-SSM Model for Spatiotemporal Air Pollution Prediction.

This module provides the main interface for the hybrid Generalized Additive
Model–State Space Model framework, integrating spatial LUR modeling with
temporal dynamics via Kalman filtering.

The model decomposes observations as:
    y(s,t) = μ(s) + α(t) + ε(s,t)

where:
    - μ(s) is the spatial mean from GAM-LUR
    - α(t) is the latent temporal state from SSM
    - ε(s,t) is observation noise

References
----------
.. [1] Cressie, N., & Wikle, C. K. (2011). Statistics for spatio-temporal data.
       John Wiley & Sons.
.. [2] Wikle, C. K., Zammit-Mangion, A., & Cressie, N. (2019). Spatio-temporal
       statistics with R. Chapman and Hall/CRC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from gam_ssm_lur.models.spatial_gam import SpatialGAM, GAMSummary
from gam_ssm_lur.models.state_space import StateSpaceModel, SSMPrediction, SSMDiagnostics


logger = logging.getLogger(__name__)


@dataclass
class HybridPrediction:
    """Container for hybrid model predictions.
    
    Attributes
    ----------
    total : NDArray
        Total predictions (spatial + temporal), shape (T, n_locations)
    spatial : NDArray
        Spatial component from GAM
    temporal : NDArray
        Temporal component from SSM (smoothed states)
    std : NDArray
        Prediction standard deviations
    lower : NDArray
        Lower prediction interval bound
    upper : NDArray
        Upper prediction interval bound
    """
    total: NDArray
    spatial: NDArray
    temporal: NDArray
    std: NDArray
    lower: NDArray
    upper: NDArray


@dataclass
class HybridSummary:
    """Summary of hybrid model fit.
    
    Attributes
    ----------
    gam_summary : GAMSummary
        Summary of GAM component
    ssm_diagnostics : SSMDiagnostics
        Diagnostics of SSM component
    total_rmse : float
        Root mean square error of full model
    total_mae : float
        Mean absolute error
    total_r2 : float
        Coefficient of determination
    coverage_95 : float
        Empirical coverage of 95% prediction intervals
    """
    gam_summary: GAMSummary
    ssm_diagnostics: SSMDiagnostics
    total_rmse: float
    total_mae: float
    total_r2: float
    coverage_95: float


class HybridGAMSSM:
    """Hybrid Generalized Additive Model–State Space Model.
    
    Integrates a GAM-based Land Use Regression model for capturing spatial
    heterogeneity with a State Space Model for modeling temporal dynamics
    and providing uncertainty quantification.
    
    Parameters
    ----------
    n_splines : int
        Number of spline basis functions for GAM
    gam_lam : float or 'auto'
        Smoothing parameter for GAM
    state_dim : int, optional
        Dimension of SSM latent state. Defaults to number of locations.
    em_max_iter : int
        Maximum EM iterations for SSM parameter estimation
    em_tol : float
        Convergence tolerance for EM algorithm
    scalability_mode : {'auto', 'dense', 'diagonal', 'block'}
        Computational mode for Kalman filter operations
    regularization : float
        Regularization constant for numerical stability
    confidence_level : float
        Confidence level for prediction intervals
    random_state : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    gam_ : SpatialGAM
        Fitted GAM component
    ssm_ : StateSpaceModel
        Fitted SSM component
    is_fitted_ : bool
        Whether the model has been fitted
        
    Examples
    --------
    >>> from gam_ssm_lur import HybridGAMSSM
    >>> 
    >>> # Prepare data
    >>> X_spatial = ...  # Spatial features (n_obs, n_features)
    >>> y = ...          # Observations (n_obs,)
    >>> time_idx = ...   # Time indices (n_obs,)
    >>> loc_idx = ...    # Location indices (n_obs,)
    >>> 
    >>> # Fit model
    >>> model = HybridGAMSSM(n_splines=10, em_max_iter=50)
    >>> model.fit(X_spatial, y, time_index=time_idx, location_index=loc_idx)
    >>> 
    >>> # Predict with uncertainty
    >>> predictions = model.predict(X_spatial_new, return_intervals=True)
    >>> print(f"RMSE: {model.evaluate(y_true, predictions.total)['rmse']:.3f}")
    """
    
    def __init__(
        self,
        n_splines: int = 10,
        gam_lam: Union[float, Literal["auto"]] = "auto",
        state_dim: Optional[int] = None,
        em_max_iter: int = 50,
        em_tol: float = 1e-6,
        scalability_mode: Literal["auto", "dense", "diagonal", "block"] = "auto",
        regularization: float = 1e-6,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ):
        self.n_splines = n_splines
        self.gam_lam = gam_lam
        self.state_dim = state_dim
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.scalability_mode = scalability_mode
        self.regularization = regularization
        self.confidence_level = confidence_level
        self.random_state = random_state
        
        # Components
        self.gam_: Optional[SpatialGAM] = None
        self.ssm_: Optional[StateSpaceModel] = None
        
        # Data info
        self.n_locations_: Optional[int] = None
        self.n_times_: Optional[int] = None
        self.location_ids_: Optional[NDArray] = None
        self.time_ids_: Optional[NDArray] = None
        
        # Training data (for residuals)
        self._X_train: Optional[NDArray] = None
        self._y_train: Optional[NDArray] = None
        self._y_matrix: Optional[NDArray] = None  # Reshaped as (T, n_locations)
        self._residual_matrix: Optional[NDArray] = None  # GAM residuals reshaped
        
        self.is_fitted_ = False
        
    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Union[NDArray, pd.Series],
        time_index: Union[NDArray, pd.Series],
        location_index: Optional[Union[NDArray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "HybridGAMSSM":
        """Fit the hybrid GAM-SSM model.
        
        Parameters
        ----------
        X : array-like of shape (n_obs, n_features)
            Spatial features for each observation
        y : array-like of shape (n_obs,)
            Target values (pollutant concentrations)
        time_index : array-like of shape (n_obs,)
            Time index for each observation (integer or datetime)
        location_index : array-like of shape (n_obs,), optional
            Location index for each observation. If not provided,
            assumes data is already in (T, n_locations) matrix form.
        feature_names : list of str, optional
            Names of features
            
        Returns
        -------
        self : HybridGAMSSM
            Fitted model
        """
        logger.info("Fitting Hybrid GAM-SSM model")
        
        # Convert inputs
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or list(X.columns)
            X = X.values
        X = np.asarray(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y)
        
        if isinstance(time_index, pd.Series):
            time_index = time_index.values
        time_index = np.asarray(time_index)
        
        # Store training data
        self._X_train = X
        self._y_train = y
        self._time_index_train = time_index.copy() if time_index is not None else None
        self._location_index_train = location_index.copy() if location_index is not None else None

        # Reshape to matrix form (T, n_locations)
        if location_index is not None:
            if isinstance(location_index, pd.Series):
                location_index = location_index.values
            location_index = np.asarray(location_index)

            self._y_matrix, self.location_ids_, self.time_ids_ = self._reshape_to_matrix(
                y, time_index, location_index
            )
        else:
            # Assume already in matrix form
            if y.ndim == 1:
                raise ValueError(
                    "If location_index is not provided, y must be 2D (T, n_locations)"
                )
            self._y_matrix = y
            self.time_ids_ = np.unique(time_index)
            self.location_ids_ = np.arange(y.shape[1])
            
        self.n_times_, self.n_locations_ = self._y_matrix.shape
        
        logger.info(f"Data shape: {self.n_times_} time steps × {self.n_locations_} locations")
        
        # Step 1: Fit GAM spatial component
        logger.info("Step 1: Fitting GAM spatial component")
        self.gam_ = SpatialGAM(
            n_splines=self.n_splines,
            lam=self.gam_lam,
        )
        self.gam_.fit(X, y, feature_names=feature_names)
        
        # Get GAM residuals
        gam_residuals = self.gam_.get_residuals(X, y)
        
        # Reshape residuals to matrix form
        if location_index is not None:
            residual_matrix, _, _ = self._reshape_to_matrix(
                gam_residuals, time_index, location_index
            )
        else:
            residual_matrix = gam_residuals.reshape(self.n_times_, self.n_locations_)
            
        logger.info(f"GAM fitted. R²={self.gam_.summary().r_squared:.4f}")
        logger.info(f"Residual matrix shape: {residual_matrix.shape}")
        self._residual_matrix = residual_matrix
        
        # Step 2: Fit SSM on residuals
        logger.info("Step 2: Fitting SSM on GAM residuals")
        self.ssm_ = StateSpaceModel(
            state_dim=self.state_dim,
            em_max_iter=self.em_max_iter,
            em_tol=self.em_tol,
            scalability_mode=self.scalability_mode,
            regularization=self.regularization,
            random_state=self.random_state,
        )
        self.ssm_.fit(residual_matrix)
        
        self.is_fitted_ = True
        
        # Log final performance
        predictions = self.predict()
        metrics = self.evaluate(self._y_matrix.flatten(), predictions.total.flatten())
        logger.info(
            f"Hybrid model fitted. "
            f"RMSE={metrics['rmse']:.4f}, "
            f"R²={metrics['r2']:.4f}, "
            f"Coverage={metrics['coverage_95']:.1%}"
        )
        
        return self
        
    def _reshape_to_matrix(
        self,
        values: NDArray,
        time_index: NDArray,
        location_index: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Reshape flat observations to (T, n_locations) matrix.
        
        Handles missing observations by filling with NaN.
        """
        unique_times = np.unique(time_index)
        unique_locs = np.unique(location_index)
        
        n_times = len(unique_times)
        n_locs = len(unique_locs)
        
        # Create time and location mappings
        time_map = {t: i for i, t in enumerate(unique_times)}
        loc_map = {l: i for i, l in enumerate(unique_locs)}
        
        # Initialize matrix with NaN
        matrix = np.full((n_times, n_locs), np.nan)
        
        # Fill in values
        for val, t, l in zip(values, time_index, location_index):
            matrix[time_map[t], loc_map[l]] = val
            
        # Handle missing values (simple interpolation for now)
        for j in range(n_locs):
            col = matrix[:, j]
            if np.any(np.isnan(col)):
                # Linear interpolation
                valid = ~np.isnan(col)
                if np.sum(valid) > 1:
                    matrix[:, j] = np.interp(
                        np.arange(n_times),
                        np.where(valid)[0],
                        col[valid]
                    )
                elif np.sum(valid) == 1:
                    matrix[:, j] = col[valid][0]
                else:
                    matrix[:, j] = 0.0
                    
        return matrix, unique_locs, unique_times
        
    def predict(
        self,
        X: Optional[Union[NDArray, pd.DataFrame]] = None,
    ) -> HybridPrediction:
        """Generate predictions with uncertainty quantification.

        Parameters
        ----------
        X : array-like, optional
            Spatial features for prediction. Uses training data if not provided.

        Returns
        -------
        HybridPrediction
            Container with predictions and intervals
        """
        self._check_fitted()

        if X is None:
            # Build full grid from training data
            if self._location_index_train is not None:
                # Get unique locations and their features
                unique_locs = np.unique(self._location_index_train)
                loc_to_features = {}
                for i, loc in enumerate(self._location_index_train):
                    if loc not in loc_to_features:
                        loc_to_features[loc] = self._X_train[i]

                # Create full grid: all time×location combinations
                X_grid = []
                for _ in range(self.n_times_):
                    for loc in self.location_ids_:
                        X_grid.append(loc_to_features[loc])
                X = np.array(X_grid)
            else:
                X = self._X_train

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)

        # Spatial predictions from GAM
        gam_pred, gam_std = self.gam_.predict(X, return_std=True)

        # Reshape to matrix form
        spatial_matrix = gam_pred.reshape(self.n_times_, self.n_locations_)
        
        # Temporal predictions from SSM
        ssm_pred = self.ssm_.predict(confidence_level=self.confidence_level)
        
        # Total prediction = spatial + temporal
        total = spatial_matrix + ssm_pred.mean
        
        # Combined uncertainty (assuming independence)
        gam_std_matrix = gam_std.reshape(self.n_times_, self.n_locations_)
        combined_std = np.sqrt(gam_std_matrix**2 + ssm_pred.std**2)
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        lower = total - z_score * combined_std
        upper = total + z_score * combined_std
        
        return HybridPrediction(
            total=total,
            spatial=spatial_matrix,
            temporal=ssm_pred.mean,
            std=combined_std,
            lower=lower,
            upper=upper,
        )

    def predict_in_observation_order(
        self,
        time_index: Union[NDArray, pd.Series],
        location_index: Union[NDArray, pd.Series],
        X: Optional[Union[NDArray, pd.DataFrame]] = None,
    ) -> NDArray:
        """Return predictions aligned to the original observation order.

        This remaps the internal time/location grid back to the provided
        observation ordering so metrics like R² are computed on correctly
        paired values.

        Parameters
        ----------
        time_index : array-like
            Time index per observation (same ordering as y provided to fit).
        location_index : array-like
            Location index per observation.
        X : array-like, optional
            Features for prediction in the same order as time/location. If
            None, uses the stored training features.

        Returns
        -------
        NDArray
            Predicted values aligned to the observation order.
        """
        self._check_fitted()

        if isinstance(time_index, pd.Series):
            time_index = time_index.values
        if isinstance(location_index, pd.Series):
            location_index = location_index.values

        time_index = np.asarray(time_index)
        location_index = np.asarray(location_index)

        if time_index.shape[0] != location_index.shape[0]:
            raise ValueError("time_index and location_index must have the same length")

        # Compute full grid predictions (uses training grid if X is None)
        full_pred = self.predict(X).total  # shape (T, n_locations)

        # Build lookup for grid positions
        time_map = {t: i for i, t in enumerate(self.time_ids_)}
        loc_map = {l: j for j, l in enumerate(self.location_ids_)}

        try:
            aligned = np.array([
                full_pred[time_map[t], loc_map[l]]
                for t, l in zip(time_index, location_index)
            ])
        except KeyError as exc:
            missing_key = exc.args[0]
            raise KeyError(
                f"Index '{missing_key}' not found in fitted model time/location IDs. "
                "Ensure the provided indices match those used in fit()."
            ) from exc

        return aligned
        
    def forecast(
        self,
        X_future: Union[NDArray, pd.DataFrame],
        n_steps: int,
    ) -> HybridPrediction:
        """Forecast future values.
        
        Parameters
        ----------
        X_future : array-like
            Spatial features for future locations/times
        n_steps : int
            Number of time steps to forecast
            
        Returns
        -------
        HybridPrediction
            Forecasted values with uncertainty
        """
        self._check_fitted()
        
        if isinstance(X_future, pd.DataFrame):
            X_future = X_future.values
        X_future = np.asarray(X_future)
        
        # Spatial prediction for future
        gam_pred, gam_std = self.gam_.predict(X_future, return_std=True)
        spatial_matrix = gam_pred.reshape(n_steps, self.n_locations_)
        gam_std_matrix = gam_std.reshape(n_steps, self.n_locations_)
        
        # Temporal forecast
        ssm_forecast = self.ssm_.forecast(n_steps, confidence_level=self.confidence_level)
        
        # Combine
        total = spatial_matrix + ssm_forecast.mean
        combined_std = np.sqrt(gam_std_matrix**2 + ssm_forecast.std**2)
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        lower = total - z_score * combined_std
        upper = total + z_score * combined_std
        
        return HybridPrediction(
            total=total,
            spatial=spatial_matrix,
            temporal=ssm_forecast.mean,
            std=combined_std,
            lower=lower,
            upper=upper,
        )
        
    def evaluate(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        y_lower: Optional[NDArray] = None,
        y_upper: Optional[NDArray] = None,
    ) -> Dict[str, float]:
        """Evaluate model performance.
        
        Parameters
        ----------
        y_true : NDArray
            True values
        y_pred : NDArray
            Predicted values
        y_lower : NDArray, optional
            Lower prediction interval
        y_upper : NDArray, optional
            Upper prediction interval
            
        Returns
        -------
        Dict with performance metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Basic metrics
        residuals = y_true - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        mbe = np.mean(residuals)  # Mean Bias Error (positive = under-prediction)

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot

        corr = np.corrcoef(y_true, y_pred)[0, 1]

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mbe': mbe,
            'r2': r2,
            'correlation': corr,
        }
        
        # Coverage and calibration metrics if intervals provided
        if y_lower is not None and y_upper is not None:
            from scipy import stats

            y_lower = np.asarray(y_lower).flatten()
            y_upper = np.asarray(y_upper).flatten()

            # 95% coverage
            coverage_95 = np.mean((y_true >= y_lower) & (y_true <= y_upper))
            interval_width = np.mean(y_upper - y_lower)
            metrics['coverage_95'] = coverage_95
            metrics['interval_width'] = interval_width

            # 90% coverage
            y_std = (y_upper - y_lower) / (2 * 1.96)  # Reconstruct std from 95% interval
            z_90 = stats.norm.ppf(0.95)  # 90% = 0.90, so (1 + 0.90) / 2 = 0.95
            lower_90 = y_pred - z_90 * y_std
            upper_90 = y_pred + z_90 * y_std
            coverage_90 = np.mean((y_true >= lower_90) & (y_true <= upper_90))
            metrics['coverage_90'] = coverage_90

            # CRPS (Continuous Ranked Probability Score)
            z = (y_true - y_pred) / y_std
            crps = np.mean(
                y_std * (z * (2 * stats.norm.cdf(z) - 1) +
                        2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
            )
            metrics['crps'] = crps
        else:
            # Compute from predictions
            from scipy import stats

            predictions = self.predict()
            y_lower = predictions.lower.flatten()
            y_upper = predictions.upper.flatten()
            y_true_full = self._y_matrix.flatten()
            y_pred_full = predictions.total.flatten()

            # 95% coverage
            coverage_95 = np.mean((y_true_full >= y_lower) & (y_true_full <= y_upper))
            metrics['coverage_95'] = coverage_95

            # 90% coverage
            y_std = (y_upper - y_lower) / (2 * 1.96)
            z_90 = stats.norm.ppf(0.95)
            lower_90 = y_pred_full - z_90 * y_std
            upper_90 = y_pred_full + z_90 * y_std
            coverage_90 = np.mean((y_true_full >= lower_90) & (y_true_full <= upper_90))
            metrics['coverage_90'] = coverage_90

            # CRPS
            z = (y_true_full - y_pred_full) / y_std
            crps = np.mean(
                y_std * (z * (2 * stats.norm.cdf(z) - 1) +
                        2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
            )
            metrics['crps'] = crps

        return metrics

    def evaluate_in_observation_order(
        self,
        y_true: NDArray,
        time_index: Union[NDArray, pd.Series],
        location_index: Union[NDArray, pd.Series],
        X: Optional[Union[NDArray, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """Evaluate metrics using predictions aligned to observation order.

        Useful when the input data are not already sorted in the internal
        (time, location) grid order used by the model.
        """
        aligned_pred = self.predict_in_observation_order(
            time_index=time_index,
            location_index=location_index,
            X=X,
        )
        return self.evaluate(y_true=y_true, y_pred=aligned_pred)
        
    def summary(self) -> HybridSummary:
        """Get model summary.
        
        Returns
        -------
        HybridSummary
            Summary of model components and performance
        """
        self._check_fitted()
        
        predictions = self.predict()
        metrics = self.evaluate(
            self._y_matrix.flatten(),
            predictions.total.flatten(),
            predictions.lower.flatten(),
            predictions.upper.flatten(),
        )
        
        return HybridSummary(
            gam_summary=self.gam_.summary(),
            ssm_diagnostics=self.ssm_.get_diagnostics(),
            total_rmse=metrics['rmse'],
            total_mae=metrics['mae'],
            total_r2=metrics['r2'],
            coverage_95=metrics['coverage_95'],
        )
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from GAM component.
        
        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        self._check_fitted()
        return self.gam_.get_feature_importance()
        
    def get_smoothed_states(self) -> NDArray:
        """Get smoothed latent states from SSM.
        
        Returns
        -------
        NDArray
            Smoothed states, shape (T, n_locations)
        """
        self._check_fitted()
        return self.ssm_.smoother_result_.smoothed_means
        
    def get_em_convergence(self) -> pd.DataFrame:
        """Get EM algorithm convergence history.
        
        Returns
        -------
        pd.DataFrame
            Convergence diagnostics
        """
        self._check_fitted()
        return self.ssm_.get_em_history()
        
    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
    def save(self, directory: Union[str, Path]) -> None:
        """Save model to directory.
        
        Parameters
        ----------
        directory : str or Path
            Directory to save model files
        """
        self._check_fitted()
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save components
        self.gam_.save(directory / "gam.pkl")
        self.ssm_.save(directory / "ssm.pkl")
        
        # Save metadata
        import json
        metadata = {
            'n_splines': self.n_splines,
            'gam_lam': self.gam_lam if isinstance(self.gam_lam, float) else 'auto',
            'state_dim': self.state_dim,
            'em_max_iter': self.em_max_iter,
            'em_tol': self.em_tol,
            'scalability_mode': self.scalability_mode,
            'regularization': self.regularization,
            'confidence_level': self.confidence_level,
            'n_locations': self.n_locations_,
            'n_times': self.n_times_,
        }
        with open(directory / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save training data
        np.savez(
            directory / "training_data.npz",
            X_train=self._X_train,
            y_train=self._y_train,
            y_matrix=self._y_matrix,
            residual_matrix=self._residual_matrix,
            location_ids=self.location_ids_,
            time_ids=self.time_ids_,
        )
        
        logger.info(f"Model saved to {directory}")
        
    @classmethod
    def load(cls, directory: Union[str, Path]) -> "HybridGAMSSM":
        """Load model from directory.
        
        Parameters
        ----------
        directory : str or Path
            Directory containing saved model files
            
        Returns
        -------
        HybridGAMSSM
            Loaded model
        """
        import json
        
        directory = Path(directory)
        
        # Load metadata
        with open(directory / "metadata.json", 'r') as f:
            metadata = json.load(f)
            
        # Create model
        model = cls(
            n_splines=metadata['n_splines'],
            gam_lam=metadata['gam_lam'],
            state_dim=metadata['state_dim'],
            em_max_iter=metadata['em_max_iter'],
            em_tol=metadata['em_tol'],
            scalability_mode=metadata['scalability_mode'],
            regularization=metadata['regularization'],
            confidence_level=metadata['confidence_level'],
        )
        
        # Load components
        model.gam_ = SpatialGAM.load(directory / "gam.pkl")
        model.ssm_ = StateSpaceModel.load(directory / "ssm.pkl")
        
        # Load training data
        data = np.load(directory / "training_data.npz", allow_pickle=True)
        model._X_train = data['X_train']
        model._y_train = data['y_train']
        model._y_matrix = data['y_matrix']
        model._residual_matrix = data['residual_matrix']
        model.location_ids_ = data['location_ids']
        model.time_ids_ = data['time_ids']
        
        # Recompute SSM filter/smoother results for predictions using stored residuals
        model.ssm_._restore_inference(model._residual_matrix)
        
        model.n_locations_ = metadata['n_locations']
        model.n_times_ = metadata['n_times']
        model.is_fitted_ = True
        
        logger.info(f"Model loaded from {directory}")
        return model
