"""
Spatial Generalized Additive Model (GAM) for Land Use Regression.

This module implements a GAM-based approach to land use regression, capturing
non-linear relationships between spatial/environmental covariates and air
pollutant concentrations.

References
----------
.. [1] Hastie, T., & Tibshirani, R. (1986). Generalized additive models.
       Statistical Science, 1(3), 297-310.
.. [2] Wood, S. N. (2017). Generalized Additive Models: An Introduction with R.
       Chapman and Hall/CRC.
.. [3] Hoek, G., et al. (2008). A review of land-use regression models to assess
       spatial variation of outdoor air pollution. Atmospheric Environment, 42(33).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

try:
    from pygam import GAM, LinearGAM, s, te, l
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False

from gam_ssm_lur.models.base import BaseEstimator, ModelSummary
from gam_ssm_lur.utils import (
    ensure_array,
    extract_feature_names,
    compute_r_squared,
    compute_aic,
    compute_bic,
)

logger = logging.getLogger(__name__)


@dataclass
class GAMSummary:
    """Summary statistics for fitted GAM.
    
    Attributes
    ----------
    r_squared : float
        Coefficient of determination
    adj_r_squared : float
        Adjusted R-squared
    deviance_explained : float
        Proportion of deviance explained
    gcv_score : float
        Generalized Cross-Validation score
    aic : float
        Akaike Information Criterion
    n_samples : int
        Number of training samples
    n_features : int
        Number of features
    edfs : NDArray
        Effective degrees of freedom for each smooth term
    feature_names : List[str]
        Names of features
    """
    r_squared: float
    adj_r_squared: float
    deviance_explained: float
    gcv_score: float
    aic: float
    n_samples: int
    n_features: int
    edfs: NDArray
    feature_names: List[str]


@dataclass
class PartialDependence:
    """Partial dependence results for a single feature.
    
    Attributes
    ----------
    feature_name : str
        Name of the feature
    grid : NDArray
        Feature values on the grid
    response : NDArray
        Partial dependence values
    confidence_lower : NDArray
        Lower confidence bound
    confidence_upper : NDArray
        Upper confidence bound
    """
    feature_name: str
    grid: NDArray
    response: NDArray
    confidence_lower: NDArray
    confidence_upper: NDArray


class SpatialGAM(BaseEstimator):
    """Generalized Additive Model for spatial air pollution prediction.

    Implements a GAM-based Land Use Regression model that captures non-linear
    relationships between pollutant concentrations and spatial covariates
    (land use, road network, traffic).

    Parameters
    ----------
    n_splines : int
        Number of spline basis functions per smooth term
    spline_order : int
        Order of B-splines (default 3 = cubic)
    lam : float or 'auto'
        Smoothing parameter. 'auto' uses grid search.
    link : {'identity', 'log'}
        Link function for the GAM
    distribution : {'normal', 'gamma', 'poisson'}
        Response distribution family
    fit_intercept : bool
        Whether to fit an intercept term
    max_iter : int
        Maximum iterations for fitting
    tol : float
        Convergence tolerance

    Attributes
    ----------
    gam_ : GAM
        Fitted pygam model
    feature_names_ : List[str]
        Names of features used in fitting
    is_fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    >>> model = SpatialGAM(n_splines=10, lam='auto')
    >>> model.fit(X, y)
    >>> predictions = model.predict(X_new)
    >>> summary = model.summary()
    """

    def __init__(
        self,
        n_splines: int = 10,
        spline_order: int = 3,
        lam: Union[float, Literal["auto"]] = "auto",
        link: Literal["identity", "log"] = "identity",
        distribution: Literal["normal", "gamma", "poisson"] = "normal",
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        super().__init__()

        if not HAS_PYGAM:
            raise ImportError("pygam is required for SpatialGAM. Install with: pip install pygam")

        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam
        self.link = link
        self.distribution = distribution
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

        # pygam requires n_splines > spline_order; bump if user passes too few
        if self.n_splines <= self.spline_order:
            logger.info(
                "Increasing n_splines from %d to %d to satisfy pygam constraint "
                "n_splines > spline_order",
                self.n_splines,
                self.spline_order + 1,
            )
            self.n_splines = self.spline_order + 1

        self.gam_: Optional[GAM] = None
        self.feature_names_: Optional[List[str]] = None
        self._X_train: Optional[NDArray] = None
        self._y_train: Optional[NDArray] = None
        
    def _build_formula(self, n_features: int) -> GAM:
        """Build GAM formula with spline terms for each feature."""
        # Build sum of spline terms
        terms = s(0, n_splines=self.n_splines, spline_order=self.spline_order)
        for i in range(1, n_features):
            terms += s(i, n_splines=self.n_splines, spline_order=self.spline_order)
            
        return LinearGAM(
            terms,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        
    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Union[NDArray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> "SpatialGAM":
        """Fit the GAM to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features (spatial covariates)
        y : array-like of shape (n_samples,)
            Target values (pollutant concentrations)
        feature_names : list of str, optional
            Names for each feature. Inferred from DataFrame if available.
            
        Returns
        -------
        self : SpatialGAM
            Fitted model
        """
        # Handle pandas inputs using utility functions
        X, self.feature_names_ = extract_feature_names(X, feature_names)
        y = ensure_array(y)

        # Store training data for residual computation
        self._X_train = X
        self._y_train = y
        
        n_samples, n_features = X.shape
        logger.info(f"Fitting GAM with {n_samples} samples and {n_features} features")
        
        # Build and fit model
        self.gam_ = self._build_formula(n_features)
        
        if self.lam == "auto":
            # Grid search for optimal smoothing parameter
            logger.info("Performing grid search for smoothing parameters")
            lam_grid = np.logspace(-3, 3, 11)
            self.gam_.gridsearch(X, y, lam=lam_grid)
        else:
            self.gam_.fit(X, y)
            
        self.is_fitted_ = True
        logger.info(f"GAM fitted. R²={self.gam_.statistics_['pseudo_r2']['explained_deviance']:.4f}")
        
        return self
        
    def predict(
        self,
        X: Union[NDArray, pd.DataFrame],
        return_std: bool = False,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Generate predictions for new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for prediction
        return_std : bool
            If True, also return prediction standard errors
            
        Returns
        -------
        y_pred : NDArray
            Predicted values
        y_std : NDArray, optional
            Prediction standard errors (if return_std=True)
        """
        self._check_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        
        y_pred = self.gam_.predict(X)
        
        if return_std:
            # Compute prediction intervals
            y_intervals = self.gam_.prediction_intervals(X, width=0.95)
            y_std = (y_intervals[:, 1] - y_intervals[:, 0]) / (2 * 1.96)
            return y_pred, y_std
            
        return y_pred
        
    def get_residuals(
        self,
        X: Optional[Union[NDArray, pd.DataFrame]] = None,
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> NDArray:
        """Compute residuals from fitted model.
        
        Parameters
        ----------
        X : array-like, optional
            Features. Uses training data if not provided.
        y : array-like, optional
            Targets. Uses training data if not provided.
            
        Returns
        -------
        residuals : NDArray
            Residuals (y - y_pred)
        """
        self._check_fitted()
        
        if X is None:
            X = self._X_train
        if y is None:
            y = self._y_train
            
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        y_pred = self.predict(X)
        return y - y_pred
        
    def summary(self) -> GAMSummary:
        """Get model summary statistics.
        
        Returns
        -------
        GAMSummary
            Container with model statistics
        """
        self._check_fitted()
        
        stats = self.gam_.statistics_
        
        # Compute R-squared
        y_pred = self.predict(self._X_train)
        ss_res = np.sum((self._y_train - y_pred) ** 2)
        ss_tot = np.sum((self._y_train - np.mean(self._y_train)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
        n = len(self._y_train)
        p = self._X_train.shape[1]
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        return GAMSummary(
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            deviance_explained=stats['pseudo_r2']['explained_deviance'],
            gcv_score=stats.get('GCV', np.nan),
            aic=stats.get('AIC', np.nan),
            n_samples=n,
            n_features=p,
            edfs=np.array(stats['edof_per_coef']),
            feature_names=self.feature_names_,
        )
        
    def partial_dependence(
        self,
        feature_idx: int,
        grid_size: int = 100,
        confidence_level: float = 0.95,
    ) -> PartialDependence:
        """Compute partial dependence for a single feature.
        
        Parameters
        ----------
        feature_idx : int
            Index of feature to analyze
        grid_size : int
            Number of points in the grid
        confidence_level : float
            Confidence level for intervals
            
        Returns
        -------
        PartialDependence
            Partial dependence results
        """
        self._check_fitted()
        
        # Get feature range from training data
        feature_values = self._X_train[:, feature_idx]
        grid = np.linspace(feature_values.min(), feature_values.max(), grid_size)
        
        # Compute partial dependence using pygam
        XX = self.gam_.generate_X_grid(term=feature_idx, n=grid_size)
        pdep = self.gam_.partial_dependence(term=feature_idx, X=XX, width=confidence_level)
        
        return PartialDependence(
            feature_name=self.feature_names_[feature_idx],
            grid=grid,
            response=pdep[:, 0],
            confidence_lower=pdep[:, 1],
            confidence_upper=pdep[:, 2],
        )
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Compute feature importance based on deviance explained.
        
        Returns
        -------
        pd.DataFrame
            Feature importance scores sorted by importance
        """
        self._check_fitted()
        
        # Use effective degrees of freedom as a proxy for importance
        edfs = self.gam_.statistics_['edof_per_coef']
        
        # Compute partial R-squared for each feature
        importances = []
        y_pred_full = self.predict(self._X_train)
        ss_res_full = np.sum((self._y_train - y_pred_full) ** 2)
        
        for i in range(self._X_train.shape[1]):
            # Crude approximation: zero out feature and recompute
            X_zeroed = self._X_train.copy()
            X_zeroed[:, i] = np.mean(X_zeroed[:, i])
            y_pred_zeroed = self.predict(X_zeroed)
            ss_res_zeroed = np.sum((self._y_train - y_pred_zeroed) ** 2)
            
            # Increase in residual SS = importance
            importance = (ss_res_zeroed - ss_res_full) / ss_res_full
            importances.append(importance)
            
        df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances,
            'edf': edfs[:len(self.feature_names_)],
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
        
    def _get_state_dict(self) -> Dict:
        """Get model state for serialization."""
        import pickle
        return {
            'gam': self.gam_,
            'feature_names': self.feature_names_,
            'params': {
                'n_splines': self.n_splines,
                'spline_order': self.spline_order,
                'lam': self.lam,
                'link': self.link,
                'distribution': self.distribution,
                'fit_intercept': self.fit_intercept,
                'max_iter': self.max_iter,
                'tol': self.tol,
            },
            'training_data': {
                'X': self._X_train,
                'y': self._y_train,
            }
        }

    def _set_state_dict(self, state: Dict) -> None:
        """Set model state from deserialization."""
        self.gam_ = state['gam']
        self.feature_names_ = state['feature_names']
        params = state['params']
        self.n_splines = params['n_splines']
        self.spline_order = params['spline_order']
        self.lam = params['lam']
        self.link = params['link']
        self.distribution = params['distribution']
        self.fit_intercept = params['fit_intercept']
        self.max_iter = params['max_iter']
        self.tol = params['tol']
        if 'training_data' in state:
            self._X_train = state['training_data']['X']
            self._y_train = state['training_data']['y']

    def save(self, filepath: str) -> None:
        """Save fitted model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save model
        """
        import pickle
        
        self._check_fitted()
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'gam': self.gam_,
                'feature_names': self.feature_names_,
                'params': {
                    'n_splines': self.n_splines,
                    'spline_order': self.spline_order,
                    'lam': self.lam,
                    'link': self.link,
                    'distribution': self.distribution,
                    'fit_intercept': self.fit_intercept,
                },
                'training_data': {
                    'X': self._X_train,
                    'y': self._y_train,
                },
            }, f)
            
        logger.info(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> "SpatialGAM":
        """Load fitted model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to saved model
            
        Returns
        -------
        SpatialGAM
            Loaded model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        model = cls(**data['params'])
        model.gam_ = data['gam']
        model.feature_names_ = data['feature_names']
        model._X_train = data['training_data']['X']
        model._y_train = data['training_data']['y']
        model.is_fitted_ = True
        
        logger.info(f"Model loaded from {filepath}")
        return model
