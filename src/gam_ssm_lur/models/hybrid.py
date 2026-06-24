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
from gam_ssm_lur.data import SpatiotemporalDataset, StaticData, TemporalData, CalibrationResult


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
        state_dim: int = 3,
        em_max_iter: int = 50,
        em_tol: float = 1e-4,
        scalability_mode: Literal["auto", "dense", "diagonal", "block"] = "auto",
        regularization: float = 1e-6,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        max_eigenvalue: float = 0.98,
    ):
        self.n_splines = n_splines
        self.gam_lam = gam_lam
        self.state_dim = state_dim
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.scalability_mode = scalability_mode
        self.regularization = regularization
        self.confidence_level = confidence_level
        # Maximum spectral radius for the SSM's dynamic-block transition
        # matrix (Hamilton, 1994, Ch. 1). Lower values more aggressively
        # suppress within-sample drift over the study window.
        self.max_eigenvalue = max_eigenvalue
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
        self._residual_matrix: Optional[NDArray] = None  # calibrated satellite field reshaped
        self._location_index_train: Optional[NDArray] = None
        self._time_index_train: Optional[NDArray] = None

        # Spatial loading matrix: maps (state_dim,) temporal state → (n_locations,)
        self.Z_spatial_: Optional[NDArray] = None

        # Per-location SVD truncation residual variance (variance NOT captured by
        # the top state_dim SVD factors, stored so predict() can include it).
        self._truncation_var_: Optional[NDArray] = None

        # Variance of calibration residuals (satellite vs surface EPA), representing
        # the measurement representativeness error when predicting surface NO₂.
        self._sigma2_obs: float = 0.0

        # Optional spatially-resolved traffic correction (T, n_locations), calibrated
        # against EPA residuals via traffic_field.calibrate_traffic_field(). Kept
        # separate from the satellite-driven SSM because that component is bottlenecked
        # to k SVD factors and cannot represent road-network-scale spatial detail.
        self._traffic_field: Optional[NDArray] = None
        self._traffic_calibration = None

        # True GAM coefficient (beta_init from pooled OLS + jointly-estimated
        # beta_delta), used by predict() to scale the GAM baseline consistently
        # with what the SSM's own observation equation estimates -- see
        # fit_from_dataset. None until fitted.
        self.beta_total_: Optional[float] = None

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
        
        # Step 2: Project residuals to low-dimensional factors via SVD,
        # then fit SSM on the projected temporal scores.
        logger.info("Step 2: Fitting SSM on GAM residuals")
        residual_filled = np.where(np.isnan(residual_matrix), 0.0, residual_matrix)
        U, s, Vt = np.linalg.svd(residual_filled, full_matrices=False)
        k = min(self.state_dim, len(s))
        obs_projected = U[:, :k] * s[:k]       # (T, k)
        self.Z_spatial_ = Vt[:k, :].T          # (n_locations, k)

        self.ssm_ = StateSpaceModel(
            state_dim=k,
            em_max_iter=self.em_max_iter,
            em_tol=self.em_tol,
            scalability_mode=self.scalability_mode,
            regularization=self.regularization,
            random_state=self.random_state,
            max_eigenvalue=self.max_eigenvalue,
        )
        self.ssm_.fit(obs_projected)

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
        
    def fit_from_dataset(
        self,
        static: "StaticData",
        temporal: "TemporalData",
        calibration: Optional["CalibrationResult"] = None,
        feature_selector: Optional[object] = None,
        id_cols: Optional[List[str]] = None,
        target_col: str = "atmos_no2",
    ) -> "HybridGAMSSM":
        """Fit the hybrid model from a loaded :class:`SpatiotemporalDataset`.

        This is the high-level entry point when using the standard Dublin-style
        data contract. It handles:

        1. Feature preparation (dropping id columns, applying selector if given).
        2. Static GAM-LUR fitting on the spatial features and target.
        3. Building the forcing matrix from activity anomaly and met forcing.
        4. SSM fitting on the GAM residuals with external forcing.

        The dense gridded observations (e.g. satellite) and the calibration
        coefficients are stored and used by :meth:`predict` for Kalman updates.
        Point observations (e.g. EPA stations) are stored for validation only.

        Parameters
        ----------
        static : StaticData
            Output of :meth:`SpatiotemporalDataset.load_static`.
        temporal : TemporalData
            Output of :meth:`SpatiotemporalDataset.load_temporal`.
        calibration : CalibrationResult, optional
            Output of :meth:`SpatiotemporalDataset.calibrate_dense_obs`.
            If None, dense observations enter the filter uncalibrated.
        feature_selector : FeatureSelector, optional
            Pre-fitted or to-be-fitted feature selector. If None, all
            non-id columns from ``static.features`` are used.
        id_cols : list of str, optional
            Columns to exclude from the feature matrix.
            Default: ``["grid_id", "latitude", "longitude"]``.
        target_col : str
            Name of the target column in ``static.target``.
            Default ``atmos_no2``.

        Returns
        -------
        self : HybridGAMSSM
            Fitted model.
        """
        from gam_ssm_lur.features import FeatureSelector

        if id_cols is None:
            id_cols = ["grid_id", "latitude", "longitude"]

        # ── Prepare feature matrix ───────────────────────────────────────────
        feat_cols = [c for c in static.features.columns if c not in id_cols]
        X_df = static.features[feat_cols]

        # Align target with features on grid_id
        target_merged = static.features[["grid_id"]].merge(
            static.target[["grid_id", target_col]], on="grid_id", how="left"
        )
        y = target_merged[target_col].values

        if feature_selector is not None:
            if not getattr(feature_selector, "_is_fitted", False):
                X_df = feature_selector.fit_transform(X_df, y)
            else:
                X_df = feature_selector.transform(X_df)

        X = X_df.values
        feature_names = list(X_df.columns)

        # ── Build forcing matrix (T, n_forcing) ─────────────────────────────
        dates = temporal.dates
        T = len(dates)

        # Activity forcing: city-wide scalar per day → shape (T, 1)
        act_map = dict(
            zip(temporal.activity_forcing["date"],
                temporal.activity_forcing["delta_activity"])
        )
        activity_vec = np.array([act_map.get(d, 0.0) for d in dates])  # (T,)

        # Met forcing: per-cell per-day → aggregate to city mean for SSM
        # (cell-level spatial variation captured in B_ via OLS)
        if "grid_id" in temporal.met_forcing.columns:
            met_pivot = temporal.met_forcing.pivot(
                index="date", columns="grid_id", values="met_forcing"
            )
            met_mean = met_pivot.reindex(dates).mean(axis=1).fillna(0.0).values  # (T,)
        else:
            met_map = temporal.met_forcing.set_index("date")["met_forcing"].to_dict()
            met_mean = np.array([met_map.get(d, 0.0) for d in dates])  # (T,)

        # Leading all-ones column gives a jointly-estimated intercept (its B_tilde
        # column absorbs any systematic scale gap between the calibrated
        # satellite field and the GAM baseline -- e.g. the ~3.5 ug/m3 GAM-vs-EPA
        # bias -- via the same augmented EM as traffic/wind, rather than a
        # separately pre-computed mean offset).
        forcing_matrix = np.column_stack([np.ones(T), activity_vec, met_mean])  # (T, 3)

        # Store sigma2_obs: represents satellite-to-surface measurement uncertainty.
        # Included in prediction intervals so they reflect uncertainty at point
        # locations (EPA stations), not just at the satellite grid scale.
        self._sigma2_obs = float(calibration.sigma2_obs) if calibration is not None else 0.0

        # Store for later use (calibration, validation)
        self._calibration = calibration
        self._temporal = temporal
        self._static = static
        self._dates = dates

        logger.info("Fitting GAM spatial component")
        self.gam_ = SpatialGAM(n_splines=self.n_splines, lam=self.gam_lam)
        self.gam_.fit(X, y, feature_names=feature_names)
        gam_residuals = self.gam_.get_residuals(X, y)

        # Residuals are per-cell (not time-varying): treat as T=1 for now
        # The full spatiotemporal residual matrix is built from dense obs later
        self._X_train = X
        self._y_train = y
        self.n_locations_ = len(y)
        self.n_times_ = T
        self.location_ids_ = np.array(static.grid_ids)
        self.time_ids_ = np.array(dates)
        self._y_matrix = None            # no panel obs available for SSM init
        self._residual_matrix = None

        # Build residual matrix from dense obs (satellite) for SSM fitting
        lur_prior = self.gam_.predict(X)
        grid_id_to_idx = {gid: i for i, gid in enumerate(static.grid_ids)}
        lur_prior_arr = lur_prior  # (n_locations,)

        # Vectorised: calibrate all satellite obs then pivot to (T, n_locations)
        dense = temporal.dense_obs.copy()
        if calibration is not None:
            dense["obs_dense"] = calibration.apply(dense["obs_dense"].values)

        # Map grid_id → location index; drop cells not in the model grid
        dense["loc_idx"] = dense["grid_id"].map(grid_id_to_idx)
        dense = dense.dropna(subset=["loc_idx"])
        dense["loc_idx"] = dense["loc_idx"].astype(int)

        # Map dates → time index
        date_to_tidx = {d: i for i, d in enumerate(dates)}
        dense["t_idx"] = dense["date"].map(date_to_tidx)
        dense = dense.dropna(subset=["t_idx"])
        dense["t_idx"] = dense["t_idx"].astype(int)

        # Initial GAM coefficient via simple pooled OLS (y ~ beta_init * GAM),
        # used ONLY to choose a sensible matrix for the loadings SVD below --
        # NOT assumed fixed. Using raw y (implicit beta=0) or y-GAM (implicit
        # beta=1, the original bug) both make the loadings step internally
        # inconsistent with whatever beta the regression step later estimates;
        # this two-step estimator (Doz, Giannone & Reichlin, 2011) calls for
        # a *sensible* initial matrix, not an arbitrary extreme.
        # Fit WITH an intercept: forcing the line through the origin would
        # badly bias the slope given the calibration step's own intercept
        # (~6 in this dataset) -- y is not expected to be ~0 when GAM is ~0.
        gam_at_obs = lur_prior_arr[dense["loc_idx"].values]
        design = np.column_stack([np.ones(len(gam_at_obs)), gam_at_obs])
        ols_coeffs, _, _, _ = np.linalg.lstsq(design, dense["obs_dense"].values, rcond=None)
        beta0_init, beta_init = float(ols_coeffs[0]), float(ols_coeffs[1])
        logger.info(
            "Initial GAM relationship (pooled OLS with intercept, for SVD basis only): "
            "y = %.4f + %.4f * GAM", beta0_init, beta_init,
        )

        dense["residual"] = dense["obs_dense"].values - beta_init * gam_at_obs
        y_matrix = np.full((T, self.n_locations_), np.nan)
        y_matrix[dense["t_idx"].values, dense["loc_idx"].values] = dense["residual"].values
        self._residual_matrix = y_matrix

        # ── Project the deviation-from-(beta_init*GAM) field to low-dimensional
        # spatial loadings ── NaN cells (missing satellite obs) filled with 0
        # before decomposition.
        state_dim = self.state_dim
        y_filled = np.where(np.isnan(y_matrix), 0.0, y_matrix)
        U, s, Vt = np.linalg.svd(y_filled, full_matrices=False)
        k = min(state_dim, len(s))
        scores = U[:, :k] * s[:k]            # (T, k) — temporal scores of the deviation field
        self.Z_spatial_ = Vt[:k, :].T        # (n_locations, k) — spatial loadings (Lambda)
        var_explained = 100 * (s[:k] ** 2).sum() / (s ** 2).sum()
        logger.info(
            "Deviation-field SVD: top %d factors explain %.1f%% of variance → SSM obs_dim=%d",
            k, var_explained, k,
        )

        # Project the GAM baseline into score-space: g_tilde = Lambda' @ GAM_prior.
        # Lambda's columns are orthonormal (right singular vectors of an SVD),
        # so this is also Lambda's pseudo-inverse projection. The EM jointly
        # estimates beta_delta (the residual GAM coefficient beyond beta_init)
        # alongside the forcing effects and dynamics, below; the true GAM
        # coefficient is beta_init + beta_delta (see self.beta_total_).
        g_tilde = self.Z_spatial_.T @ lur_prior_arr  # (k,)

        # Per-location variance of the field NOT captured by the top k SVD
        # factors. These are included in prediction intervals so that spatial
        # patterns outside the low-dimensional subspace contribute to uncertainty.
        y_reconstructed = (U[:, :k] * s[:k]) @ Vt[:k, :]   # (T, n_locations)
        truncation_error = y_filled - y_reconstructed       # (T, n_locations)
        valid_mask = ~np.isnan(y_matrix)
        self._truncation_var_ = np.where(
            valid_mask.sum(axis=0) > 1,
            np.nanvar(np.where(valid_mask, truncation_error, np.nan), axis=0),
            0.0,
        )
        logger.info(
            "Truncation variance: mean=%.3f, median=%.3f µg/m³ (added to prediction std)",
            self._truncation_var_.mean(), np.median(self._truncation_var_),
        )

        logger.info("Fitting SSM with jointly-estimated GAM coefficient and forcing effects")
        self.ssm_ = StateSpaceModel(
            state_dim=k,
            em_max_iter=self.em_max_iter,
            em_tol=self.em_tol,
            scalability_mode=self.scalability_mode,
            regularization=self.regularization,
            random_state=self.random_state,
            max_eigenvalue=self.max_eigenvalue,
        )
        self.ssm_.fit(scores, forcing_matrix=forcing_matrix, g_tilde=g_tilde)

        # True GAM coefficient = the initial pooled-OLS estimate (used only
        # to choose the SVD basis above) plus the jointly-estimated residual
        # correction. This is what predict() uses, not beta_init or 1.0 alone.
        self.beta_total_ = beta_init + self.ssm_.beta_
        logger.info(
            "GAM coefficient: beta_init=%.4f + beta_delta=%.4f = beta_total=%.4f "
            "(old buggy design assumed 1.0)",
            beta_init, self.ssm_.beta_, self.beta_total_,
        )

        self.is_fitted_ = True
        logger.info("HybridGAMSSM fitted via fit_from_dataset()")
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

        # Reshape to (T, n_locations).
        # fit_from_dataset provides static X (n_locations,) — tile across time.
        if gam_pred.shape[0] == self.n_locations_:
            spatial_matrix  = np.tile(gam_pred, (self.n_times_, 1))
            gam_std_tiled   = np.tile(gam_std,  (self.n_times_, 1))
        else:
            spatial_matrix  = gam_pred.reshape(self.n_times_, self.n_locations_)
            gam_std_tiled   = gam_std.reshape(self.n_times_, self.n_locations_)
        
        # Temporal predictions from SSM → expand to all locations via Z_spatial_.
        # ssm_.predict() returns only the dynamic (alpha) sub-block -- beta and
        # B_tilde are separate, jointly-estimated fixed-effect states (see
        # add_traffic_correction / fit_from_dataset), not folded into alpha_t.
        ssm_pred = self.ssm_.predict(confidence_level=self.confidence_level)
        # ssm_pred.mean: (T, k),  Z_spatial_: (n_locations, k)
        temporal_matrix = ssm_pred.mean @ self.Z_spatial_.T          # (T, n_locations)
        temporal_std = np.sqrt(
            (ssm_pred.std ** 2) @ (self.Z_spatial_ ** 2).T           # (T, n_locations)
        )

        # Forcing contribution (traffic anomaly, wind): B_tilde @ u_t in score
        # space, mapped to all locations via the same spatial loadings Lambda.
        # beta*GAM is intentionally NOT added here -- it was only used
        # internally (via g_tilde) to correctly separate genuine satellite
        # dynamics from the GAM-correlated part of the signal; the GAM's own
        # contribution to the surface prediction is spatial_matrix above, at
        # its native annual-mean scale, not beta-scaled.
        if self.ssm_.B_ is not None and self.ssm_._forcing_matrix is not None:
            forcing_scores = self.ssm_._forcing_matrix @ self.ssm_.B_.T   # (T, k)
            forcing_matrix_term = forcing_scores @ self.Z_spatial_.T      # (T, n_locations)
        else:
            forcing_matrix_term = 0.0

        # Total prediction = beta_total * GAM baseline + temporal deviation
        # (satellite-driven dynamics + forcing). beta_total (see
        # fit_from_dataset) is the model's own jointly-estimated relationship
        # between calibrated satellite/EPA-scale observations and the GAM --
        # using 1.0 here (as the original buggy design implicitly did) would
        # be internally inconsistent with what the SSM's observation equation
        # actually estimates. spatial_matrix itself (returned unscaled, below)
        # still represents the GAM's own native annual-mean prediction.
        beta_total = self.beta_total_ if self.beta_total_ is not None else 1.0
        total = beta_total * spatial_matrix + temporal_matrix + forcing_matrix_term

        # Optional spatially-resolved traffic correction (see add_traffic_correction).
        # Added independently of the satellite-driven SSM term so it can carry
        # road-network-scale spatial detail the k-factor SVD bottleneck discards.
        if self._traffic_field is not None and self._traffic_calibration is not None:
            traffic_term = self._traffic_calibration.apply(self._traffic_field)
            total = total + np.nan_to_num(traffic_term, nan=0.0)

        # SSM observation noise (H) propagated through Z_spatial_.
        # H is estimated in the k-dimensional projected (temporal-score) space;
        # each diagonal entry is the noise variance for one SVD factor.
        # Propagating through Z_spatial_ gives the contribution per location.
        if self.ssm_.H_ is not None:
            h_diag = (np.diag(self.ssm_.H_) if self.ssm_.H_.ndim == 2
                      else self.ssm_.H_.ravel())
            obs_noise_var = h_diag @ (self.Z_spatial_ ** 2).T    # (n_locations,)
        else:
            obs_noise_var = np.zeros(self.n_locations_)

        # SVD truncation variance: variance of residuals not captured by top k factors.
        trunc_var = (self._truncation_var_
                     if self._truncation_var_ is not None
                     else np.zeros(self.n_locations_))

        # Combined uncertainty (independence assumption across components).
        # Note: sigma2_obs (satellite-to-surface mismatch) is intentionally excluded
        # here because it absorbs all TROPOMI-EPA correlation noise and would produce
        # uninformatively wide intervals (~78 µg/m³ width for r≈0.2).
        # Use calibrate_intervals_conformal() for properly calibrated prediction bands.
        combined_std = np.sqrt(
            gam_std_tiled ** 2
            + temporal_std ** 2
            + obs_noise_var[np.newaxis, :]     # broadcast: (1, n_locations)
            + trunc_var[np.newaxis, :]          # broadcast: (1, n_locations)
        )

        from scipy import stats
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        lower = total - z_score * combined_std
        upper = total + z_score * combined_std
        
        return HybridPrediction(
            total=total,
            spatial=spatial_matrix,
            temporal=temporal_matrix,
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
        
        # Temporal forecast → expand to all locations via Z_spatial_
        ssm_forecast = self.ssm_.forecast(n_steps, confidence_level=self.confidence_level)
        temporal_matrix = ssm_forecast.mean @ self.Z_spatial_.T
        temporal_std = np.sqrt((ssm_forecast.std ** 2) @ (self.Z_spatial_ ** 2).T)

        # Combine. No forcing term here: ssm_.forecast() extrapolates only the
        # dynamic (alpha) sub-block, since future traffic/wind values are not
        # available to this method's signature; beta*GAM is likewise not
        # added, for the same reason as in predict() above.
        total = spatial_matrix + temporal_matrix
        combined_std = np.sqrt(gam_std_matrix ** 2 + temporal_std ** 2)
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        lower = total - z_score * combined_std
        upper = total + z_score * combined_std
        
        return HybridPrediction(
            total=total,
            spatial=spatial_matrix,
            temporal=temporal_matrix,
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

    def add_traffic_correction(
        self,
        traffic_field: NDArray,
        epa_eval: pd.DataFrame,
        y_col: str = "obs_value",
    ) -> "TrafficFieldCalibration":
        """Fit and attach a spatially-resolved traffic correction term.

        Must be called after :meth:`fit` / :meth:`fit_from_dataset`. Computes
        the model's current (GAM + satellite-SSM) prediction, calibrates
        ``traffic_field`` against the residual EPA variance that prediction
        fails to explain, and stores the result so :meth:`predict` adds it as
        an independent additive term — kept outside the k-factor SVD
        bottleneck so it can carry road-network-scale spatial detail that the
        satellite-driven component cannot.

        Parameters
        ----------
        traffic_field : (T, n_locations) ndarray
            Output of ``traffic_field.compute_traffic_field``, in the same
            (time, location) order as ``self.time_ids_`` / ``self.location_ids_``.
        epa_eval : pd.DataFrame
            Must contain ``t_idx``, ``loc_idx``, and ``y_col`` columns.

        Returns
        -------
        TrafficFieldCalibration
            The fitted beta0/beta1/r, also stored on the model.
        """
        from gam_ssm_lur.traffic_field import calibrate_traffic_field

        self._check_fitted()
        baseline_pred = self.predict().total  # GAM + satellite-SSM only (traffic not yet attached)
        calibration = calibrate_traffic_field(
            traffic_field=traffic_field,
            gam_ssm_pred=baseline_pred,
            epa_eval=epa_eval,
            y_col=y_col,
        )
        self._traffic_field = traffic_field
        self._traffic_calibration = calibration
        return calibration

    def calibrate_intervals_conformal(
        self,
        y_obs: NDArray,
        t_idx: NDArray,
        loc_idx: NDArray,
        alpha: float = 0.05,
        station_ids: Optional[NDArray] = None,
    ) -> float:
        """Estimate a conformal prediction scale factor using held-out EPA observations.

        Implements leave-one-station-out (LOSO) split conformal prediction:
        for each station, calibrate on the remaining stations, compute the
        (1-α) normalised-residual quantile, and report the median correction
        factor across leave-one-out folds.

        The resulting scale factor ``q_hat`` should be multiplied by the
        prediction std returned by :meth:`predict` to obtain intervals with
        empirical (1-α) marginal coverage at station locations.

        Parameters
        ----------
        y_obs : array-like of shape (n_obs,)
            EPA point observations used for calibration.
        t_idx : array-like of shape (n_obs,)
            Time indices (rows of the model prediction grid).
        loc_idx : array-like of shape (n_obs,)
            Location indices (columns of the model prediction grid).
        alpha : float
            Miscoverage level (default 0.05 → 95% intervals).
        station_ids : array-like of shape (n_obs,), optional
            Station identifier per observation.  Required for LOSO; if None,
            falls back to a single-split calibration (all obs used together).

        Returns
        -------
        q_hat : float
            Conformal correction factor.  Multiply prediction std by this value.
        """
        self._check_fitted()

        y_obs = np.asarray(y_obs).ravel()
        t_idx = np.asarray(t_idx).ravel().astype(int)
        loc_idx = np.asarray(loc_idx).ravel().astype(int)

        pred = self.predict()
        y_pred = pred.total[t_idx, loc_idx]
        y_std = pred.std[t_idx, loc_idx]
        scores = np.abs(y_obs - y_pred) / (y_std + 1e-9)

        if station_ids is None:
            n = len(scores)
            level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            q_hat = float(np.quantile(scores, level))
            return q_hat

        station_ids = np.asarray(station_ids).ravel()
        unique_stations = np.unique(station_ids)
        q_hats = []
        for held_out in unique_stations:
            cal_mask = station_ids != held_out
            cal_scores = scores[cal_mask]
            n = len(cal_scores)
            if n < 2:
                continue
            level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            q_hats.append(np.quantile(cal_scores, level))

        if not q_hats:
            # Fallback if too few stations for LOSO
            n = len(scores)
            level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            return float(np.quantile(scores, level))

        return float(np.median(q_hats))

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
            'sigma2_obs': self._sigma2_obs,
            'beta_total': self.beta_total_,
        }
        with open(directory / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save training data
        loc_idx = (self._location_index_train
                   if self._location_index_train is not None
                   else np.array([]))
        np.savez(
            directory / "training_data.npz",
            X_train=self._X_train,
            y_train=self._y_train,
            y_matrix=self._y_matrix if self._y_matrix is not None else np.array([]),
            residual_matrix=self._residual_matrix if self._residual_matrix is not None else np.array([]),
            location_ids=self.location_ids_,
            time_ids=self.time_ids_,
            location_index_train=loc_idx,
            Z_spatial=self.Z_spatial_ if self.Z_spatial_ is not None else np.array([]),
            truncation_var=(self._truncation_var_ if self._truncation_var_ is not None
                            else np.array([])),
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
        model._y_matrix = data['y_matrix'] if data['y_matrix'].size else None
        model._residual_matrix = data['residual_matrix'] if data['residual_matrix'].size else None
        model.location_ids_ = data['location_ids']
        model.time_ids_ = data['time_ids']
        loc_idx = data['location_index_train'] if 'location_index_train' in data else np.array([])
        model._location_index_train = loc_idx if loc_idx.size else None
        model.Z_spatial_ = data['Z_spatial'] if 'Z_spatial' in data and data['Z_spatial'].size else None

        # Recompute SSM filter/smoother results using stored projected residuals
        # Restore on the projected observations (Z_spatial_ maps back to full space)
        if model.Z_spatial_ is not None and model._residual_matrix is not None:
            residual_filled = np.where(np.isnan(model._residual_matrix), 0.0, model._residual_matrix)
            k = model.Z_spatial_.shape[1]
            U, s, _ = np.linalg.svd(residual_filled, full_matrices=False)
            obs_projected = U[:, :k] * s[:k]
            model.ssm_._restore_inference(obs_projected)
        elif model._residual_matrix is not None:
            model.ssm_._restore_inference(model._residual_matrix)
        
        model.n_locations_ = metadata['n_locations']
        model.n_times_ = metadata['n_times']
        model._sigma2_obs = float(metadata.get('sigma2_obs', 0.0))
        beta_total = metadata.get('beta_total', None)
        model.beta_total_ = float(beta_total) if beta_total is not None else None
        trunc = data.get('truncation_var', np.array([]))
        model._truncation_var_ = trunc if trunc.size else None
        model.is_fitted_ = True
        
        logger.info(f"Model loaded from {directory}")
        return model
