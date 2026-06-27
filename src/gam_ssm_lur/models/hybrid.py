"""
Hybrid GAM-SSM model for spatiotemporal NO2 prediction: a GAM-LUR spatial
mean plus an SSM temporal correction.

    y(s,t) = mu(s) + alpha(t) + eps(s,t)

mu(s): spatial mean from GAM-LUR. alpha(t): latent temporal state from SSM.
eps(s,t): observation noise.

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
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gam_ssm_lur.data import (
    CalibrationResult,
    StaticData,
    TemporalData,
)
from gam_ssm_lur.models.spatial_gam import GAMSummary, SpatialGAM
from gam_ssm_lur.models.state_space import (
    SSMDiagnostics,
    StateSpaceModel,
)

if TYPE_CHECKING:
    from gam_ssm_lur.traffic_field import TrafficFieldCalibration

logger = logging.getLogger(__name__)


@dataclass
class HybridPrediction:
    """total = spatial + temporal (+ forcing), each (T, n_locations); std/lower/upper for intervals."""

    total: NDArray
    spatial: NDArray
    temporal: NDArray
    std: NDArray
    lower: NDArray
    upper: NDArray


@dataclass
class HybridSummary:
    """GAM + SSM diagnostics plus full-model RMSE/MAE/R2/95%-coverage."""

    gam_summary: GAMSummary
    ssm_diagnostics: SSMDiagnostics
    total_rmse: float
    total_mae: float
    total_r2: float
    coverage_95: float


class HybridGAMSSM:
    """GAM-LUR spatial model + SSM temporal model, combined additively with uncertainty propagation."""

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
        # max spectral radius for the SSM's dynamic-block T (Hamilton, 1994,
        # Ch. 1); lower -> more aggressive drift suppression over the window
        self.max_eigenvalue = max_eigenvalue
        self.random_state = random_state

        self.gam_: Optional[SpatialGAM] = None
        self.ssm_: Optional[StateSpaceModel] = None

        self.n_locations_: Optional[int] = None
        self.n_times_: Optional[int] = None
        self.location_ids_: Optional[NDArray] = None
        self.time_ids_: Optional[NDArray] = None

        self._X_train: Optional[NDArray] = None
        self._y_train: Optional[NDArray] = None
        self._y_matrix: Optional[NDArray] = None  # reshaped (T, n_locations)
        self._residual_matrix: Optional[NDArray] = (
            None  # calibrated satellite field reshaped
        )
        self._location_index_train: Optional[NDArray] = None
        self._time_index_train: Optional[NDArray] = None

        self.Z_spatial_: Optional[NDArray] = (
            None  # maps (state_dim,) temporal state -> (n_locations,)
        )

        # per-location variance NOT captured by the top state_dim SVD factors
        self._truncation_var_: Optional[NDArray] = None

        # variance of calibration residuals (satellite vs surface EPA) -- the
        # measurement representativeness error when predicting surface NO2
        self._sigma2_obs: float = 0.0
        self._calibration = None  # CalibrationResult, set by fit_from_dataset()

        # optional spatially-resolved traffic correction (T, n_locations),
        # calibrated against EPA residuals (traffic_field.calibrate_traffic_field).
        # Kept separate from the satellite-driven SSM since that's bottlenecked
        # to k SVD factors and can't represent road-network-scale detail.
        self._traffic_field: Optional[NDArray] = None
        self._traffic_calibration = None

        # true GAM coefficient = beta_init (pooled OLS) + jointly-estimated
        # beta_delta; used by predict() (see fit_from_dataset)
        self.beta_total_: Optional[float] = None

        self.is_fitted_ = False

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Union[NDArray, pd.Series],
        time_index: Union[NDArray, pd.Series],
        location_index: Optional[Union[NDArray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> HybridGAMSSM:
        """Fit GAM-LUR on (X, y), then SSM on the GAM residuals' SVD scores.
        location_index optional if y is already (T, n_locations)."""
        logger.info("Fitting Hybrid GAM-SSM model")

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

        self._X_train = X
        self._y_train = y
        self._time_index_train = time_index.copy() if time_index is not None else None
        self._location_index_train = (
            location_index.copy() if location_index is not None else None
        )

        if location_index is not None:
            if isinstance(location_index, pd.Series):
                location_index = location_index.values
            location_index = np.asarray(location_index)

            self._y_matrix, self.location_ids_, self.time_ids_ = (
                self._reshape_to_matrix(y, time_index, location_index)
            )
        else:
            if y.ndim == 1:
                raise ValueError(
                    "If location_index is not provided, y must be 2D (T, n_locations)"
                )
            self._y_matrix = y
            self.time_ids_ = np.unique(time_index)
            self.location_ids_ = np.arange(y.shape[1])

        self.n_times_, self.n_locations_ = self._y_matrix.shape

        logger.info(
            f"Data shape: {self.n_times_} time steps x {self.n_locations_} locations"
        )

        logger.info("Step 1: Fitting GAM spatial component")
        self.gam_ = SpatialGAM(n_splines=self.n_splines, lam=self.gam_lam)
        self.gam_.fit(X, y, feature_names=feature_names)

        gam_residuals = self.gam_.get_residuals(X, y)

        if location_index is not None:
            residual_matrix, _, _ = self._reshape_to_matrix(
                gam_residuals, time_index, location_index
            )
        else:
            residual_matrix = gam_residuals.reshape(self.n_times_, self.n_locations_)

        logger.info(f"GAM fitted. R2={self.gam_.summary().r_squared:.4f}")
        logger.info(f"Residual matrix shape: {residual_matrix.shape}")
        self._residual_matrix = residual_matrix

        logger.info("Step 2: Fitting SSM on GAM residuals")
        residual_filled = np.where(np.isnan(residual_matrix), 0.0, residual_matrix)
        U, s, Vt = np.linalg.svd(residual_filled, full_matrices=False)
        k = min(self.state_dim, len(s))
        obs_projected = U[:, :k] * s[:k]  # (T, k)
        self.Z_spatial_ = Vt[:k, :].T  # (n_locations, k)

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

        predictions = self.predict()
        metrics = self.evaluate(self._y_matrix.flatten(), predictions.total.flatten())
        logger.info(
            f"Hybrid model fitted. "
            f"RMSE={metrics['rmse']:.4f}, "
            f"R2={metrics['r2']:.4f}, "
            f"Coverage={metrics['coverage_95']:.1%}"
        )

        return self

    def fit_from_dataset(
        self,
        static: StaticData,
        temporal: TemporalData,
        calibration: Optional[CalibrationResult] = None,
        feature_selector: Optional[object] = None,
        id_cols: Optional[List[str]] = None,
        target_col: str = "atmos_no2",
    ) -> HybridGAMSSM:
        """High-level entry point for the standard Dublin-style data contract:
        static GAM-LUR fit, build forcing matrix from activity/met data, then
        SSM fit on the calibrated dense (satellite) field with that forcing.
        calibration (SpatiotemporalDataset.calibrate_dense_obs output), if
        None, leaves dense observations uncalibrated. id_cols defaults to
        ["grid_id", "latitude", "longitude"].
        """

        if id_cols is None:
            id_cols = ["grid_id", "latitude", "longitude"]

        # Prepare feature matrix
        # ----------------------
        feat_cols = [c for c in static.features.columns if c not in id_cols]
        X_df = static.features[feat_cols]

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

        # Build forcing matrix (T, n_forcing)
        # ------------------------------------
        dates = temporal.dates
        T = len(dates)

        act_map = dict(
            zip(
                temporal.activity_forcing["date"],
                temporal.activity_forcing["delta_activity"],
            )
        )
        activity_vec = np.array([act_map.get(d, 0.0) for d in dates])  # (T,)

        # met forcing aggregated to city mean for the SSM (cell-level spatial
        # variation is captured by B_ via the joint EM, not here)
        if "grid_id" in temporal.met_forcing.columns:
            met_pivot = temporal.met_forcing.pivot(
                index="date", columns="grid_id", values="met_forcing"
            )
            met_mean = met_pivot.reindex(dates).mean(axis=1).fillna(0.0).values  # (T,)
        else:
            met_map = temporal.met_forcing.set_index("date")["met_forcing"].to_dict()
            met_mean = np.array([met_map.get(d, 0.0) for d in dates])  # (T,)

        # leading all-ones column gives a jointly-estimated intercept -- its
        # B_tilde column absorbs any systematic scale gap between the
        # calibrated satellite field and the GAM baseline (e.g. the ~3.5
        # ug/m3 GAM-vs-EPA bias) via the same augmented EM as traffic/wind,
        # rather than a separately pre-computed mean offset
        forcing_matrix = np.column_stack([np.ones(T), activity_vec, met_mean])  # (T, 3)

        # satellite-to-surface measurement uncertainty, included in prediction
        # intervals so they reflect point-location (EPA station) uncertainty
        self._sigma2_obs = (
            float(calibration.sigma2_obs) if calibration is not None else 0.0
        )

        self._calibration = calibration
        self._temporal = temporal
        self._static = static
        self._dates = dates

        logger.info("Fitting GAM spatial component")
        self.gam_ = SpatialGAM(n_splines=self.n_splines, lam=self.gam_lam)
        self.gam_.fit(X, y, feature_names=feature_names)

        # residuals are per-cell (not time-varying): the full spatiotemporal
        # residual matrix is built from dense obs below
        self._X_train = X
        self._y_train = y
        self.n_locations_ = len(y)
        self.n_times_ = T
        self.location_ids_ = np.array(static.grid_ids)
        self.time_ids_ = np.array(dates)
        self._y_matrix = None
        self._residual_matrix = None

        lur_prior = self.gam_.predict(X)
        grid_id_to_idx = {gid: i for i, gid in enumerate(static.grid_ids)}
        lur_prior_arr = lur_prior  # (n_locations,)

        dense = temporal.dense_obs.copy()
        if calibration is not None:
            dense["obs_dense"] = calibration.apply(dense["obs_dense"].values)

        dense["loc_idx"] = dense["grid_id"].map(grid_id_to_idx)
        dense = dense.dropna(subset=["loc_idx"])
        dense["loc_idx"] = dense["loc_idx"].astype(int)

        date_to_tidx = {d: i for i, d in enumerate(dates)}
        dense["t_idx"] = dense["date"].map(date_to_tidx)
        dense = dense.dropna(subset=["t_idx"])
        dense["t_idx"] = dense["t_idx"].astype(int)

        # Initial GAM coefficient via pooled OLS (y ~ beta_init * GAM + intercept),
        # used ONLY to pick a sensible matrix for the loadings SVD below -- NOT
        # assumed fixed. Raw y (implicit beta=0) or y-GAM (implicit beta=1, the
        # original bug) both make the loadings internally inconsistent with
        # whatever beta the regression step later estimates; this two-step
        # estimator (Doz, Giannone & Reichlin, 2011) needs a sensible initial
        # matrix, not an arbitrary extreme. Fit WITH an intercept: through-origin
        # would badly bias the slope given the calibration step's own intercept
        # (~6 in this dataset) -- y isn't expected to be ~0 when GAM is ~0.
        gam_at_obs = lur_prior_arr[dense["loc_idx"].values]
        design = np.column_stack([np.ones(len(gam_at_obs)), gam_at_obs])
        ols_coeffs, _, _, _ = np.linalg.lstsq(
            design, dense["obs_dense"].values, rcond=None
        )
        beta0_init, beta_init = float(ols_coeffs[0]), float(ols_coeffs[1])
        logger.info(
            "Initial GAM relationship (pooled OLS with intercept, for SVD basis only): "
            "y = %.4f + %.4f * GAM",
            beta0_init,
            beta_init,
        )

        dense["residual"] = dense["obs_dense"].values - beta_init * gam_at_obs
        y_matrix = np.full((T, self.n_locations_), np.nan)
        y_matrix[dense["t_idx"].values, dense["loc_idx"].values] = dense[
            "residual"
        ].values
        self._residual_matrix = y_matrix

        # project the deviation-from-(beta_init*GAM) field to low-dimensional
        # spatial loadings; NaN cells (missing satellite obs) filled with 0
        state_dim = self.state_dim
        y_filled = np.where(np.isnan(y_matrix), 0.0, y_matrix)
        U, s, Vt = np.linalg.svd(y_filled, full_matrices=False)
        k = min(state_dim, len(s))
        scores = U[:, :k] * s[:k]  # (T, k) -- temporal scores of the deviation field
        self.Z_spatial_ = Vt[:k, :].T  # (n_locations, k) -- spatial loadings (Lambda)
        var_explained = 100 * (s[:k] ** 2).sum() / (s**2).sum()
        logger.info(
            "Deviation-field SVD: top %d factors explain %.1f%% of variance -> SSM obs_dim=%d",
            k,
            var_explained,
            k,
        )

        # Project the GAM baseline into score-space: g_tilde = Lambda' @ GAM_prior.
        # Lambda's columns are orthonormal (right singular vectors), so this
        # is also Lambda's pseudo-inverse projection. The EM jointly estimates
        # beta_delta (residual GAM coefficient beyond beta_init) alongside the
        # forcing effects and dynamics; true GAM coefficient = beta_init +
        # beta_delta (see self.beta_total_).
        g_tilde = self.Z_spatial_.T @ lur_prior_arr  # (k,)

        # per-location variance NOT captured by the top k SVD factors --
        # included in prediction intervals so spatial patterns outside the
        # low-dimensional subspace contribute to uncertainty
        y_reconstructed = (U[:, :k] * s[:k]) @ Vt[:k, :]  # (T, n_locations)
        truncation_error = y_filled - y_reconstructed  # (T, n_locations)
        valid_mask = ~np.isnan(y_matrix)
        self._truncation_var_ = np.where(
            valid_mask.sum(axis=0) > 1,
            np.nanvar(np.where(valid_mask, truncation_error, np.nan), axis=0),
            0.0,
        )
        logger.info(
            "Truncation variance: mean=%.3f, median=%.3f ug/m3 (added to prediction std)",
            self._truncation_var_.mean(),
            np.median(self._truncation_var_),
        )

        logger.info(
            "Fitting SSM with jointly-estimated GAM coefficient and forcing effects"
        )
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

        # true GAM coefficient = initial pooled-OLS estimate (used only to
        # choose the SVD basis above) + the jointly-estimated residual correction
        self.beta_total_ = beta_init + self.ssm_.beta_
        logger.info(
            "GAM coefficient: beta_init=%.4f + beta_delta=%.4f = beta_total=%.4f "
            "(old buggy design assumed 1.0)",
            beta_init,
            self.ssm_.beta_,
            self.beta_total_,
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
        """Flat (values, time_index, location_index) -> (T, n_locations) matrix, NaN-filled for missing,
        then linearly interpolated per location."""
        unique_times = np.unique(time_index)
        unique_locs = np.unique(location_index)

        n_times = len(unique_times)
        n_locs = len(unique_locs)

        time_map = {t: i for i, t in enumerate(unique_times)}
        loc_map = {loc: i for i, loc in enumerate(unique_locs)}

        matrix = np.full((n_times, n_locs), np.nan)

        for val, t, loc in zip(values, time_index, location_index):
            matrix[time_map[t], loc_map[loc]] = val

        for j in range(n_locs):
            col = matrix[:, j]
            if np.any(np.isnan(col)):
                valid = ~np.isnan(col)
                if np.sum(valid) > 1:
                    matrix[:, j] = np.interp(
                        np.arange(n_times), np.where(valid)[0], col[valid]
                    )
                elif np.sum(valid) == 1:
                    matrix[:, j] = col[valid][0]
                else:
                    matrix[:, j] = 0.0

        return matrix, unique_locs, unique_times

    def predict(
        self, X: Optional[Union[NDArray, pd.DataFrame]] = None
    ) -> HybridPrediction:
        """total = beta_total*GAM + SSM temporal deviation + forcing term (+ traffic correction),
        with combined uncertainty. Uses training data/grid if X is None."""
        self._check_fitted()

        if X is None:
            if self._location_index_train is not None:
                loc_to_features = {}
                for i, loc in enumerate(self._location_index_train):
                    if loc not in loc_to_features:
                        loc_to_features[loc] = self._X_train[i]

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

        gam_pred, gam_std = self.gam_.predict(X, return_std=True)

        # fit_from_dataset provides static X (n_locations,) -- tile across time
        if gam_pred.shape[0] == self.n_locations_:
            spatial_matrix = np.tile(gam_pred, (self.n_times_, 1))
            gam_std_tiled = np.tile(gam_std, (self.n_times_, 1))
        else:
            spatial_matrix = gam_pred.reshape(self.n_times_, self.n_locations_)
            gam_std_tiled = gam_std.reshape(self.n_times_, self.n_locations_)

        # ssm_.predict() returns only the dynamic (alpha) sub-block -- beta and
        # B_tilde are separate, jointly-estimated fixed-effect states, not
        # folded into alpha_t
        ssm_pred = self.ssm_.predict(confidence_level=self.confidence_level)
        temporal_matrix = ssm_pred.mean @ self.Z_spatial_.T  # (T, n_locations)
        temporal_std = np.sqrt((ssm_pred.std**2) @ (self.Z_spatial_**2).T)

        # forcing contribution (traffic anomaly, wind): B_tilde @ u_t in score
        # space, mapped to all locations via Z_spatial_. beta*GAM is
        # intentionally NOT added here -- g_tilde only separated genuine
        # satellite dynamics from the GAM-correlated part of the signal; the
        # GAM's own contribution is spatial_matrix above at its native scale.
        if self.ssm_.B_ is not None and self.ssm_._forcing_matrix is not None:
            forcing_scores = self.ssm_._forcing_matrix @ self.ssm_.B_.T  # (T, k)
            forcing_matrix_term = forcing_scores @ self.Z_spatial_.T  # (T, n_locations)
        else:
            forcing_matrix_term = 0.0

        # beta_total (fit_from_dataset) is the model's own jointly-estimated
        # GAM-to-observation relationship -- using 1.0 (the original buggy
        # design) would be inconsistent with what the SSM actually estimates
        beta_total = self.beta_total_ if self.beta_total_ is not None else 1.0
        total = beta_total * spatial_matrix + temporal_matrix + forcing_matrix_term

        # optional spatially-resolved traffic correction, added independently
        # of the satellite-driven SSM term so it can carry road-network-scale
        # spatial detail the k-factor SVD bottleneck discards
        if self._traffic_field is not None and self._traffic_calibration is not None:
            traffic_term = self._traffic_calibration.apply(self._traffic_field)
            total = total + np.nan_to_num(traffic_term, nan=0.0)

        # H is estimated in k-dim projected space; propagate through Z_spatial_
        # to get the per-location noise contribution
        if self.ssm_.H_ is not None:
            h_diag = (
                np.diag(self.ssm_.H_)
                if self.ssm_.H_.ndim == 2
                else self.ssm_.H_.ravel()
            )
            obs_noise_var = h_diag @ (self.Z_spatial_**2).T  # (n_locations,)
        else:
            obs_noise_var = np.zeros(self.n_locations_)

        trunc_var = (
            self._truncation_var_
            if self._truncation_var_ is not None
            else np.zeros(self.n_locations_)
        )

        # independence assumption across components. sigma2_obs (satellite-to-
        # surface mismatch) is intentionally excluded here -- it absorbs all
        # TROPOMI-EPA correlation noise and would produce uninformatively wide
        # intervals (~78 ug/m3 width for r~0.2); use calibrate_intervals_conformal()
        # for properly calibrated prediction bands.
        combined_std = np.sqrt(
            gam_std_tiled**2
            + temporal_std**2
            + obs_noise_var[np.newaxis, :]
            + trunc_var[np.newaxis, :]
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
        """Remap the internal (time, location) grid prediction back to the original observation order."""
        self._check_fitted()

        if isinstance(time_index, pd.Series):
            time_index = time_index.values
        if isinstance(location_index, pd.Series):
            location_index = location_index.values

        time_index = np.asarray(time_index)
        location_index = np.asarray(location_index)

        if time_index.shape[0] != location_index.shape[0]:
            raise ValueError("time_index and location_index must have the same length")

        full_pred = self.predict(X).total  # (T, n_locations)

        time_map = {t: i for i, t in enumerate(self.time_ids_)}
        loc_map = {loc: j for j, loc in enumerate(self.location_ids_)}

        try:
            aligned = np.array(
                [
                    full_pred[time_map[t], loc_map[loc]]
                    for t, loc in zip(time_index, location_index)
                ]
            )
        except KeyError as exc:
            missing_key = exc.args[0]
            raise KeyError(
                f"Index '{missing_key}' not found in fitted model time/location IDs. "
                "Ensure the provided indices match those used in fit()."
            ) from exc

        return aligned

    def forecast(
        self, X_future: Union[NDArray, pd.DataFrame], n_steps: int
    ) -> HybridPrediction:
        """Forecast n_steps ahead. No forcing/beta*GAM term: ssm_.forecast() extrapolates
        only the dynamic (alpha) sub-block since future traffic/wind values aren't available here."""
        self._check_fitted()

        if isinstance(X_future, pd.DataFrame):
            X_future = X_future.values
        X_future = np.asarray(X_future)

        gam_pred, gam_std = self.gam_.predict(X_future, return_std=True)
        spatial_matrix = gam_pred.reshape(n_steps, self.n_locations_)
        gam_std_matrix = gam_std.reshape(n_steps, self.n_locations_)

        ssm_forecast = self.ssm_.forecast(
            n_steps, confidence_level=self.confidence_level
        )
        temporal_matrix = ssm_forecast.mean @ self.Z_spatial_.T
        temporal_std = np.sqrt((ssm_forecast.std**2) @ (self.Z_spatial_**2).T)

        total = spatial_matrix + temporal_matrix
        combined_std = np.sqrt(gam_std_matrix**2 + temporal_std**2)

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
        """RMSE/MAE/MBE/R2/correlation, plus coverage/CRPS if intervals given (or computed via predict())."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        residuals = y_true - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        mbe = np.mean(residuals)  # positive = under-prediction

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot

        corr = np.corrcoef(y_true, y_pred)[0, 1]

        metrics = {"rmse": rmse, "mae": mae, "mbe": mbe, "r2": r2, "correlation": corr}

        if y_lower is not None and y_upper is not None:
            from scipy import stats

            y_lower = np.asarray(y_lower).flatten()
            y_upper = np.asarray(y_upper).flatten()

            coverage_95 = np.mean((y_true >= y_lower) & (y_true <= y_upper))
            interval_width = np.mean(y_upper - y_lower)
            metrics["coverage_95"] = coverage_95
            metrics["interval_width"] = interval_width

            y_std = (y_upper - y_lower) / (
                2 * 1.96
            )  # reconstruct std from 95% interval
            z_90 = stats.norm.ppf(0.95)
            lower_90 = y_pred - z_90 * y_std
            upper_90 = y_pred + z_90 * y_std
            coverage_90 = np.mean((y_true >= lower_90) & (y_true <= upper_90))
            metrics["coverage_90"] = coverage_90

            z = (y_true - y_pred) / y_std
            crps = np.mean(
                y_std
                * (
                    z * (2 * stats.norm.cdf(z) - 1)
                    + 2 * stats.norm.pdf(z)
                    - 1 / np.sqrt(np.pi)
                )
            )
            metrics["crps"] = crps
        else:
            from scipy import stats

            predictions = self.predict()
            y_lower = predictions.lower.flatten()
            y_upper = predictions.upper.flatten()
            y_true_full = self._y_matrix.flatten()
            y_pred_full = predictions.total.flatten()

            coverage_95 = np.mean((y_true_full >= y_lower) & (y_true_full <= y_upper))
            metrics["coverage_95"] = coverage_95

            y_std = (y_upper - y_lower) / (2 * 1.96)
            z_90 = stats.norm.ppf(0.95)
            lower_90 = y_pred_full - z_90 * y_std
            upper_90 = y_pred_full + z_90 * y_std
            coverage_90 = np.mean((y_true_full >= lower_90) & (y_true_full <= upper_90))
            metrics["coverage_90"] = coverage_90

            z = (y_true_full - y_pred_full) / y_std
            crps = np.mean(
                y_std
                * (
                    z * (2 * stats.norm.cdf(z) - 1)
                    + 2 * stats.norm.pdf(z)
                    - 1 / np.sqrt(np.pi)
                )
            )
            metrics["crps"] = crps

        return metrics

    def evaluate_in_observation_order(
        self,
        y_true: NDArray,
        time_index: Union[NDArray, pd.Series],
        location_index: Union[NDArray, pd.Series],
        X: Optional[Union[NDArray, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """evaluate() using predictions aligned to observation order (when input isn't already grid-ordered)."""
        aligned_pred = self.predict_in_observation_order(
            time_index=time_index, location_index=location_index, X=X
        )
        return self.evaluate(y_true=y_true, y_pred=aligned_pred)

    def add_traffic_correction(
        self,
        traffic_field: NDArray,
        epa_eval: pd.DataFrame,
        y_col: str = "obs_value",
    ) -> TrafficFieldCalibration:
        """Fit and attach a spatially-resolved traffic correction. Must be called after fit()/
        fit_from_dataset(). Calibrates traffic_field (T, n_locations), same order as self.time_ids_/
        location_ids_, against the EPA residual the current GAM+SSM prediction doesn't explain --
        kept outside the k-factor SVD bottleneck so it can carry road-network-scale detail.
        """
        from gam_ssm_lur.traffic_field import calibrate_traffic_field

        self._check_fitted()
        baseline_pred = (
            self.predict().total
        )  # GAM + satellite-SSM only (traffic not yet attached)
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
        """Leave-one-station-out split conformal calibration: per held-out station, calibrate
        on the rest, take the (1-alpha) normalised-residual quantile, return the median q_hat
        across folds. Multiply predict()'s std by q_hat for empirical (1-alpha) coverage at
        station locations. Without station_ids, falls back to a single-split calibration.
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
            n = len(scores)
            level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            return float(np.quantile(scores, level))

        return float(np.median(q_hats))

    def summary(self) -> HybridSummary:
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
            total_rmse=metrics["rmse"],
            total_mae=metrics["mae"],
            total_r2=metrics["r2"],
            coverage_95=metrics["coverage_95"],
        )

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        return self.gam_.get_feature_importance()

    def get_smoothed_states(self) -> NDArray:
        self._check_fitted()
        return self.ssm_.smoother_result_.smoothed_means

    def get_em_convergence(self) -> pd.DataFrame:
        self._check_fitted()
        return self.ssm_.get_em_history()

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def save(self, directory: Union[str, Path]) -> None:
        self._check_fitted()

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.gam_.save(directory / "gam.pkl")
        self.ssm_.save(directory / "ssm.pkl")

        import json

        metadata = {
            "n_splines": self.n_splines,
            "gam_lam": self.gam_lam if isinstance(self.gam_lam, float) else "auto",
            "state_dim": self.state_dim,
            "em_max_iter": self.em_max_iter,
            "em_tol": self.em_tol,
            "scalability_mode": self.scalability_mode,
            "regularization": self.regularization,
            "confidence_level": self.confidence_level,
            "n_locations": self.n_locations_,
            "n_times": self.n_times_,
            "sigma2_obs": self._sigma2_obs,
            "beta_total": self.beta_total_,
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        loc_idx = (
            self._location_index_train
            if self._location_index_train is not None
            else np.array([])
        )
        np.savez(
            directory / "training_data.npz",
            X_train=self._X_train,
            y_train=self._y_train,
            y_matrix=self._y_matrix if self._y_matrix is not None else np.array([]),
            residual_matrix=self._residual_matrix
            if self._residual_matrix is not None
            else np.array([]),
            location_ids=self.location_ids_,
            time_ids=self.time_ids_,
            location_index_train=loc_idx,
            Z_spatial=self.Z_spatial_ if self.Z_spatial_ is not None else np.array([]),
            truncation_var=self._truncation_var_
            if self._truncation_var_ is not None
            else np.array([]),
        )

        logger.info(f"Model saved to {directory}")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> HybridGAMSSM:
        import json

        directory = Path(directory)

        with open(directory / "metadata.json") as f:
            metadata = json.load(f)

        model = cls(
            n_splines=metadata["n_splines"],
            gam_lam=metadata["gam_lam"],
            state_dim=metadata["state_dim"],
            em_max_iter=metadata["em_max_iter"],
            em_tol=metadata["em_tol"],
            scalability_mode=metadata["scalability_mode"],
            regularization=metadata["regularization"],
            confidence_level=metadata["confidence_level"],
        )

        model.gam_ = SpatialGAM.load(directory / "gam.pkl")
        model.ssm_ = StateSpaceModel.load(directory / "ssm.pkl")

        data = np.load(directory / "training_data.npz", allow_pickle=True)
        model._X_train = data["X_train"]
        model._y_train = data["y_train"]
        model._y_matrix = data["y_matrix"] if data["y_matrix"].size else None
        model._residual_matrix = (
            data["residual_matrix"] if data["residual_matrix"].size else None
        )
        model.location_ids_ = data["location_ids"]
        model.time_ids_ = data["time_ids"]
        loc_idx = (
            data["location_index_train"]
            if "location_index_train" in data
            else np.array([])
        )
        model._location_index_train = loc_idx if loc_idx.size else None
        model.Z_spatial_ = (
            data["Z_spatial"]
            if "Z_spatial" in data and data["Z_spatial"].size
            else None
        )

        # recompute SSM filter/smoother on the projected observations
        # (Z_spatial_ maps back to full space)
        if model.Z_spatial_ is not None and model._residual_matrix is not None:
            residual_filled = np.where(
                np.isnan(model._residual_matrix), 0.0, model._residual_matrix
            )
            k = model.Z_spatial_.shape[1]
            U, s, _ = np.linalg.svd(residual_filled, full_matrices=False)
            obs_projected = U[:, :k] * s[:k]
            model.ssm_._restore_inference(obs_projected)
        elif model._residual_matrix is not None:
            model.ssm_._restore_inference(model._residual_matrix)

        model.n_locations_ = metadata["n_locations"]
        model.n_times_ = metadata["n_times"]
        model._sigma2_obs = float(metadata.get("sigma2_obs", 0.0))
        beta_total = metadata.get("beta_total", None)
        model.beta_total_ = float(beta_total) if beta_total is not None else None
        trunc = data.get("truncation_var", np.array([]))
        model._truncation_var_ = trunc if trunc.size else None
        model.is_fitted_ = True

        logger.info(f"Model loaded from {directory}")
        return model
