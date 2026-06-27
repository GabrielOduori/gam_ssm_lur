"""Spatially-resolved traffic forcing for the SSM temporal component.

The existing SSM forcing (``data.py: _load_activity_forcing``) collapses all
SCATS detectors to a single city-wide scalar per day, which the Kalman filter
applies uniformly across all grid cells via the B matrix. That scalar cannot
make the temporal correction follow the road network, because all spatial
information in the detector network is discarded before it reaches the model.

This module instead builds a per-cell, per-day traffic intensity field by
inverse-distance weighting each grid cell to its nearest SCATS detectors,
using the same overpass-window aggregation as the rest of the pipeline. The
resulting (T, n_locations) field is calibrated against EPA residuals (the
portion of EPA NO2 variance not already explained by GAM + the satellite-
driven SSM correction) before being added to the model's prediction — kept
as an independent additive term, not fed through the existing k=3 SVD
bottleneck, since that bottleneck is what discards spatial detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

DEG_TO_M = 111_000.0  # approx metres per degree latitude, fine for Dublin's extent


@dataclass
class TrafficFieldCalibration:
    """OLS calibration of the per-cell traffic field against EPA residuals.

    Fits: ``epa_residual = beta0 + beta1 * traffic_intensity``, where
    epa_residual is the EPA observation minus the GAM+SSM prediction
    (i.e. what the satellite-driven model failed to explain).
    """
    beta0: float
    beta1: float
    n_obs: int
    r: float

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return self.beta0 + self.beta1 * np.asarray(raw)


def build_cell_detector_weights(
    cell_lat: np.ndarray,
    cell_lon: np.ndarray,
    det_lat: np.ndarray,
    det_lon: np.ndarray,
    k: int = 5,
    p: float = 1.5,
    dist_floor_m: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build inverse-distance weights from each grid cell to its k nearest detectors.

    Geometry only — time-invariant, computed once and reused for every day.

    Returns
    -------
    idxs : (n_cells, k) int array
        Index of each of the k nearest detectors, into the detector array.
    weights : (n_cells, k) float array
        Normalised inverse-distance weights (rows sum to 1).
    """
    tree = cKDTree(np.column_stack([det_lat, det_lon]))
    dists_deg, idxs = tree.query(np.column_stack([cell_lat, cell_lon]), k=k)
    dists_m = dists_deg * DEG_TO_M
    weights = 1.0 / np.power(np.maximum(dists_m, dist_floor_m), p)
    weights /= weights.sum(axis=1, keepdims=True)
    return idxs, weights


def compute_traffic_field(
    traffic_raw: pd.DataFrame,
    dates: List,
    idxs: np.ndarray,
    weights: np.ndarray,
    detector_site_ids: np.ndarray,
    overpass_window_start: int = 11,
    overpass_window_end: int = 14,
    site_col: str = "site_id",
    timestamp_col: str = "traffic_end_time",
    value_col: str = "traffic_volume",
) -> np.ndarray:
    """Build the (T, n_cells) per-cell traffic intensity field.

    For each day, the overpass-window mean volume at each detector is
    distance-weighted onto every grid cell using the precomputed
    ``idxs``/``weights`` from :func:`build_cell_detector_weights`.

    Parameters
    ----------
    traffic_raw : pd.DataFrame
        Raw (unaggregated) traffic time series with detector id, timestamp,
        and volume columns.
    dates : list
        Ordered list of T dates matching the model's time index.
    idxs, weights : arrays from build_cell_detector_weights
    detector_site_ids : array
        Detector site_id values in the same order used to build idxs/weights.

    Returns
    -------
    field : (T, n_cells) ndarray
        NaN where no detector data is available for that day.
    """
    df = traffic_raw.dropna(subset=[timestamp_col, value_col]).copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["hour"] = df[timestamp_col].dt.hour
    df["date"] = df[timestamp_col].dt.date
    win = df[(df["hour"] >= overpass_window_start) & (df["hour"] <= overpass_window_end)]
    daily = win.groupby([site_col, "date"])[value_col].mean()

    n_cells = idxs.shape[0]
    T = len(dates)
    field = np.full((T, n_cells), np.nan)

    site_to_pos = {sid: i for i, sid in enumerate(detector_site_ids)}

    for t, d in enumerate(dates):
        try:
            day_vals = daily.xs(d, level="date")
        except KeyError:
            logger.warning("No traffic data for %s; leaving field row as NaN", d)
            continue
        vol_by_pos = np.full(len(detector_site_ids), np.nan)
        for sid, v in day_vals.items():
            pos = site_to_pos.get(sid)
            if pos is not None:
                vol_by_pos[pos] = v

        vol_neighbors = vol_by_pos[idxs]  # (n_cells, k)
        valid = ~np.isnan(vol_neighbors)
        # renormalise weights over only the valid (non-NaN) neighbours per cell
        w = np.where(valid, weights, 0.0)
        w_sum = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            field[t, :] = np.where(
                w_sum > 0,
                np.nansum(np.where(valid, vol_neighbors * w, 0.0), axis=1) / np.where(w_sum > 0, w_sum, 1.0),
                np.nan,
            )

    logger.info(
        "Traffic field built: shape=%s, %.1f%% cells with data per day (median)",
        field.shape, 100 * np.median((~np.isnan(field)).mean(axis=1)),
    )
    return field


def calibrate_traffic_field(
    traffic_field: np.ndarray,
    gam_ssm_pred: np.ndarray,
    epa_eval: pd.DataFrame,
    y_col: str = "obs_value",
    min_obs: int = 10,
) -> TrafficFieldCalibration:
    """Fit traffic field calibration against EPA residuals.

    Parameters
    ----------
    traffic_field : (T, n_cells)
        Output of compute_traffic_field.
    gam_ssm_pred : (T, n_cells)
        Existing model's prediction (GAM + satellite-driven SSM), same shape.
    epa_eval : pd.DataFrame
        Must contain ``t_idx``, ``loc_idx``, and ``y_col`` columns (as used
        elsewhere in the pipeline for EPA validation alignment).
    """
    t_idx = epa_eval["t_idx"].values
    loc_idx = epa_eval["loc_idx"].values
    y_obs = epa_eval[y_col].values
    pred = gam_ssm_pred[t_idx, loc_idx]
    traffic_vals = traffic_field[t_idx, loc_idx]

    mask = ~(np.isnan(traffic_vals) | np.isnan(pred) | np.isnan(y_obs))
    n = int(mask.sum())
    if n < min_obs:
        logger.warning(
            "Only %d valid traffic-EPA pairs (need %d); using null calibration (beta1=0).", n, min_obs,
        )
        return TrafficFieldCalibration(beta0=0.0, beta1=0.0, n_obs=n, r=float("nan"))

    residual = y_obs[mask] - pred[mask]
    x = traffic_vals[mask]
    X = np.column_stack([np.ones(n), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, residual, rcond=None)
    beta0, beta1 = float(coeffs[0]), float(coeffs[1])
    r = float(np.corrcoef(residual, x)[0, 1]) if n > 1 else float("nan")

    logger.info(
        "Traffic field calibration: N=%d  residual = %.4f + %.6f * traffic  r=%.4f",
        n, beta0, beta1, r,
    )
    return TrafficFieldCalibration(beta0=beta0, beta1=beta1, n_obs=n, r=r)
