"""Utility helpers for array/matrix operations."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


def ensure_array(data: Union[NDArray, pd.DataFrame, pd.Series]) -> NDArray:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    return np.asarray(data)


def extract_feature_names(
    X: Union[NDArray, pd.DataFrame],
    provided_names: Optional[List[str]] = None,
) -> Tuple[NDArray, List[str]]:
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_array = X.values
    else:
        X_array = np.asarray(X)
        if provided_names is not None:
            feature_names = provided_names
        else:
            feature_names = [f"x{i}" for i in range(X_array.shape[1])]
    return X_array, feature_names


def compute_prediction_intervals(
    mean: NDArray,
    std: NDArray,
    confidence_level: float = 0.95,
) -> Tuple[NDArray, NDArray]:
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    lower = mean - z_score * std
    upper = mean + z_score * std
    return lower, upper


def extract_diagonal(
    cov: NDArray,
    return_format: str = "vector",
) -> NDArray:
    if cov.ndim == 1:
        if return_format == "matrix":
            return np.diag(cov)
        return cov
    elif cov.ndim == 2:
        diag = np.diag(cov)
        if return_format == "matrix":
            return np.diag(diag)
        return diag
    elif cov.ndim == 3:
        diag = np.diagonal(cov, axis1=1, axis2=2)
        if return_format == "matrix":
            n_steps, n_dim = diag.shape
            result = np.zeros_like(cov)
            for i in range(n_steps):
                result[i] = np.diag(diag[i])
            return result
        return diag
    else:
        raise ValueError(f"Unsupported covariance dimension: {cov.ndim}")


def ensure_diagonal_matrix(P: NDArray) -> NDArray:
    if P.ndim == 1:
        return np.diag(P)
    return P


def compute_aic(log_likelihood: float, n_params: int) -> float:
    return 2 * n_params - 2 * log_likelihood


def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    return n_params * np.log(n_obs) - 2 * log_likelihood


def compute_r_squared(y_true: NDArray, y_pred: NDArray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def reshape_flat_to_matrix(
    flat_data: NDArray,
    n_times: int,
    n_locations: int,
) -> NDArray:
    if flat_data.size != n_times * n_locations:
        raise ValueError(
            f"Cannot reshape array of size {flat_data.size} into "
            f"shape ({n_times}, {n_locations})"
        )
    return flat_data.reshape(n_times, n_locations)


def reshape_matrix_to_flat(matrix: NDArray) -> NDArray:
    return matrix.flatten()
