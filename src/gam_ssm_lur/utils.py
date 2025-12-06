"""Utility functions for GAM-SSM-LUR models."""

from __future__ import annotations

from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


def ensure_array(data: Union[NDArray, pd.DataFrame, pd.Series]) -> NDArray:
    """Convert pandas DataFrame/Series or array-like to numpy array.

    Parameters
    ----------
    data : array-like, DataFrame, or Series
        Input data

    Returns
    -------
    NDArray
        Numpy array
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    return np.asarray(data)


def extract_feature_names(
    X: Union[NDArray, pd.DataFrame],
    provided_names: Optional[List[str]] = None,
) -> Tuple[NDArray, List[str]]:
    """Extract feature names from DataFrame or use provided names.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix
    provided_names : list of str, optional
        Explicitly provided feature names

    Returns
    -------
    X_array : NDArray
        Feature matrix as numpy array
    feature_names : list of str
        Feature names
    """
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
    """Compute prediction intervals using normal distribution.

    Parameters
    ----------
    mean : NDArray
        Mean predictions
    std : NDArray
        Standard deviation of predictions
    confidence_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% intervals)

    Returns
    -------
    lower : NDArray
        Lower bounds of prediction intervals
    upper : NDArray
        Upper bounds of prediction intervals
    """
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    lower = mean - z_score * std
    upper = mean + z_score * std
    return lower, upper


def extract_diagonal(
    cov: NDArray,
    return_format: str = 'vector',
) -> NDArray:
    """Extract diagonal elements from covariance matrix/matrices.

    Handles both 2D (single matrix) and 3D (sequence of matrices) inputs.

    Parameters
    ----------
    cov : NDArray
        Covariance matrix (2D) or sequence of covariance matrices (3D)
    return_format : str, default='vector'
        Format for return value:
        - 'vector': Return diagonal as 1D/2D array
        - 'matrix': Return diagonal matrix/matrices

    Returns
    -------
    NDArray
        Diagonal elements or diagonal matrix
    """
    if cov.ndim == 1:
        # Already a diagonal vector
        if return_format == 'matrix':
            return np.diag(cov)
        return cov
    elif cov.ndim == 2:
        # Single covariance matrix
        diag = np.diag(cov)
        if return_format == 'matrix':
            return np.diag(diag)
        return diag
    elif cov.ndim == 3:
        # Sequence of covariance matrices
        diag = np.diagonal(cov, axis1=1, axis2=2)
        if return_format == 'matrix':
            # Convert to diagonal matrices
            n_steps, n_dim = diag.shape
            result = np.zeros_like(cov)
            for i in range(n_steps):
                result[i] = np.diag(diag[i])
            return result
        return diag
    else:
        raise ValueError(f"Unsupported covariance dimension: {cov.ndim}")


def ensure_diagonal_matrix(P: NDArray) -> NDArray:
    """Ensure covariance is in full matrix form (not diagonal vector).

    Parameters
    ----------
    P : NDArray
        Covariance in vector or matrix form

    Returns
    -------
    NDArray
        Full covariance matrix
    """
    if P.ndim == 1:
        return np.diag(P)
    return P


def compute_aic(log_likelihood: float, n_params: int) -> float:
    """Compute Akaike Information Criterion.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of parameters

    Returns
    -------
    float
        AIC value
    """
    return 2 * n_params - 2 * log_likelihood


def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Compute Bayesian Information Criterion.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of parameters
    n_obs : int
        Number of observations

    Returns
    -------
    float
        BIC value
    """
    return n_params * np.log(n_obs) - 2 * log_likelihood


def compute_r_squared(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute R-squared (coefficient of determination).

    Parameters
    ----------
    y_true : NDArray
        True values
    y_pred : NDArray
        Predicted values

    Returns
    -------
    float
        R-squared value
    """
    # Handle NaN values
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
    """Reshape flat array to (n_times, n_locations) matrix.

    Parameters
    ----------
    flat_data : NDArray
        Flat array of length n_times * n_locations
    n_times : int
        Number of time steps
    n_locations : int
        Number of locations

    Returns
    -------
    NDArray
        Matrix of shape (n_times, n_locations)
    """
    if flat_data.size != n_times * n_locations:
        raise ValueError(
            f"Cannot reshape array of size {flat_data.size} into "
            f"shape ({n_times}, {n_locations})"
        )
    return flat_data.reshape(n_times, n_locations)


def reshape_matrix_to_flat(matrix: NDArray) -> NDArray:
    """Reshape (n_times, n_locations) matrix to flat array.

    Parameters
    ----------
    matrix : NDArray
        Matrix of shape (n_times, n_locations)

    Returns
    -------
    NDArray
        Flat array
    """
    return matrix.flatten()
