"""Base classes for GAM-SSM-LUR models."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray


class BaseEstimator(ABC):
    """Base class for all estimators in gam-ssm-lur.

    Provides common functionality for model fitting, validation, and serialization.
    """

    def __init__(self):
        """Initialize base estimator."""
        self.is_fitted_ = False

    def _check_fitted(self) -> None:
        """Check if model is fitted.

        Raises
        ------
        RuntimeError
            If model is not fitted
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @abstractmethod
    def fit(self, *args, **kwargs) -> BaseEstimator:
        """Fit the model.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Generate predictions.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _get_state_dict(self) -> Dict[str, Any]:
        """Get model state for serialization.

        Returns
        -------
        dict
            Dictionary containing model state
        """
        pass

    @abstractmethod
    def _set_state_dict(self, state: Dict[str, Any]) -> None:
        """Set model state from deserialization.

        Parameters
        ----------
        state : dict
            Dictionary containing model state
        """
        pass

    def save(self, filepath: str | Path) -> None:
        """Save model to disk using pickle.

        Parameters
        ----------
        filepath : str or Path
            Path to save the model

        Raises
        ------
        RuntimeError
            If model is not fitted
        """
        self._check_fitted()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = self._get_state_dict()

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str | Path) -> BaseEstimator:
        """Load model from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model

        Returns
        -------
        BaseEstimator
            Loaded model instance

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create instance and restore state
        model = cls.__new__(cls)
        model._set_state_dict(state)
        model.is_fitted_ = True

        return model

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        params = {}
        for key in self.__dict__:
            if not key.endswith('_'):  # Skip fitted attributes
                params[key] = getattr(self, key)
        return params

    def set_params(self, **params) -> BaseEstimator:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ModelSummary:
    """Container for model summary statistics.

    Attributes
    ----------
    r_squared : float
        Coefficient of determination
    rmse : float
        Root mean squared error
    mae : float
        Mean absolute error
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    n_params : int
        Number of model parameters
    n_obs : int
        Number of observations
    additional : dict
        Additional model-specific metrics
    """

    def __init__(
        self,
        r_squared: Optional[float] = None,
        rmse: Optional[float] = None,
        mae: Optional[float] = None,
        aic: Optional[float] = None,
        bic: Optional[float] = None,
        n_params: Optional[int] = None,
        n_obs: Optional[int] = None,
        **additional,
    ):
        """Initialize model summary.

        Parameters
        ----------
        r_squared : float, optional
            R-squared value
        rmse : float, optional
            Root mean squared error
        mae : float, optional
            Mean absolute error
        aic : float, optional
            Akaike Information Criterion
        bic : float, optional
            Bayesian Information Criterion
        n_params : int, optional
            Number of parameters
        n_obs : int, optional
            Number of observations
        **additional
            Additional model-specific metrics
        """
        self.r_squared = r_squared
        self.rmse = rmse
        self.mae = mae
        self.aic = aic
        self.bic = bic
        self.n_params = n_params
        self.n_obs = n_obs
        self.additional = additional

    def __repr__(self) -> str:
        """String representation of summary."""
        lines = ["Model Summary", "=" * 40]

        if self.r_squared is not None:
            lines.append(f"R²:        {self.r_squared:.4f}")
        if self.rmse is not None:
            lines.append(f"RMSE:      {self.rmse:.4f}")
        if self.mae is not None:
            lines.append(f"MAE:       {self.mae:.4f}")
        if self.aic is not None:
            lines.append(f"AIC:       {self.aic:.2f}")
        if self.bic is not None:
            lines.append(f"BIC:       {self.bic:.2f}")
        if self.n_params is not None:
            lines.append(f"Params:    {self.n_params}")
        if self.n_obs is not None:
            lines.append(f"Obs:       {self.n_obs}")

        if self.additional:
            lines.append("")
            lines.append("Additional Metrics:")
            for key, value in self.additional.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary.

        Returns
        -------
        dict
            Summary as dictionary
        """
        result = {
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'mae': self.mae,
            'aic': self.aic,
            'bic': self.bic,
            'n_params': self.n_params,
            'n_obs': self.n_obs,
        }
        result.update(self.additional)
        return {k: v for k, v in result.items() if v is not None}
