"""Abstract base classes for GAM-SSM-LUR estimators."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseEstimator(ABC):
    """Abstract base for all estimators."""

    def __init__(self):
        self.is_fitted_ = False

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @abstractmethod
    def fit(self, *args, **kwargs) -> BaseEstimator:
        """Fit the model — must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Generate predictions — must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_state_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _set_state_dict(self, state: Dict[str, Any]) -> None:
        pass

    def save(self, filepath: str | Path) -> None:
        """Pickle model state to disk."""
        self._check_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self._get_state_dict(), f)

    @classmethod
    def load(cls, filepath: str | Path) -> BaseEstimator:
        """Restore model from pickled state."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        model = cls.__new__(cls)
        model._set_state_dict(state)
        model.is_fitted_ = True
        return model

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return constructor params (sklearn compat)."""
        return {k: getattr(self, k) for k in self.__dict__ if not k.endswith("_")}

    def set_params(self, **params) -> BaseEstimator:
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ModelSummary:
    """Container for model summary statistics."""

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
        self.r_squared = r_squared
        self.rmse = rmse
        self.mae = mae
        self.aic = aic
        self.bic = bic
        self.n_params = n_params
        self.n_obs = n_obs
        self.additional = additional

    def __repr__(self) -> str:
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
        result = {
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "mae": self.mae,
            "aic": self.aic,
            "bic": self.bic,
            "n_params": self.n_params,
            "n_obs": self.n_obs,
        }
        result.update(self.additional)
        return {k: v for k, v in result.items() if v is not None}
