"""
GAM-SSM-LUR: Hybrid Generalized Additive Model–State Space Model for Land Use Regression.

A Python framework for spatiotemporal air pollution modeling that integrates
Generalized Additive Models (GAMs) with State Space Models (SSMs) for improved
prediction accuracy and uncertainty quantification.

Example
-------
>>> from gam_ssm_lur import HybridGAMSSM
>>> model = HybridGAMSSM()
>>> model.fit(X, y, time_index=timestamps)
>>> predictions = model.predict(X_new, return_std=True)
"""

__version__ = "0.1.0"
__author__ = "Gabriel Oduori"
__email__ = "gabriel.oduori@ucd.ie"

# Core models
from gam_ssm_lur.models import HybridGAMSSM, SpatialGAM, StateSpaceModel

# Inference algorithms
from gam_ssm_lur.inference import KalmanFilter, RTSSmoother, EMEstimator

# Utilities
from gam_ssm_lur.features import FeatureSelector, FeatureExtractor
from gam_ssm_lur.evaluation import ModelEvaluator
from gam_ssm_lur.visualization import (
    SpatialVisualizer,
    TemporalVisualizer,
    ModelComparisonVisualizer,
    DiagnosticsVisualizer,
    create_publication_figure_set,
)

# Base classes and utilities
from gam_ssm_lur.base import BaseEstimator, ModelSummary
from gam_ssm_lur import utils

__all__ = [
    # Main model
    "HybridGAMSSM",
    # Components
    "SpatialGAM",
    "StateSpaceModel",
    "KalmanFilter",
    "RTSSmoother",
    "EMEstimator",
    # Base classes
    "BaseEstimator",
    "ModelSummary",
    # Utilities
    "utils",
    "FeatureSelector",
    "FeatureExtractor",
    "ModelEvaluator",
    # Visualization
    "SpatialVisualizer",
    "TemporalVisualizer",
    "ModelComparisonVisualizer",
    "DiagnosticsVisualizer",
    "create_publication_figure_set",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
