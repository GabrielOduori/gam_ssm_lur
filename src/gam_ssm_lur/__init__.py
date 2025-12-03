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

from gam_ssm_lur.hybrid_model import HybridGAMSSM
from gam_ssm_lur.spatial_gam import SpatialGAM
from gam_ssm_lur.state_space import StateSpaceModel
from gam_ssm_lur.kalman import KalmanFilter, RTSSmoother
from gam_ssm_lur.em_estimator import EMEstimator
from gam_ssm_lur.features import FeatureSelector, FeatureExtractor
from gam_ssm_lur.evaluation import ModelEvaluator
from gam_ssm_lur.visualization import (
    SpatialVisualizer,
    TemporalVisualizer,
    ModelComparisonVisualizer,
    DiagnosticsVisualizer,
    create_publication_figure_set,
)

__all__ = [
    # Main model
    "HybridGAMSSM",
    # Components
    "SpatialGAM",
    "StateSpaceModel",
    "KalmanFilter",
    "RTSSmoother",
    "EMEstimator",
    # Utilities
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
