"""
GAM-SSM-LUR: Hybrid Generalized Additive Model–State Space Model
for Land Use Regression.

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

# Core models and base classes
from gam_ssm_lur import utils
from gam_ssm_lur.data import (
    CalibrationResult,
    SpatiotemporalDataset,
    StaticData,
    TemporalData,
)
from gam_ssm_lur.evaluation import ModelEvaluator

# Utilities
from gam_ssm_lur.features import (
    FeatureSelector,
    filter_sparse_cells,
    inverse_distance_transform,
)

# Inference algorithms
from gam_ssm_lur.inference import EMEstimator, KalmanFilter, RTSSmoother
from gam_ssm_lur.models import (
    BaseEstimator,
    HybridGAMSSM,
    ModelSummary,
    SpatialGAM,
    StateSpaceModel,
)
from gam_ssm_lur.visualization import (
    DiagnosticsVisualizer,
    ModelComparisonVisualizer,
    SpatialVisualizer,
    TemporalVisualizer,
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
    # Base classes
    "BaseEstimator",
    "ModelSummary",
    # Utilities
    "utils",
    "FeatureSelector",
    "inverse_distance_transform",
    "filter_sparse_cells",
    # Data loading
    "SpatiotemporalDataset",
    "StaticData",
    "TemporalData",
    "CalibrationResult",
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
