"""
Core model classes for GAM-SSM-LUR.

This module contains the main model implementations:
- HybridGAMSSM: The hybrid GAM-SSM model
- SpatialGAM: Spatial Generalized Additive Model component
- StateSpaceModel: State space model component for temporal dynamics
- BaseEstimator: Base class for all estimators
- ModelSummary: Summary dataclass for model results
"""

from gam_ssm_lur.models.base import BaseEstimator, ModelSummary
from gam_ssm_lur.models.hybrid import HybridGAMSSM
from gam_ssm_lur.models.spatial_gam import SpatialGAM, GAMSummary
from gam_ssm_lur.models.state_space import StateSpaceModel, SSMPrediction, SSMDiagnostics

__all__ = [
    "BaseEstimator",
    "ModelSummary",
    "HybridGAMSSM",
    "SpatialGAM",
    "GAMSummary",
    "StateSpaceModel",
    "SSMPrediction",
    "SSMDiagnostics",
]
