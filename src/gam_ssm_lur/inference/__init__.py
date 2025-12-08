"""
Inference algorithms for GAM-SSM-LUR.

This module contains algorithms for parameter estimation and state inference:
- KalmanFilter: Forward Kalman filtering
- RTSSmoother: Rauch-Tung-Striebel backward smoothing
- EMEstimator: Expectation-Maximization algorithm for parameter learning
"""

from gam_ssm_lur.inference.kalman import (
    KalmanFilter,
    FilterResult,
)
from gam_ssm_lur.inference.smoother import RTSSmoother, SmootherResult
from gam_ssm_lur.inference.em import EMEstimator, EMResult

__all__ = [
    "KalmanFilter",
    "RTSSmoother",
    "FilterResult",
    "SmootherResult",
    "EMEstimator",
    "EMResult",
]
