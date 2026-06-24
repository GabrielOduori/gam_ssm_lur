"""
End-to-end tests for HybridGAMSSM.fit_from_dataset(), the method actually
used by experiments/reproduce_paper.py. This is the integration point for
the joint beta/B_tilde estimation introduced to replace the previous
fixed-coefficient (beta=1) GAM subtraction.

The headline regression test (test_temporal_correction_not_anti_correlated_
with_gam) directly reproduces the diagnostic that uncovered the original
bug: when the satellite field only weakly tracks the GAM baseline, naively
subtracting GAM at full strength (beta=1) before SVD manufactures a spurious
negative correlation between the resulting temporal correction and the GAM's
own spatial pattern. Jointly estimating beta should eliminate this.
"""

import numpy as np
import pandas as pd
import pytest

from gam_ssm_lur.models.hybrid import HybridGAMSSM
from gam_ssm_lur.data import StaticData, TemporalData


def _build_synthetic_dataset(n_cells=60, n_times=30, true_beta=0.4, seed=0):
    rng = np.random.default_rng(seed)
    grid_ids = [f"c{i}" for i in range(n_cells)]
    lat = rng.uniform(53.30, 53.40, n_cells)
    lon = rng.uniform(-6.40, -6.10, n_cells)

    feat1 = rng.uniform(0, 1, n_cells)
    feat2 = rng.uniform(0, 1, n_cells)
    features = pd.DataFrame({
        "grid_id": grid_ids, "latitude": lat, "longitude": lon,
        "feat1": feat1, "feat2": feat2,
    })

    # Smooth, learnable target so the GAM recovers a clear spatial pattern.
    target_vals = 10 + 8 * feat1 - 5 * feat2 + rng.normal(0, 0.2, n_cells)
    target = pd.DataFrame({
        "grid_id": grid_ids, "latitude": lat, "longitude": lon,
        "atmos_no2": target_vals,
    })
    static = StaticData(features=features, target=target, grid_ids=grid_ids)

    dates = pd.date_range("2023-06-01", periods=n_times, freq="D").date.tolist()

    # A smooth spatial pattern that is DELIBERATELY uncorrelated with the GAM
    # target (orthogonal-ish random pattern), used as the "true" dynamic
    # factor loading -- mimicking a satellite signal that carries genuine,
    # independent spatial information beyond the GAM baseline.
    true_loading = rng.normal(0, 1, n_cells)
    true_loading -= true_loading.mean()

    true_factor = np.zeros(n_times)
    for t in range(1, n_times):
        true_factor[t] = 0.7 * true_factor[t - 1] + rng.normal(0, 0.3)

    dense_rows = []
    for t, d in enumerate(dates):
        # obs_dense = beta*target + true dynamic pattern + noise
        obs = (
            true_beta * target_vals
            + true_loading * true_factor[t]
            + rng.normal(0, 0.3, n_cells)
        )
        for i, gid in enumerate(grid_ids):
            dense_rows.append({"grid_id": gid, "date": d, "obs_dense": obs[i]})
    dense_obs = pd.DataFrame(dense_rows)

    point_obs = pd.DataFrame({
        "station_id": [], "grid_id": [], "date": [], "obs_value": [],
    })

    activity_forcing = pd.DataFrame({
        "date": dates,
        "activity_mean": rng.normal(100, 10, n_times),
        "delta_activity": rng.normal(0, 0.1, n_times),
    })
    met_forcing = pd.DataFrame({
        "date": dates,
        "met_forcing": rng.normal(0, 1, n_times),
    })

    temporal = TemporalData(
        dense_obs=dense_obs, point_obs=point_obs,
        activity_forcing=activity_forcing, met_forcing=met_forcing,
        dates=dates,
    )
    return static, temporal, target_vals, true_loading, true_beta


class TestFitFromDataset:
    def test_runs_end_to_end_and_produces_finite_predictions(self):
        static, temporal, _, _, _ = _build_synthetic_dataset()
        model = HybridGAMSSM(
            state_dim=2, n_splines=4, em_max_iter=30,
            scalability_mode="dense", random_state=0,
        )
        model.fit_from_dataset(static, temporal, calibration=None)

        assert model.is_fitted_
        assert model.ssm_.beta_ is not None
        assert np.isfinite(model.ssm_.beta_)

        pred = model.predict()
        assert pred.total.shape == (30, 60)
        assert np.all(np.isfinite(pred.total))
        assert np.all(np.isfinite(pred.std))

    def test_beta_recovered_reasonably_close_to_truth(self):
        static, temporal, _, _, true_beta = _build_synthetic_dataset(true_beta=0.4)
        model = HybridGAMSSM(
            state_dim=2, n_splines=4, em_max_iter=40,
            scalability_mode="dense", random_state=0,
        )
        model.fit_from_dataset(static, temporal, calibration=None)

        # model.beta_total_ = beta_init (pooled OLS, used for the SVD basis)
        # + model.ssm_.beta_ (the jointly-estimated residual correction).
        # Not exact recovery (GAM fit + SVD truncation add noise), but should
        # land reasonably close to the true generating coefficient.
        assert abs(model.beta_total_ - true_beta) < 0.4

    def test_temporal_correction_not_anti_correlated_with_gam(self):
        """Regression test for the original bug: the satellite field here is
        constructed to be only weakly related to the GAM target (similar to
        the real TROPOMI-EPA r^2 ~ 0.055 case). The old design (subtracting
        GAM at beta=1 before SVD) would manufacture a strong NEGATIVE
        correlation between the resulting temporal correction and the GAM's
        own spatial pattern. Jointly estimating beta should not.
        """
        static, temporal, target_vals, _, _ = _build_synthetic_dataset(
            true_beta=0.3, seed=2,
        )
        model = HybridGAMSSM(
            state_dim=2, n_splines=4, em_max_iter=40,
            scalability_mode="dense", random_state=0,
        )
        model.fit_from_dataset(static, temporal, calibration=None)

        gam_pred = model.gam_.predict(model._X_train)
        ssm_pred = model.ssm_.predict()
        # temporal correction at a representative timestep
        t = 15
        correction_t = ssm_pred.mean[t] @ model.Z_spatial_.T

        corr = np.corrcoef(gam_pred, correction_t)[0, 1]
        # The old (beta=1) design produced correlations around -0.5 to -0.65
        # in the real data under this same weak-relationship condition.
        assert corr > -0.3, (
            f"Temporal correction is strongly anti-correlated with GAM "
            f"(r={corr:.3f}); the beta=1 subtraction bug may have regressed."
        )
