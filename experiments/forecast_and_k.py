#!/usr/bin/env python
"""
(a) Filter-only one-day-ahead forecast evaluation, and (b) sensitivity of
the GAM-SSM to the latent dimension k.

Forecast evaluation: the paper's reported skill uses RTS-smoothed states,
which condition on the full 30-day record including future days. Here the
same fitted model is evaluated with three information sets at EPA
station-days:

  smoothed   alpha_{t|T}   -- retrospective (as in the paper)
  filtered   alpha_{t|t}   -- real-time nowcast (no future data)
  one-step   alpha_{t|t-1} -- genuine one-day-ahead forecast

For the one-step and filtered variants, the fixed effects (beta, B_tilde)
are held at their full-sample EM estimates, so results are an upper bound
on operational forecast skill. The exogenous forcing u_t (traffic anomaly,
wind) is treated as known at forecast time.

k-sensitivity: refits the SSM with k in {1, 2, 3, 5, 10} (shared GAM),
recording SVD variance retained and pooled vs-EPA accuracy.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from gam_ssm_lur import HybridGAMSSM
from gam_ssm_lur.data import SpatiotemporalDataset
from gam_ssm_lur.evaluation import ModelEvaluator
from gam_ssm_lur.features import filter_sparse_cells, inverse_distance_transform
from gam_ssm_lur.models.spatial_gam import SpatialGAM
from pipeline.features import prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forecast_and_k")

EXPERIMENTS = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ID_COLS = ["grid_id", "latitude", "longitude"]


def reconstruct_total(model, alpha, include_forcing=True):
    """Grid-level field from a (T, k) alpha trajectory, mirroring
    HybridGAMSSM.predict() but with caller-supplied states."""
    gam_pred = model.gam_.predict(model._X_train)
    spatial = np.tile(gam_pred, (model.n_times_, 1))
    temporal = alpha @ model.Z_spatial_.T
    forcing = 0.0
    if include_forcing and model.ssm_.B_ is not None:
        forcing = (model.ssm_._forcing_matrix @ model.ssm_.B_.T) @ model.Z_spatial_.T
    beta_total = model.beta_total_ if model.beta_total_ is not None else 1.0
    return beta_total * spatial + temporal + forcing


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = SpatiotemporalDataset(
        data_dir=str(args.data_dir),
        target_col=args.target_col,
        dense_obs_file="satellite_retreavals.csv",
        dense_obs_value_col="tropomi_no2",
        point_obs_file="epa_timeseries.csv",
        point_obs_value_col="epa_no2",
        activity_file="traffic_timeseries.csv",
        activity_value_col="traffic_volume",
        met_forcing_file="wind_sector_2023-06_daily.csv",
        grid_geojson="grid/grid.geojson",
    )
    static = ds.load_static()
    temporal = ds.load_temporal()
    cal = ds.calibrate_dense_obs(temporal, static)

    feat_df = inverse_distance_transform(static.features)
    feat_df, y_full = filter_sparse_cells(
        feat_df,
        static.target[args.target_col],
        min_nonzero_features=1,
        id_cols=ID_COLS,
    )
    X_df, static_sel, _ = prepare_features(feat_df, y_full, args, output_dir)
    static_sel.target = static.target
    static_sel.grid_ids = list(feat_df["grid_id"].values)

    logger.info("Fitting shared GAM")
    shared_gam = SpatialGAM(n_splines=args.n_splines, lam="auto")
    shared_gam.fit(X_df.values, y_full.values, feature_names=list(X_df.columns))

    evaluator = ModelEvaluator()

    def fit_model(k):
        m = HybridGAMSSM(
            n_splines=args.n_splines,
            state_dim=k,
            em_max_iter=args.em_max_iter,
            em_tol=args.em_tol,
            scalability_mode="auto",
            random_state=42,
        )
        m.fit_from_dataset(
            static=static_sel,
            temporal=temporal,
            calibration=cal,
            prefit_gam=shared_gam,
        )
        return m

    # ---------------------------------------------------------------- forecast
    model = fit_model(3)
    grid_id_to_idx = {gid: i for i, gid in enumerate(model.location_ids_)}
    date_to_tidx = {d: i for i, d in enumerate(model.time_ids_)}
    epa = temporal.point_obs.copy()
    epa["loc_idx"] = epa["grid_id"].map(grid_id_to_idx)
    epa["t_idx"] = epa["date"].map(date_to_tidx)
    epa = epa.dropna(subset=["loc_idx", "t_idx", "obs_value"])
    epa[["loc_idx", "t_idx"]] = epa[["loc_idx", "t_idx"]].astype(int)

    d = model.ssm_._dynamic_dim
    fr = model.ssm_.filter_result_
    sm = model.ssm_.smoother_result_
    variants = {
        "smoothed (t|T, as reported)": sm.smoothed_means[:, :d],
        "filtered nowcast (t|t)": fr.filtered_means[:, :d],
        "one-day-ahead forecast (t|t-1)": fr.predicted_means[:, :d],
    }

    rows = []
    for name, alpha in variants.items():
        total = reconstruct_total(model, np.asarray(alpha))
        # skip day 0 for the one-step variant (prediction from the prior only)
        sub = epa[epa["t_idx"] >= 1] if "t-1" in name else epa
        y = sub["obs_value"].values
        yp = total[sub["t_idx"].values, sub["loc_idx"].values]
        a = evaluator.compute_accuracy(y, yp)
        rows.append(
            {
                "variant": name,
                "rmse": a.rmse,
                "mae": a.mae,
                "mbe": a.mbe,
                "r2": a.r2,
                "r": a.correlation,
                "n": len(y),
            }
        )

    # persistence on the same station-days for reference
    epa_sorted = epa.sort_values(["station_id", "t_idx"]).copy()
    epa_sorted["obs_prev"] = epa_sorted.groupby("station_id")["obs_value"].shift(1)
    pers = epa_sorted.dropna(subset=["obs_prev"])
    a = evaluator.compute_accuracy(pers["obs_value"].values, pers["obs_prev"].values)
    rows.append(
        {
            "variant": "persistence (today = yesterday)",
            "rmse": a.rmse,
            "mae": a.mae,
            "mbe": a.mbe,
            "r2": a.r2,
            "r": a.correlation,
            "n": len(pers),
        }
    )

    fc = pd.DataFrame(rows)
    fc.to_csv(output_dir / "forecast_eval.csv", index=False)
    print("\n== Forecast evaluation (vs EPA station-days) ==")
    print(fc.round(3).to_string(index=False))

    # ---------------------------------------------------------------- k-sens
    krows = []
    for k in [1, 2, 3, 5, 10]:
        t_k = time.time()
        mk = fit_model(k)
        pk = mk.predict()
        y = epa["obs_value"].values
        yp = pk.total[epa["t_idx"].values, epa["loc_idx"].values]
        a = evaluator.compute_accuracy(y, yp)
        # SVD variance retained by k factors
        rm = mk._residual_matrix
        y_filled = np.where(np.isnan(rm), 0.0, rm)
        s = np.linalg.svd(y_filled, compute_uv=False)
        var_ret = 100 * (s[: min(k, len(s))] ** 2).sum() / (s**2).sum()
        krows.append(
            {
                "k": k,
                "svd_var_retained_pct": var_ret,
                "rmse": a.rmse,
                "mae": a.mae,
                "r2": a.r2,
                "r": a.correlation,
                "em_iterations": mk.ssm_.get_diagnostics().em_iterations,
                "fit_seconds": time.time() - t_k,
            }
        )
        logger.info("k=%d done (%.1fs)", k, time.time() - t_k)

    kdf = pd.DataFrame(krows)
    kdf.to_csv(output_dir / "k_sensitivity.csv", index=False)
    print("\n== k-sensitivity (global calibration, smoothed, vs EPA) ==")
    print(kdf.round(3).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", "-d", default=str(DATA_DIR))
    p.add_argument(
        "--output-dir", "-o", default=str(EXPERIMENTS / "results" / "forecast_and_k")
    )
    p.add_argument("--target-col", default="atmos_no2")
    p.add_argument("--n-splines", type=int, default=10)
    p.add_argument("--em-max-iter", type=int, default=50)
    p.add_argument("--em-tol", type=float, default=1e-4)
    p.add_argument("--skip-feature-selection", action="store_true", default=True)
    p.add_argument("--corr-threshold", type=float, default=0.8)
    p.add_argument("--vif-threshold", type=float, default=10.0)
    p.add_argument("--importance-threshold", type=float, default=0.95)
    run(p.parse_args())


if __name__ == "__main__":
    main()
