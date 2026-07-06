#!/usr/bin/env python
"""Nested LOOCV: for each EPA station the Eq. 5 calibration and SSM are refit
excluding that station; conformal q is from the remaining 8 stations only.

Outputs are directly comparable to model_comparison.csv (Table 6).

Usage:
    python experiments/nested_loocv.py
    python experiments/nested_loocv.py --output-dir experiments/results/nested_loocv
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
logger = logging.getLogger("nested_loocv")

EXPERIMENTS = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ID_COLS = ["grid_id", "latitude", "longitude"]


def build_epa_eval(temporal, location_ids, time_ids) -> pd.DataFrame:
    grid_id_to_idx = {gid: i for i, gid in enumerate(location_ids)}
    date_to_tidx = {d: i for i, d in enumerate(time_ids)}
    epa = temporal.point_obs.copy()
    epa["loc_idx"] = epa["grid_id"].map(grid_id_to_idx)
    epa["t_idx"] = epa["date"].map(date_to_tidx)
    epa = epa.dropna(subset=["loc_idx", "t_idx", "obs_value"])
    epa[["loc_idx", "t_idx"]] = epa[["loc_idx", "t_idx"]].astype(int)
    return epa


def pooled_metrics(evaluator, y_obs, y_pred, y_std_cal) -> dict:
    acc = evaluator.compute_accuracy(y_obs, y_pred)
    out = {
        "rmse": acc.rmse,
        "mae": acc.mae,
        "mbe": acc.mbe,
        "r2": acc.r2,
        "r": acc.correlation,
    }
    if y_std_cal is not None:
        cal = evaluator.compute_calibration(y_obs, y_pred, y_std_cal)
        out["coverage_50"] = cal.coverage.get("50%", np.nan)
        out["coverage_95"] = cal.coverage.get("95%", np.nan)
        out["interval_width"] = cal.mean_interval_width
        out["crps"] = cal.crps
    return out


def run(args) -> None:
    t0 = time.time()
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

    logger.info("Fitting shared GAM (target=%s, no EPA dependence)", args.target_col)
    t_gam = time.time()
    shared_gam = SpatialGAM(n_splines=args.n_splines, lam="auto")
    shared_gam.fit(X_df.values, y_full.values, feature_names=list(X_df.columns))
    logger.info("Shared GAM fitted in %.1fs", time.time() - t_gam)

    stations = sorted(temporal.point_obs["station_id"].dropna().unique())
    logger.info("Stations for nested LOOCV: %s", stations)

    evaluator = ModelEvaluator()

    def fit_ssm(calibration):
        model = HybridGAMSSM(
            n_splines=args.n_splines,
            em_max_iter=args.em_max_iter,
            em_tol=args.em_tol,
            scalability_mode="auto",
            random_state=42,
        )
        model.fit_from_dataset(
            static=static_sel,
            temporal=temporal,
            calibration=calibration,
            prefit_gam=shared_gam,
        )
        return model

    # Global-calibration reference
    # ----------------------------
    cal_global = ds.calibrate_dense_obs(temporal, static)
    model_global = fit_ssm(cal_global)
    epa_eval = build_epa_eval(
        temporal, model_global.location_ids_, model_global.time_ids_
    )
    pred_g = model_global.predict()
    y_obs_g = epa_eval["obs_value"].values
    yp_g = pred_g.total[epa_eval["t_idx"].values, epa_eval["loc_idx"].values]
    ystd_g = pred_g.std[epa_eval["t_idx"].values, epa_eval["loc_idx"].values]
    q95_g = model_global.calibrate_intervals_conformal(
        y_obs=y_obs_g,
        t_idx=epa_eval["t_idx"].values,
        loc_idx=epa_eval["loc_idx"].values,
        alpha=0.05,
        station_ids=epa_eval["station_id"].values,
    )
    global_row = pooled_metrics(evaluator, y_obs_g, yp_g, ystd_g * q95_g)
    global_row["model"] = "GAM-SSM (global calibration, as in main run)"
    logger.info(
        "Global reference: %s",
        {k: round(v, 3) for k, v in global_row.items() if k != "model"},
    )

    # Per-station folds
    # -----------------
    fold_rows = []
    cal_rows = []
    heldout_frames = []

    for sid in stations:
        t_fold = time.time()
        cal_s = ds.calibrate_dense_obs(temporal, static, exclude_station=sid)
        cal_rows.append(
            {
                "held_out_station": sid,
                "beta0": cal_s.beta0,
                "beta1": cal_s.beta1,
                "r": cal_s.r,
                "sigma2_obs": cal_s.sigma2_obs,
                "n_collocated": cal_s.n_collocated,
            }
        )
        model_s = fit_ssm(cal_s)
        pred_s = model_s.predict()

        held = epa_eval[epa_eval["station_id"] == sid]
        train = epa_eval[epa_eval["station_id"] != sid]

        # conformal factor from the 8 training stations only
        q95 = model_s.calibrate_intervals_conformal(
            y_obs=train["obs_value"].values,
            t_idx=train["t_idx"].values,
            loc_idx=train["loc_idx"].values,
            alpha=0.05,
            station_ids=train["station_id"].values,
        )

        y_obs = held["obs_value"].values
        yp = pred_s.total[held["t_idx"].values, held["loc_idx"].values]
        ystd = pred_s.std[held["t_idx"].values, held["loc_idx"].values]
        ystd_cal = ystd * q95

        heldout_frames.append(
            pd.DataFrame(
                {
                    "station_id": sid,
                    "date": held["date"].values,
                    "obs": y_obs,
                    "pred": yp,
                    "std_cal": ystd_cal,
                }
            )
        )

        m = pooled_metrics(evaluator, y_obs, yp, ystd_cal)
        m["station_id"] = sid
        m["n_days"] = len(y_obs)
        m["q_hat_95"] = q95
        # station temporal r^2 (squared correlation, as in station_temporal_r2.csv)
        m["temporal_r2"] = (
            float(np.corrcoef(y_obs, yp)[0, 1] ** 2)
            if len(y_obs) >= 3 and np.std(y_obs) > 0 and np.std(yp) > 0
            else float("nan")
        )
        fold_rows.append(m)
        logger.info(
            "Fold %s done in %.1fs  RMSE=%.2f MAE=%.2f r=%.3f cov95=%.2f",
            sid,
            time.time() - t_fold,
            m["rmse"],
            m["mae"],
            m["r"],
            m.get("coverage_95", np.nan),
        )

    # Pooled
    # ------
    all_held = pd.concat(heldout_frames, ignore_index=True)
    nested_row = pooled_metrics(
        evaluator,
        all_held["obs"].values,
        all_held["pred"].values,
        all_held["std_cal"].values,
    )
    nested_row["model"] = "GAM-SSM (nested LOOCV, leakage-free)"
    logger.info(
        "Nested pooled:   %s",
        {k: round(v, 3) for k, v in nested_row.items() if k != "model"},
    )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "nested_loocv_per_station.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(
        output_dir / "nested_calibration_coefficients.csv", index=False
    )
    all_held.to_csv(output_dir / "nested_heldout_predictions.csv", index=False)
    comparison = pd.DataFrame([global_row, nested_row]).set_index("model")
    comparison.to_csv(output_dir / "nested_vs_global_comparison.csv")

    logger.info("Median nested temporal r^2: %.3f", fold_df["temporal_r2"].median())
    logger.info("Outputs in %s   total %.1f min", output_dir, (time.time() - t0) / 60)
    print()
    print(comparison.round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", "-d", default=str(DATA_DIR))
    p.add_argument(
        "--output-dir", "-o", default=str(EXPERIMENTS / "results" / "nested_loocv")
    )
    p.add_argument("--target-col", default="atmos_no2")
    p.add_argument("--n-splines", type=int, default=10)
    p.add_argument("--em-max-iter", type=int, default=50)
    p.add_argument("--em-tol", type=float, default=1e-4)
    # feature selection: reuse the latest run's selected_features.txt so the
    # nested run uses exactly the paper's 31 features
    p.add_argument("--skip-feature-selection", action="store_true", default=True)
    p.add_argument("--corr-threshold", type=float, default=0.8)
    p.add_argument("--vif-threshold", type=float, default=10.0)
    p.add_argument("--importance-threshold", type=float, default=0.95)
    run(p.parse_args())


if __name__ == "__main__":
    main()
