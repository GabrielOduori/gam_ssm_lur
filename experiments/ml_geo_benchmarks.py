#!/usr/bin/env python
"""ML and geostatistical benchmarks (RF, GBM/XGBoost, GP kriging) under the
same LOSO protocol as the GAM-SSM, trained directly on EPA station readings.

Usage:
    python experiments/ml_geo_benchmarks.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from gam_ssm_lur.data import SpatiotemporalDataset
from gam_ssm_lur.evaluation import ModelEvaluator
from gam_ssm_lur.features import filter_sparse_cells, inverse_distance_transform
from pipeline.features import prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_geo_benchmarks")

EXPERIMENTS = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ID_COLS = ["grid_id", "latitude", "longitude"]


def build_design(ds, static, temporal, args, output_dir):
    """Station-day design matrix: selected static features at the station
    cell + daily covariates. Raw (uncalibrated) TROPOMI only, so no
    EPA-derived quantity enters the features."""
    feat_df = inverse_distance_transform(static.features)
    feat_df, y_full = filter_sparse_cells(
        feat_df,
        static.target[args.target_col],
        min_nonzero_features=1,
        id_cols=ID_COLS,
    )
    X_df, _, _ = prepare_features(feat_df, y_full, args, output_dir)
    X_df = X_df.copy()
    X_df["grid_id"] = feat_df["grid_id"].values
    X_df["latitude"] = feat_df["latitude"].values
    X_df["longitude"] = feat_df["longitude"].values

    obs = temporal.point_obs.dropna(subset=["obs_value"]).copy()

    act = dict(
        zip(
            temporal.activity_forcing["date"],
            temporal.activity_forcing["delta_activity"],
        )
    )
    if "grid_id" in temporal.met_forcing.columns:
        met = temporal.met_forcing.groupby("date")["met_forcing"].mean().to_dict()
    else:
        met = temporal.met_forcing.set_index("date")["met_forcing"].to_dict()

    trop = temporal.dense_obs.set_index(["grid_id", "date"])["obs_dense"]

    df = obs.merge(X_df, on="grid_id", how="inner")
    df["traffic_anom"] = df["date"].map(act).fillna(0.0)
    df["wind_forcing"] = df["date"].map(met).fillna(0.0)
    df["tropomi_raw"] = [
        trop.get((g, d), np.nan) for g, d in zip(df["grid_id"], df["date"])
    ]
    df["tropomi_missing"] = df["tropomi_raw"].isna().astype(float)
    df["tropomi_raw"] = df["tropomi_raw"].fillna(0.0)
    dates_sorted = sorted(obs["date"].unique())
    day_idx = {d: i for i, d in enumerate(dates_sorted)}
    df["day_idx"] = df["date"].map(day_idx).astype(float)

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "station_id",
            "grid_id",
            "date",
            "obs_value",
            "timestamp_utc",
            "latitude",
            "longitude",
        }
    ]
    logger.info(
        "Design matrix: %d station-days, %d features", len(df), len(feature_cols)
    )
    return df, feature_cols


def loso_eval(df, fit_predict, evaluator):
    """Pooled LOSO metrics using a fit_predict(train_df, test_df) callable."""
    preds = []
    for sid in sorted(df["station_id"].unique()):
        tr = df[df["station_id"] != sid]
        te = df[df["station_id"] == sid]
        yp = fit_predict(tr, te)
        preds.append(pd.DataFrame({"obs": te["obs_value"].values, "pred": yp}))
    allp = pd.concat(preds, ignore_index=True)
    a = evaluator.compute_accuracy(allp["obs"].values, allp["pred"].values)
    return {
        "rmse": a.rmse,
        "mae": a.mae,
        "mbe": a.mbe,
        "r2": a.r2,
        "r": a.correlation,
        "n": len(allp),
    }


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
    df, feat_cols = build_design(ds, static, temporal, args, output_dir)
    evaluator = ModelEvaluator()

    rows = []

    def rf_fit(tr, te):
        m = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        m.fit(tr[feat_cols], tr["obs_value"])
        return m.predict(te[feat_cols])

    rows.append(
        {
            "model": "RF ST-LUR (trained on EPA, LOSO)",
            **loso_eval(df, rf_fit, evaluator),
        }
    )
    logger.info("RF done: %s", rows[-1])

    try:
        from xgboost import XGBRegressor

        def gbm_fit(tr, te):
            m = XGBRegressor(n_estimators=500, random_state=42, verbosity=0)
            m.fit(tr[feat_cols], tr["obs_value"])
            return m.predict(te[feat_cols])

        gbm_name = "XGBoost ST-LUR (trained on EPA, LOSO)"
    except ImportError:

        def gbm_fit(tr, te):
            m = GradientBoostingRegressor(random_state=42)
            m.fit(tr[feat_cols], tr["obs_value"])
            return m.predict(te[feat_cols])

        gbm_name = "GBM ST-LUR (trained on EPA, LOSO)"

    rows.append({"model": gbm_name, **loso_eval(df, gbm_fit, evaluator)})
    logger.info("%s done: %s", gbm_name, rows[-1])

    # GP on (lat, lon, day): spatio-temporal kriging analogue
    def gp_fit(tr, te):
        Xtr = tr[["latitude", "longitude", "day_idx"]].values.copy()
        Xte = te[["latitude", "longitude", "day_idx"]].values.copy()
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        kern = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(
            1.0
        )
        m = GaussianProcessRegressor(kernel=kern, normalize_y=True, random_state=42)
        m.fit(Xtr, tr["obs_value"].values)
        return m.predict(Xte)

    rows.append(
        {
            "model": "GP spatio-temporal kriging (lat, lon, day; LOSO)",
            **loso_eval(df, gp_fit, evaluator),
        }
    )
    logger.info("GP done: %s", rows[-1])

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "ml_geo_benchmarks.csv", index=False)
    # LaTeX export so the manuscript benchmark rows are regenerable from code
    with open(output_dir / "ml_geo_benchmarks.tex", "w") as fh:
        fh.write(
            out.to_latex(
                index=False,
                float_format="%.2f",
                caption=(
                    "Machine learning and geostatistical benchmarks under the "
                    "leave-one-station-out protocol (RMSE, MAE, MBE in ug/m3)."
                ),
                label="tab:ml-geo-benchmarks",
            )
        )
    print()
    print(out.round(3).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", "-d", default=str(DATA_DIR))
    p.add_argument(
        "--output-dir", "-o", default=str(EXPERIMENTS / "results" / "ml_geo_benchmarks")
    )
    p.add_argument("--target-col", default="atmos_no2")
    p.add_argument("--skip-feature-selection", action="store_true", default=True)
    p.add_argument("--corr-threshold", type=float, default=0.8)
    p.add_argument("--vif-threshold", type=float, default=10.0)
    p.add_argument("--importance-threshold", type=float, default=0.95)
    run(p.parse_args())


if __name__ == "__main__":
    main()
