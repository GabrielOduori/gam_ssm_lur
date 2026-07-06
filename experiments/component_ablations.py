#!/usr/bin/env python
"""Component ablations: GAM+regression, GAM+day FEs, TROPOMI-only, and SSM
forcing knock-outs evaluated against EPA station-days (review Sec. 14).

Usage:
    python experiments/component_ablations.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
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
logger = logging.getLogger("component_ablations")

EXPERIMENTS = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ID_COLS = ["grid_id", "latitude", "longitude"]


def metrics_row(name, y, yp, evaluator):
    a = evaluator.compute_accuracy(np.asarray(y), np.asarray(yp))
    return {
        "model": name,
        "rmse": a.rmse,
        "mae": a.mae,
        "mbe": a.mbe,
        "r2": a.r2,
        "r": a.correlation,
        "n": len(y),
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

    grid_ids = list(feat_df["grid_id"].values)
    gam_at_cell = dict(zip(grid_ids, shared_gam.predict(X_df.values)))

    obs = temporal.point_obs.dropna(subset=["obs_value"]).copy()
    obs = obs[obs["grid_id"].isin(gam_at_cell)]
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
    obs["gam"] = obs["grid_id"].map(gam_at_cell)
    obs["traffic"] = obs["date"].map(act).fillna(0.0)
    obs["wind"] = obs["date"].map(met).fillna(0.0)
    dates_sorted = sorted(obs["date"].unique())
    d_idx = {d: i for i, d in enumerate(dates_sorted)}
    obs["d"] = obs["date"].map(d_idx)

    evaluator = ModelEvaluator()
    rows = []
    stations = sorted(obs["station_id"].unique())

    # GAM + covariate regression
    # --------------------------
    def loso_linear(design_fn, name):
        preds, ys = [], []
        for sid in stations:
            tr, te = obs[obs["station_id"] != sid], obs[obs["station_id"] == sid]
            Xtr, Xte = design_fn(tr), design_fn(te)
            beta, *_ = np.linalg.lstsq(Xtr, tr["obs_value"].values, rcond=None)
            preds.append(Xte @ beta)
            ys.append(te["obs_value"].values)
        rows.append(
            metrics_row(name, np.concatenate(ys), np.concatenate(preds), evaluator)
        )
        logger.info(
            "%s done: %s",
            name,
            {k: round(v, 3) for k, v in rows[-1].items() if k not in ("model",)},
        )

    loso_linear(
        lambda f: np.column_stack([np.ones(len(f)), f["gam"], f["traffic"], f["wind"]]),
        "GAM + traffic + wind regression (LOSO)",
    )

    def dayfe_design(f):
        D = np.zeros((len(f), len(dates_sorted)))
        D[np.arange(len(f)), f["d"].values] = 1.0
        return np.column_stack([f["gam"].values, D])

    loso_linear(dayfe_design, "GAM + day fixed effects (LOSO)")

    # TROPOMI-only baseline
    # ---------------------
    trop = temporal.dense_obs.set_index(["grid_id", "date"])["obs_dense"]
    preds, ys = [], []
    for sid in stations:
        cal_s = ds.calibrate_dense_obs(temporal, static, exclude_station=sid)
        te = obs[obs["station_id"] == sid]
        vals = np.array(
            [trop.get((g, d), np.nan) for g, d in zip(te["grid_id"], te["date"])]
        )
        ok = ~np.isnan(vals)
        preds.append(cal_s.apply(vals[ok]))
        ys.append(te["obs_value"].values[ok])
    rows.append(
        metrics_row(
            "TROPOMI-only, nested calibration (no GAM prior)",
            np.concatenate(ys),
            np.concatenate(preds),
            evaluator,
        )
    )
    logger.info("TROPOMI-only done")

    # SSM forcing knock-outs
    # ----------------------
    cal = ds.calibrate_dense_obs(temporal, static)
    grid_id_to_idx = {gid: i for i, gid in enumerate(static_sel.grid_ids)}

    def fit_eval_ssm(name, temporal_mod):
        m = HybridGAMSSM(
            n_splines=args.n_splines,
            em_max_iter=args.em_max_iter,
            em_tol=args.em_tol,
            scalability_mode="auto",
            random_state=42,
        )
        m.fit_from_dataset(
            static=static_sel,
            temporal=temporal_mod,
            calibration=cal,
            prefit_gam=shared_gam,
        )
        pred = m.predict()
        date_to_t = {d: i for i, d in enumerate(m.time_ids_)}
        f = obs.copy()
        f["loc"] = f["grid_id"].map(grid_id_to_idx)
        f["t"] = f["date"].map(date_to_t)
        f = f.dropna(subset=["loc", "t"])
        yp = pred.total[f["t"].astype(int).values, f["loc"].astype(int).values]
        rows.append(metrics_row(name, f["obs_value"].values, yp, evaluator))
        logger.info("%s done", name)

    zero_act = temporal.activity_forcing.assign(delta_activity=0.0)
    zero_met = temporal.met_forcing.assign(met_forcing=0.0)
    fit_eval_ssm("GAM-SSM, full forcing (reference, global cal)", temporal)
    fit_eval_ssm(
        "GAM-SSM, no traffic forcing",
        dataclasses.replace(temporal, activity_forcing=zero_act),
    )
    fit_eval_ssm(
        "GAM-SSM, no wind forcing", dataclasses.replace(temporal, met_forcing=zero_met)
    )
    fit_eval_ssm(
        "GAM-SSM, no forcing (intercept only)",
        dataclasses.replace(temporal, activity_forcing=zero_act, met_forcing=zero_met),
    )

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "component_ablations.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", "-d", default=str(DATA_DIR))
    p.add_argument(
        "--output-dir",
        "-o",
        default=str(EXPERIMENTS / "results" / "component_ablations"),
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
