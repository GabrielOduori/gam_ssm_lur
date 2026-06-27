#!/usr/bin/env python
"""
Reproduces Paper Results: GAM-SSM-LUR for Prediction.

Runs the full pipeline using SpatiotemporalDataset → fit_from_dataset()
against the data directory structure.

Usage
-----
    python experiments/reproduce_paper.py \\
        --data-dir /path/to/data \\
        --output-dir results/

Data directory structure::

    <data-dir>/
        features.csv
        target.csv
        grid/grid.geojson
        time_series/
            satellite_retreavals.csv
            epa_timeseries.csv
            traffic_timeseries.csv
            wind_sector_2023-06_daily.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from gam_ssm_lur import HybridGAMSSM
from gam_ssm_lur.data import SpatiotemporalDataset
from gam_ssm_lur.features import filter_sparse_cells, inverse_distance_transform
from gam_ssm_lur.fetch_data import ensure_data_available
from pipeline.evaluate import evaluate_model
from pipeline.features import prepare_features
from pipeline.figures import generate_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENTS = Path(__file__).resolve().parent  # experiments/
DATA_DIR = PROJECT_ROOT / "data"  # absolute, works from anywhere
ID_COLS = ["grid_id", "latitude", "longitude"]


def _elapsed(t_start: float) -> str:
    s = time.time() - t_start
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


def run(args) -> None:
    t0 = t_stage = time.time()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    logger.info("Output directory: %s", output_dir.resolve())

    ensure_data_available(data_dir)

    # ---------------------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("STEP 1 — Load data from %s", data_dir)
    logger.info("=" * 60)

    ds = SpatiotemporalDataset(
        data_dir=str(data_dir),
        target_col=args.target_col,
        dense_obs_file=args.dense_obs_file,
        dense_obs_value_col=args.dense_obs_value_col,
        point_obs_file=args.point_obs_file,
        point_obs_value_col=args.point_obs_value_col,
        activity_file=args.activity_file,
        activity_value_col=args.activity_value_col,
        met_forcing_file=args.met_forcing_file,
        grid_geojson="grid/grid.geojson",
    )
    static = ds.load_static()
    temporal = ds.load_temporal()
    cal = ds.calibrate_dense_obs(temporal, static)
    logger.info(
        "Data loaded: %d cells, %d dates, %d dense-obs rows, %d station rows",
        len(static.grid_ids),
        len(temporal.dates),
        len(temporal.dense_obs),
        len(temporal.point_obs),
    )
    logger.info("Calibration: β₀=%.3f  β₁=%.3f  r=%.3f", cal.beta0, cal.beta1, cal.r)
    logger.info("STEP 1 done in %s", _elapsed(t_stage))
    t_stage = time.time()

    # ---------------------------------------------------------------------------
    # 2. Feature engineering
    # ---------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2 — Feature engineering")
    logger.info("=" * 60)

    feat_df = inverse_distance_transform(static.features)
    feat_df, y_full = filter_sparse_cells(
        feat_df,
        static.target[args.target_col],
        min_nonzero_features=args.min_nonzero_features,
        id_cols=ID_COLS,
    )
    _, static_sel, imp = prepare_features(feat_df, y_full, args, output_dir)
    static_sel.target = static.target
    static_sel.grid_ids = list(feat_df["grid_id"].values)
    logger.info("STEP 2 done in %s", _elapsed(t_stage))
    t_stage = time.time()

    # ---------------------------------------------------------------------------
    # 3. Train model
    # ---------------------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("STEP 3 — Train GAM-SSM model")
    logger.info("=" * 60)

    model = HybridGAMSSM(
        n_splines=args.n_splines,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
        scalability_mode=args.scalability_mode,
        random_state=42,
    )
    t_fit = time.time()
    model.fit_from_dataset(static=static_sel, temporal=temporal, calibration=cal)
    logger.info("GAM fitting:  %s", _elapsed(t_fit))

    t_ssm = time.time()
    ssm_diag = model.ssm_.get_diagnostics()
    logger.info(
        "SSM: converged=%s  iterations=%d  LL=%.4e  mode=%s",
        ssm_diag.em_converged,
        ssm_diag.em_iterations,
        ssm_diag.log_likelihood,
        model.ssm_.kf_.mode,
    )
    logger.info("SSM fitting:  %s", _elapsed(t_ssm))
    model.save(output_dir / "model")
    logger.info("STEP 3 done in %s", _elapsed(t_stage))
    t_stage = time.time()

    # ---------------------------------------------------------------------------
    # 4. Evaluate
    # ---------------------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("STEP 4 — Evaluate")
    logger.info("=" * 60)

    t_inf = time.time()
    hybrid_pred = model.predict()
    logger.info("Inference:    %s", _elapsed(t_inf))

    grid_id_to_idx = {gid: i for i, gid in enumerate(model.location_ids_)}
    date_to_tidx = {d: i for i, d in enumerate(model.time_ids_)}

    epa_eval = temporal.point_obs.copy()
    epa_eval["loc_idx"] = epa_eval["grid_id"].map(grid_id_to_idx)
    epa_eval["t_idx"] = epa_eval["date"].map(date_to_tidx)
    epa_eval = epa_eval.dropna(subset=["loc_idx", "t_idx"])
    epa_eval[["loc_idx", "t_idx"]] = epa_eval[["loc_idx", "t_idx"]].astype(int)

    cv_df, moran_result, moran_weights = evaluate_model(
        model,
        temporal,
        static,
        hybrid_pred,
        epa_eval,
        args,
        output_dir,
        fig_dir,
        grid_gdf=ds.load_grid_geometry(),
    )
    logger.info("STEP 4 done in %s", _elapsed(t_stage))
    t_stage = time.time()

    # ---------------------------------------------------------------------------
    #  5. Generate figures
    # ---------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5 — Generate figures → %s", fig_dir)
    logger.info("=" * 60)

    X_train_df = pd.DataFrame(model._X_train, columns=model.gam_.feature_names_)
    generate_figures(
        model,
        ds,
        hybrid_pred,
        epa_eval,
        cv_df,
        X_train_df,
        args,
        data_dir,
        output_dir,
        fig_dir,
        imp=imp,
        moran_result=moran_result,
        moran_weights=moran_weights,
    )
    logger.info("STEP 5 done in %s", _elapsed(t_stage))

    logger.info("=" * 60)
    logger.info("Total: %s   Outputs: %s", _elapsed(t0), output_dir)
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce GAM-SSM-LUR paper results on Dublin data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", "-d", default=str(DATA_DIR), help="Root data directory"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(EXPERIMENTS / "results" / "reproduce"),
        help="Directory for all outputs",
    )

    parser.add_argument("--target-col", default="atmos_no2")
    parser.add_argument("--dense-obs-file", default="satellite_retreavals.csv")
    parser.add_argument("--dense-obs-value-col", default="tropomi_no2")
    parser.add_argument("--point-obs-file", default="epa_timeseries.csv")
    parser.add_argument("--point-obs-value-col", default="epa_no2")
    parser.add_argument("--activity-file", default="traffic_timeseries.csv")
    parser.add_argument("--activity-value-col", default="traffic_volume")
    parser.add_argument("--met-forcing-file", default="wind_sector_2023-06_daily.csv")

    parser.add_argument("--min-nonzero-features", type=int, default=1)
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=0.95,
        help="Cumulative RF importance threshold for feature selection "
        "(default 0.95 = features explaining 95%% of total importance)",
    )
    parser.add_argument("--skip-feature-selection", action="store_true")

    parser.add_argument("--n-splines", type=int, default=10)
    parser.add_argument("--em-max-iter", type=int, default=50)
    parser.add_argument("--em-tol", type=float, default=1e-4)
    parser.add_argument(
        "--scalability-mode",
        choices=["auto", "dense", "diagonal", "block"],
        default="auto",
    )

    args = parser.parse_args()

    if "--output-dir" not in sys.argv and "-o" not in sys.argv:
        from datetime import datetime

        args.output_dir = str(
            EXPERIMENTS / "results" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    run(args)


if __name__ == "__main__":
    main()
