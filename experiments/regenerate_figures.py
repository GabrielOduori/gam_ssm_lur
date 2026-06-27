#!/usr/bin/env python
"""
Regenerate figures from a saved model run without retraining.

Loads a previously completed run, reconstructs predictions, and calls the
current pipeline/figures.py so that any plotting changes take effect
immediately — no retraining needed.

Usage
-----
    # Specify a run directory
    python experiments/regenerate_figures.py \
        --run-dir experiments/results/run_20260405_213102

    # Use the most recent run automatically
    python experiments/regenerate_figures.py --latest

    # Write figures to a custom location
    python experiments/regenerate_figures.py --latest \
        --fig-dir experiments/results/run_20260405_213102/figures_v2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from gam_ssm_lur import HybridGAMSSM
from gam_ssm_lur.data import SpatiotemporalDataset
from pipeline.figures import generate_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def latest_run(results_dir: Path) -> Path:
    runs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )
    if not runs:
        raise FileNotFoundError(f"No run directories found in {results_dir}")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate figures from a saved run without retraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path, help="Path to a completed run directory")
    group.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent run in experiments/results/",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Root data directory (needed for EPA time series and grid geometry)",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Output figures directory (default: <run-dir>/figures)",
    )

    # Data column names — must match the original run
    parser.add_argument("--target-col", default="atmos_no2")
    parser.add_argument("--dense-obs-file", default="satellite_retreavals.csv")
    parser.add_argument("--dense-obs-value-col", default="tropomi_no2")
    parser.add_argument("--point-obs-file", default="epa_timeseries.csv")
    parser.add_argument("--point-obs-value-col", default="epa_no2")
    parser.add_argument("--activity-file", default="traffic_timeseries.csv")
    parser.add_argument("--activity-value-col", default="traffic_volume")
    parser.add_argument("--met-forcing-file", default="wind_sector_2023-06_daily.csv")

    args = parser.parse_args()

    run_dir = latest_run(RESULTS_DIR) if args.latest else args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    fig_dir = args.fig_dir or run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run directory : %s", run_dir)
    logger.info("Figures → %s", fig_dir)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model = HybridGAMSSM.load(run_dir / "model")

    # ── 2. Reconstruct predictions ────────────────────────────────────────────
    logger.info("Reconstructing predictions...")
    hybrid_pred = model.predict()

    # ── 3. Load grid geometry via dataset ─────────────────────────────────────
    logger.info("Loading grid geometry from %s", args.data_dir)
    ds = SpatiotemporalDataset(
        data_dir=str(args.data_dir),
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
    temporal = ds.load_temporal()

    # ── 4. Reconstruct epa_eval ───────────────────────────────────────────────
    grid_id_to_idx = {gid: i for i, gid in enumerate(model.location_ids_)}
    date_to_tidx = {d: i for i, d in enumerate(model.time_ids_)}

    epa_eval = temporal.point_obs.copy()
    epa_eval["loc_idx"] = epa_eval["grid_id"].map(grid_id_to_idx)
    epa_eval["t_idx"] = epa_eval["date"].map(date_to_tidx)
    epa_eval = epa_eval.dropna(subset=["loc_idx", "t_idx"])
    epa_eval[["loc_idx", "t_idx"]] = epa_eval[["loc_idx", "t_idx"]].astype(int)
    logger.info(
        "EPA eval: %d observation rows across %d stations",
        len(epa_eval),
        epa_eval["station_id"].nunique(),
    )

    # ── 5. Load saved LOOCV results (if present) ──────────────────────────────
    loocv_path = run_dir / "loocv_results.csv"
    cv_df = pd.read_csv(loocv_path) if loocv_path.exists() else None
    if cv_df is not None:
        logger.info("Loaded LOOCV results: %d rows", len(cv_df))
    else:
        logger.info("No LOOCV results found — skipping LOOCV scatter")

    # ── 6. Reconstruct X_train_df ─────────────────────────────────────────────
    X_train_df = pd.DataFrame(
        model._X_train,
        columns=model.gam_.feature_names_,
    )

    # ── 7. Load feature importances (if present) ──────────────────────────────
    imp_path = run_dir / "feature_importance.csv"
    imp = pd.read_csv(imp_path) if imp_path.exists() else None

    # ── 8. Reconstruct Moran weights + result (if saved) ─────────────────────
    moran_result = moran_weights = None
    moran_path = run_dir / "moran_i.csv"
    if moran_path.exists():
        try:
            from esda.moran import Moran
            from libpysal.weights import Queen

            grid_gdf = ds.load_grid_geometry()
            loc_gdf = grid_gdf[grid_gdf["grid_id"].isin(model.location_ids_)].copy()
            loc_gdf = (
                loc_gdf.set_index("grid_id").reindex(model.location_ids_).reset_index()
            )
            moran_weights = Queen.from_dataframe(loc_gdf, silence_warnings=True)
            moran_weights.transform = "r"
            lur_res = model._y_train - model.gam_.predict(model._X_train)
            moran_result = Moran(lur_res, moran_weights, permutations=999)
            logger.info(
                "Moran's I reconstructed: I=%.4f  p=%.4f",
                moran_result.I,
                moran_result.p_sim,
            )
        except ImportError:
            logger.warning("libpysal/esda not installed — Moran scatterplot skipped")

    # ── 9. Generate figures ───────────────────────────────────────────────────
    logger.info("Generating figures → %s", fig_dir)
    generate_figures(
        model=model,
        ds=ds,
        hybrid_pred=hybrid_pred,
        epa_eval=epa_eval,
        cv_df=cv_df,
        X_train_df=X_train_df,
        args=args,
        data_dir=Path(args.data_dir),
        output_dir=run_dir,
        fig_dir=fig_dir,
        imp=imp,
        moran_result=moran_result,
        moran_weights=moran_weights,
    )

    logger.info("Done. Figures saved to %s", fig_dir)


if __name__ == "__main__":
    main()
