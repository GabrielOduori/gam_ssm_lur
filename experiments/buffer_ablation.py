#!/usr/bin/env python
"""
Wind-sector vs circular-buffer ablation for the GAM-LUR spatial component.

The manuscript's flagship predictor-engineering claim is that anisotropic
wind-sector buffers outperform conventional isotropic circular buffers. This
script tests it: a circular-buffer feature set is constructed by aggregating
each sectored feature family across its 8 sectors (sum for lengths/areas/
intensities; min for nearest-distance features -- the nearest source in any
direction), which is exactly the isotropic equivalent of the same underlying
OSM/SCATS geometry. Both sets then pass through the identical pipeline
(inverse-distance transform, sparse-cell filter, three-stage feature
selection with the same seed, GAM fit), and are compared on:

  - AtmosPlan emulation fit (R^2, RMSE, MAE) on the full grid
  - leave-one-station-out spatial CV at the 9 EPA stations
    (period-mean observed vs GAM prediction at the held-out cell)
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

from gam_ssm_lur.data import SpatiotemporalDataset
from gam_ssm_lur.evaluation import ModelEvaluator
from gam_ssm_lur.features import (
    FeatureSelector,
    filter_sparse_cells,
    inverse_distance_transform,
)
from gam_ssm_lur.models.spatial_gam import SpatialGAM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("buffer_ablation")

EXPERIMENTS = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ID_COLS = ["grid_id", "latitude", "longitude"]
SECTOR_SUFFIXES = tuple(f"_s{i}" for i in range(8))


def to_circular(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate _s0.._s7 sector features into isotropic circular buffers."""
    sectored = {}
    passthrough = []
    for c in features.columns:
        base = None
        for suf in SECTOR_SUFFIXES:
            if c.endswith(suf):
                base = c[: -len(suf)]
                break
        if base is None:
            passthrough.append(c)
        else:
            sectored.setdefault(base, []).append(c)

    out = features[passthrough].copy()
    agg_cols = {}
    for base, cols in sectored.items():
        block = features[cols]
        if "distance" in base:
            agg_cols[f"{base}_circ"] = block.min(axis=1)
        else:
            agg_cols[f"{base}_circ"] = block.sum(axis=1)
    out = pd.concat([out, pd.DataFrame(agg_cols, index=features.index)], axis=1)
    logger.info(
        "Circular set: %d sectored families aggregated, %d passthrough, %d total cols",
        len(sectored),
        len(passthrough),
        len(out.columns),
    )
    return out


def add_road_totals(features: pd.DataFrame) -> pd.DataFrame:
    """Mirror pipeline.features.prepare_features: add road_length_*_total
    columns summing the 8 sectors, alongside the sectored originals."""
    road_buffers = sorted(
        {
            c.rsplit("_s", 1)[0]
            for c in features.columns
            if c.startswith("road_length_") and "_s" in c
        }
    )
    totals = {}
    for base in road_buffers:
        cols = [f"{base}_s{i}" for i in range(8) if f"{base}_s{i}" in features.columns]
        if cols:
            totals[f"{base}_total"] = features[cols].sum(axis=1)
    if totals:
        features = pd.concat(
            [features, pd.DataFrame(totals, index=features.index)], axis=1
        )
    return features


def run_pipeline(
    name, features, target, station_summary, args, output_dir, preselected=None
):
    """Full spatial pipeline for one feature set; returns a metrics row."""
    t0 = time.time()
    feat_df = inverse_distance_transform(features)
    feat_df, y_full = filter_sparse_cells(
        feat_df, target, min_nonzero_features=1, id_cols=ID_COLS
    )
    feat_cols = [c for c in feat_df.columns if c not in ID_COLS]
    X_df = feat_df[feat_cols]
    X_df = X_df.select_dtypes(include=[np.number]).fillna(
        X_df.median(numeric_only=True)
    )

    if preselected is not None:
        keep = [c for c in preselected if c in X_df.columns]
        missing = [c for c in preselected if c not in X_df.columns]
        if missing:
            logger.warning(
                "[%s] %d preselected features missing: %s", name, len(missing), missing
            )
        X_sel = X_df[keep]
    else:
        selector = FeatureSelector(
            correlation_threshold=args.corr_threshold,
            vif_threshold=args.vif_threshold,
            importance_threshold=args.importance_threshold,
            random_state=42,
        )
        X_sel = selector.fit_transform(X_df, y_full)
    logger.info(
        "[%s] selected %d / %d features", name, len(X_sel.columns), len(X_df.columns)
    )
    (Path(output_dir) / f"selected_features_{name}.txt").write_text(
        "\n".join(X_sel.columns)
    )

    gam = SpatialGAM(n_splines=args.n_splines, lam="auto")
    gam.fit(X_sel.values, np.asarray(y_full), feature_names=list(X_sel.columns))
    pred = gam.predict(X_sel.values)
    ev = ModelEvaluator()
    fit = ev.compute_accuracy(np.asarray(y_full), pred)

    # spatial LOOCV at EPA stations (mirrors ModelEvaluator.loocv_stations)
    loc_ids = list(feat_df["grid_id"].values)
    X_arr = X_sel.values
    y_arr = np.asarray(y_full)
    recs = []
    for _, row in station_summary.iterrows():
        gid = row["grid_id"]
        if gid not in loc_ids:
            logger.warning("[%s] station cell %s not in grid — skipped", name, gid)
            continue
        idx = loc_ids.index(gid)
        mask = np.ones(len(loc_ids), dtype=bool)
        mask[idx] = False
        gam_loo = SpatialGAM(n_splines=args.n_splines, lam="auto")
        gam_loo.fit(X_arr[mask], y_arr[mask])
        recs.append(
            {
                "station_id": row["station_id"],
                "obs": row["obs_value"],
                "pred": float(gam_loo.predict(X_arr[[idx]])[0]),
            }
        )
    cv = pd.DataFrame(recs)
    cv_metrics = ev.compute_accuracy(cv["obs"].values, cv["pred"].values)
    cv.to_csv(Path(output_dir) / f"loocv_{name}.csv", index=False)

    return {
        "feature_set": name,
        "n_candidate": len(X_df.columns),
        "n_selected": len(X_sel.columns),
        "fit_r2": fit.r2,
        "fit_rmse": fit.rmse,
        "fit_mae": fit.mae,
        "loocv_r2": cv_metrics.r2,
        "loocv_rmse": cv_metrics.rmse,
        "loocv_r": cv_metrics.correlation,
        "n_stations": len(cv),
        "minutes": (time.time() - t0) / 60,
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

    station_summary = temporal.point_obs.groupby(
        ["station_id", "grid_id"], as_index=False
    )["obs_value"].mean()

    paper_features = None
    sel_files = sorted(
        (PROJECT_ROOT / "experiments" / "results").glob("run_*/selected_features.txt")
    )
    if sel_files:
        paper_features = sel_files[-1].read_text().splitlines()
        logger.info(
            "Paper feature list: %s (%d features)", sel_files[-1], len(paper_features)
        )

    ws_features = add_road_totals(static.features)

    arms = [
        ("wind_sector", ws_features, None),
        ("circular", to_circular(static.features), None),
    ]
    if paper_features:
        arms.insert(0, ("wind_sector_paper31", ws_features, paper_features))

    rows = []
    for name, feats, presel in arms:
        rows.append(
            run_pipeline(
                name,
                feats,
                static.target[args.target_col],
                station_summary,
                args,
                output_dir,
                preselected=presel,
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "buffer_ablation.csv", index=False)
    print("\n== Wind-sector vs circular-buffer ablation ==")
    print(out.round(3).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", "-d", default=str(DATA_DIR))
    p.add_argument(
        "--output-dir", "-o", default=str(EXPERIMENTS / "results" / "buffer_ablation")
    )
    p.add_argument("--target-col", default="atmos_no2")
    p.add_argument("--n-splines", type=int, default=10)
    p.add_argument("--corr-threshold", type=float, default=0.8)
    p.add_argument("--vif-threshold", type=float, default=10.0)
    p.add_argument("--importance-threshold", type=float, default=0.95)
    run(p.parse_args())


if __name__ == "__main__":
    main()
