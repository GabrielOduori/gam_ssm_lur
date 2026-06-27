"""
Step 5 — Figure generation
assemble all publication figures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gam_ssm_lur.visualization import (
    DiagnosticsVisualizer,
    SpatialVisualizer,
    TemporalVisualizer,
    create_publication_figure_set,
)

logger = logging.getLogger(__name__)


def generate_figures(
    model,
    ds,
    hybrid_pred,
    epa_eval,
    cv_df,
    X_train_df,
    args,
    data_dir: Path,
    output_dir: Path,
    fig_dir: Path,
    imp=None,
    moran_result=None,
    moran_weights=None,
):  # noqa: ARG001
    """Assemble and save all publication figures.

    Parameters
    ----------
    model : HybridGAMSSM
        Fitted model.
    ds : SpatiotemporalDataset
        Dataset object (used to load grid geometry).
    hybrid_pred : HybridPrediction
        Full spatiotemporal prediction from model.predict().
    epa_eval : pd.DataFrame
        EPA station observations aligned with model indices.
    cv_df : pd.DataFrame or None
        LOOCV results (or None).
    X_train_df : pd.DataFrame
        Training feature matrix with named columns.
    args : argparse.Namespace
        CLI arguments (met_forcing_file).
    data_dir : Path
        Root data directory.
    output_dir : Path
        Run output directory.
    fig_dir : Path
        Figures subdirectory.
    imp : pd.DataFrame or None
        Feature importances for the feature importance bar chart (optional).
    """
    grid_gdf = ds.load_grid_geometry()
    grid_id_to_idx = {gid: i for i, gid in enumerate(model.location_ids_)}

    # Daily hybrid prediction surface (one row per grid_id × date)
    ssm_df = pd.DataFrame(
        {
            "grid_id": np.tile(list(model.location_ids_), len(model.time_ids_)),
            "date": np.repeat(model.time_ids_, len(model.location_ids_)),
            "no2": hybrid_pred.total.ravel(),
        }
    )

    # Per-station time series
    gam_pred_by_gid = {
        gid: float(hybrid_pred.spatial[0, grid_id_to_idx[gid]])
        for gid in model.location_ids_
        if gid in grid_id_to_idx
    }
    station_preds = pd.DataFrame(
        [
            {
                "station_id": row["station_id"],
                "grid_id": row["grid_id"],
                "date": row["date"],
                "obs_no2": row["obs_value"],
                "lur_prior": gam_pred_by_gid.get(row["grid_id"], float("nan")),
                "no2": float(hybrid_pred.total[row["t_idx"], row["loc_idx"]]),
                "pred_uncertainty": float(
                    hybrid_pred.std[row["t_idx"], row["loc_idx"]]
                ),
            }
            for _, row in epa_eval.iterrows()
        ]
    )

    # ERA5 wind sector data for wind rose inset
    wind_path = data_dir / "time_series" / args.met_forcing_file
    wind_df = pd.read_csv(wind_path) if wind_path.exists() else None

    create_publication_figure_set(
        model=model,
        output_dir=fig_dir,
        grid_gdf=grid_gdf,
        cv_df=cv_df,
        ssm_df=ssm_df,
        station_preds=station_preds,
        X_train_df=X_train_df,
        wind_df=wind_df,
    )

    # SVD scree plot — factor selection diagnostic
    if model._residual_matrix is not None:
        dv = DiagnosticsVisualizer()
        dv.plot_svd_scree(
            model._residual_matrix,
            k_chosen=model.ssm_.state_dim,
            save_path=fig_dir / "svd_scree.png",
        )

    sv = SpatialVisualizer(grid_gdf=grid_gdf, grid_ids=list(model.location_ids_))

    # Redundant with SHAP beeswarm — SHAP shows direction + magnitude, RF importance just ranks
    # if imp is not None:
    #     sv.plot_feature_importance(
    #         imp, title="Selected Feature Importances",
    #         save_path=fig_dir / "feature_importance.png",
    #     )
    # The partial dependence plots are not very informative for this model,
    # and take a long time to compute, so I have decided to omit them from
    # the paper figures for now. I may revisit this decision in the future
    # if I find a more efficient way to compute them.
    # sv.plot_partial_dependence(
    #     model.gam_,
    #     n_top=min(9, len(model.gam_.feature_names_)),
    #     save_path=fig_dir / "gam_partial_response.png",
    # )

    sv.plot_shap_summary(
        model.gam_,
        X_train_df,
        save_path=fig_dir / "shap_summary.png",
    )

    # Moran's I scatterplot (spatial autocorrelation of GAM residuals)
    if moran_result is not None and moran_weights is not None:
        lur_res = model._y_train - model.gam_.predict(model._X_train)
        dv = DiagnosticsVisualizer()
        dv.plot_moran_scatterplot(
            lur_res,
            moran_weights,
            moran_result,
            save_path=fig_dir / "moran_scatterplot.png",
        )

    # TROPOMI-EPA satellite-to-surface calibration scatter (the OLS fit
    # actually used by the model, via model._calibration, not a re-derivation)
    if model._calibration is not None:
        dv = DiagnosticsVisualizer()
        dv.plot_calibration_scatter(
            model._calibration,
            save_path=fig_dir / "tropomi_epa_calibration_scatter.png",
        )

    # Probabilistic calibration — reliability diagram + sharpness + ISS
    y_obs = epa_eval["obs_value"].values
    y_pred = hybrid_pred.total[epa_eval["t_idx"].values, epa_eval["loc_idx"].values]
    y_std = hybrid_pred.std[epa_eval["t_idx"].values, epa_eval["loc_idx"].values]
    dv = DiagnosticsVisualizer()
    dv.plot_reliability_diagram(
        y_obs,
        y_pred,
        y_std,
        save_path=fig_dir / "reliability_diagram.png",
    )

    # EPA observed vs predicted time series (per-station + standalone daily mean)
    tv = TemporalVisualizer()
    tv.plot_epa_vs_predicted_timeseries(
        station_preds,
        save_path=fig_dir / "epa_vs_predicted_timeseries.png",
        summary_save_path=fig_dir / "epa_daily_mean_timeseries.png",
    )

    # SSM latent factor heatmap (alpha_t per day) — written by pipeline/evaluate.py
    # as ssm_temporal_factors.csv earlier in the run; load it back here so the
    # figure is produced automatically rather than requiring a manual call.
    temporal_factors_path = output_dir / "ssm_temporal_factors.csv"
    if temporal_factors_path.exists():
        temporal_factors = pd.read_csv(temporal_factors_path)
        dv = DiagnosticsVisualizer()
        dv.plot_factor_heatmap(
            temporal_factors,
            save_path=fig_dir / "factor_heatmap.png",
        )
