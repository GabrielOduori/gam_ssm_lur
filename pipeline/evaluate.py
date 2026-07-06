"""
Step 4 — Evaluation:
metrics, LOOCV, calibration, diagnostic figures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gam_ssm_lur import ModelEvaluator

logger = logging.getLogger(__name__)


def _save_metrics(metrics, path: Path) -> None:
    pd.DataFrame([metrics.to_dict()]).to_csv(path, index=False)
    logger.info("Metrics → %s", path)


def evaluate_model(
    model,
    temporal,
    static,
    hybrid_pred,
    epa_eval,
    args,
    output_dir: Path,
    fig_dir: Path,
    grid_gdf=None,
):
    """Run all evaluation metrics and save CSVs / diagnostic figures.

    Parameters
    ----------
    model : HybridGAMSSM
        Fitted model.
    temporal : TemporalData
        Temporal dataset (used for LOOCV station lookup).
    static : StaticData
        Static dataset (used for Atmos-plan target lookup).
    hybrid_pred : HybridPrediction
        Full spatiotemporal prediction from model.predict().
    epa_eval : pd.DataFrame
        EPA station observations aligned with model indices
        (columns: station_id, grid_id, date, obs_value, loc_idx, t_idx).
    args : argparse.Namespace
        CLI arguments (target_col, point_obs_value_col).
    output_dir : Path
        Run output directory for CSV outputs.
    fig_dir : Path
        Figures directory for diagnostic plots.

    Returns
    -------
    cv_df : pd.DataFrame or None
        LOOCV results, or None if too few stations.
    """
    evaluator = ModelEvaluator(model)
    grid_id_to_idx = {gid: i for i, gid in enumerate(model.location_ids_)}

    # GAM-only spatial evaluation (against static target)
    lur_pred = model.gam_.predict(model._X_train)
    metrics = evaluator.compute_accuracy(model._y_train, lur_pred)
    logger.info(
        "GAM-LUR: RMSE=%.3f  MAE=%.3f  R²=%.3f  r=%.3f",
        metrics.rmse,
        metrics.mae,
        metrics.r2,
        metrics.correlation,
    )
    _save_metrics(metrics, output_dir / "gam_metrics.csv")

    # Moran's I on GAM-LUR residuals (spatial autocorrelation diagnostic)
    moran_result = moran_weights = None
    try:
        from esda.moran import Moran
        from libpysal.weights import Queen

        if grid_gdf is not None:
            gam_res = model._y_train - lur_pred
            loc_gdf = grid_gdf[grid_gdf["grid_id"].isin(model.location_ids_)].copy()
            loc_gdf = (
                loc_gdf.set_index("grid_id").reindex(model.location_ids_).reset_index()
            )
            moran_weights = Queen.from_dataframe(loc_gdf, silence_warnings=True)
            moran_weights.transform = "r"
            moran_result = Moran(gam_res, moran_weights, permutations=999)
            sig = (
                "p<0.001"
                if moran_result.p_sim < 0.001
                else f"p={moran_result.p_sim:.3f}"
            )
            status = (
                "spatially random ✓"
                if moran_result.p_sim > 0.05
                else "autocorrelation remains ⚠"
            )
            logger.info(
                "Moran's I (GAM residuals): I=%.4f  z=%.3f  %s  %s",
                moran_result.I,
                moran_result.z_norm,
                sig,
                status,
            )
            pd.DataFrame(
                [
                    {
                        "moran_I": moran_result.I,
                        "z_norm": moran_result.z_norm,
                        "p_sim": moran_result.p_sim,
                        "n_permutations": 999,
                    }
                ]
            ).to_csv(output_dir / "moran_i.csv", index=False)
    except ImportError:
        logger.warning("libpysal/esda not installed — Moran's I skipped")

    # Hybrid vs EPA daily observations
    y_obs = epa_eval["obs_value"].values
    y_hyb = hybrid_pred.total[epa_eval["t_idx"].values, epa_eval["loc_idx"].values]
    y_gam = hybrid_pred.spatial[epa_eval["t_idx"].values, epa_eval["loc_idx"].values]
    y_std_hyb = hybrid_pred.std[epa_eval["t_idx"].values, epa_eval["loc_idx"].values]

    hybrid_metrics = evaluator.compute_accuracy(y_obs, y_hyb)
    gam_temporal_metrics = evaluator.compute_accuracy(y_obs, y_gam)
    logger.info(
        "GAM only  (vs EPA, T×S): RMSE=%.3f  MAE=%.3f  R²=%.3f",
        gam_temporal_metrics.rmse,
        gam_temporal_metrics.mae,
        gam_temporal_metrics.r2,
    )
    logger.info(
        "GAM+SSM   (vs EPA, T×S): RMSE=%.3f  MAE=%.3f  R²=%.3f",
        hybrid_metrics.rmse,
        hybrid_metrics.mae,
        hybrid_metrics.r2,
    )
    _save_metrics(hybrid_metrics, output_dir / "hybrid_metrics.csv")

    # Site-specific temporal R²
    temporal_r2_records = []
    for sid, grp in epa_eval.groupby("station_id"):
        grp = grp.sort_values("t_idx")
        y_s = grp["obs_value"].values
        yh_s = hybrid_pred.total[grp["t_idx"].values, grp["loc_idx"].values]
        r2_s = (
            float(np.corrcoef(y_s, yh_s)[0, 1] ** 2)
            if len(y_s) >= 3 and np.std(y_s) > 0 and np.std(yh_s) > 0
            else float("nan")
        )
        temporal_r2_records.append({"station_id": sid, "temporal_r2": r2_s})
    temporal_r2_df = pd.DataFrame(temporal_r2_records)
    temporal_r2_df.to_csv(output_dir / "station_temporal_r2.csv", index=False)
    logger.info(
        "Site-specific temporal R² — median across %d stations: %.3f",
        len(temporal_r2_df),
        float(temporal_r2_df["temporal_r2"].median()),
    )

    # Atmos-plan annual mean vs temporal mean of hybrid predictions
    atmos_lookup = static.target.set_index("grid_id")[args.target_col]
    common_ids = [gid for gid in model.location_ids_ if gid in atmos_lookup.index]
    loc_indices = np.array([grid_id_to_idx[gid] for gid in common_ids])
    pred_annual_mean = hybrid_pred.total[:, loc_indices].mean(axis=0)
    atmos_vals = atmos_lookup.reindex(common_ids).values.astype(float)

    atmos_metrics = evaluator.compute_accuracy(atmos_vals, pred_annual_mean)
    logger.info(
        "Temporal mean vs Atmos-plan annual: RMSE=%.3f  MAE=%.3f  MBE=%.3f  R²=%.3f",
        atmos_metrics.rmse,
        atmos_metrics.mae,
        atmos_metrics.mbe,
        atmos_metrics.r2,
    )
    _save_metrics(atmos_metrics, output_dir / "atmos_plan_metrics.csv")

    pd.DataFrame(
        {
            "grid_id": common_ids,
            "atmos_no2": atmos_vals,
            "pred_annual_mean": pred_annual_mean,
            "deviation": pred_annual_mean - atmos_vals,
        }
    ).to_csv(output_dir / "atmos_plan_deviation.csv", index=False)
    logger.info(
        "Atmos-plan deviation saved (mean=%.3f)",
        float(pred_annual_mean - atmos_vals).mean()
        if False
        else float((pred_annual_mean - atmos_vals).mean()),
    )

    # Conformal calibration — leave-one-station-out correction factor.
    # Uses EPA validation observations to estimate a scale factor q̂ such that
    # prediction intervals achieve (1-α) empirical coverage at station locations.
    station_ids = epa_eval["station_id"].values
    q_hat_95 = model.calibrate_intervals_conformal(
        y_obs=y_obs,
        t_idx=epa_eval["t_idx"].values,
        loc_idx=epa_eval["loc_idx"].values,
        alpha=0.05,
        station_ids=station_ids,
    )
    q_hat_80 = model.calibrate_intervals_conformal(
        y_obs=y_obs,
        t_idx=epa_eval["t_idx"].values,
        loc_idx=epa_eval["loc_idx"].values,
        alpha=0.20,
        station_ids=station_ids,
    )
    logger.info(
        "Conformal correction factors: q̂(95%%)=%.3f  q̂(80%%)=%.3f", q_hat_95, q_hat_80
    )

    # Calibrated intervals (uncalibrated × q̂)
    y_std_calibrated = y_std_hyb * q_hat_95
    pd.DataFrame(
        [
            {
                "q_hat_95": round(q_hat_95, 4),
                "q_hat_80": round(q_hat_80, 4),
                "sigma2_obs": model._sigma2_obs,
            }
        ]
    ).to_csv(output_dir / "conformal_factors.csv", index=False)

    # Calibration metrics — report both raw and conformally calibrated
    calibration = evaluator.compute_calibration(y_obs, y_hyb, y_std_hyb)
    calibration_cal = evaluator.compute_calibration(y_obs, y_hyb, y_std_calibrated)
    logger.info(
        "Calibration (raw):    coverage_95=%.3f  width=%.3f  CRPS=%.4f",
        calibration.coverage["95%"],
        calibration.mean_interval_width,
        calibration.crps,
    )
    logger.info(
        "Calibration (conf.):  coverage_95=%.3f  width=%.3f  CRPS=%.4f",
        calibration_cal.coverage["95%"],
        calibration_cal.mean_interval_width,
        calibration_cal.crps,
    )
    pd.DataFrame(
        [
            {
                "coverage_50": calibration.coverage["50%"],
                "coverage_80": calibration.coverage["80%"],
                "coverage_90": calibration.coverage["90%"],
                "coverage_95": calibration.coverage["95%"],
                "interval_width": calibration.mean_interval_width,
                "iss": calibration.interval_skill_score,
                "crps": calibration.crps,
                "conf_coverage_95": calibration_cal.coverage["95%"],
                "conf_interval_width": calibration_cal.mean_interval_width,
                "conf_crps": calibration_cal.crps,
                "q_hat_95": round(q_hat_95, 4),
            }
        ]
    ).to_csv(output_dir / "calibration_metrics.csv", index=False)

    # Hybrid residual diagnostic figure — use calibrated std for the interval plot
    # evaluator.diagnostic_plots(
    #     y_true=y_obs,
    #     y_pred=y_hyb,
    #     y_std=y_std_calibrated,
    #     save_path=fig_dir / "hybrid_residual_diagnostics.png",
    # )

    # SSM temporal factors
    smoothed = model.ssm_.smoother_result_.smoothed_means
    ssm_states = pd.DataFrame(
        smoothed,
        index=model.time_ids_,
        columns=[f"factor_{i + 1}" for i in range(smoothed.shape[1])],
    )
    ssm_states.index.name = "date"
    ssm_states.to_csv(output_dir / "ssm_temporal_factors.csv")

    # LOOCV — use Atmos-plan target at each station's grid cell
    atmos_target = static.target.set_index("grid_id")[args.target_col]
    station_grid_ids = temporal.point_obs[["station_id", "grid_id"]].drop_duplicates()
    station_annual = station_grid_ids.copy()
    station_annual[args.point_obs_value_col] = station_annual["grid_id"].map(
        atmos_target
    )
    station_annual = station_annual.dropna(subset=[args.point_obs_value_col])
    station_annual = station_annual[station_annual["grid_id"].isin(model.location_ids_)]

    cv_df = None
    if len(station_annual) >= 2:
        logger.info("Running LOOCV on %d stations...", len(station_annual))
        cv_df = evaluator.loocv_stations(
            station_annual, obs_col=args.point_obs_value_col
        )
        cv_df.to_csv(output_dir / "loocv_results.csv", index=False)
        cv_metrics = evaluator.compute_accuracy(
            cv_df["obs_no2"].values, cv_df["no2"].values
        )
        logger.info(
            "LOOCV: RMSE=%.3f  MAE=%.3f  R²=%.3f",
            cv_metrics.rmse,
            cv_metrics.mae,
            cv_metrics.r2,
        )
        _save_metrics(cv_metrics, output_dir / "loocv_metrics.csv")
    else:
        logger.warning("Not enough stations for LOOCV (%d)", len(station_annual))

    # Model summary report — use conformally calibrated std so reported
    # coverage figures are for the corrected (publishable) intervals
    report = evaluator.summary_report(
        y_true=y_obs,
        y_pred=y_hyb,
        y_std=y_std_calibrated,
    )
    (output_dir / "model_summary.txt").write_text(report)

    # Side-by-side model comparison table
    _write_comparison_table(
        gam_spatial_metrics=metrics,
        gam_temporal_metrics=gam_temporal_metrics,
        hybrid_metrics=hybrid_metrics,
        loocv_metrics=cv_metrics if cv_df is not None else None,
        calibration=calibration_cal,  # use conformally calibrated metrics
        output_dir=output_dir,
    )

    return cv_df, moran_result, moran_weights


def _write_comparison_table(
    gam_spatial_metrics,
    gam_temporal_metrics,
    hybrid_metrics,
    loocv_metrics,
    calibration,
    output_dir: Path,
):
    """Save model_comparison.csv and model_comparison.tex."""

    def _det_row(label, m):
        return {
            "Model": label,
            "RMSE": round(m.rmse, 3),
            "MAE": round(m.mae, 3),
            "MBE": round(m.mbe, 3),
            "R²": round(m.r2, 3),
            "r": round(m.correlation, 3),
            # probabilistic columns filled for deterministic rows
            "Coverage 50%": "—",
            "Coverage 95%": "—",
            "Interval width": "—",
            "CRPS": "—",
        }

    rows = [
        _det_row("GAM-LUR (static, vs Atmos-plan)", gam_spatial_metrics),
        _det_row("GAM-LUR (vs EPA daily, no SSM)", gam_temporal_metrics),
        _det_row("GAM-SSM (vs EPA daily)", hybrid_metrics),
    ]
    if loocv_metrics is not None:
        rows.append(_det_row("GAM-SSM LOOCV (spatial CV)", loocv_metrics))

    # Probabilistic row — GAM-SSM only (GAM has no uncertainty)
    rows.append(
        {
            "Model": "GAM-SSM probabilistic (vs EPA daily)",
            "RMSE": "—",
            "MAE": "—",
            "MBE": "—",
            "R²": "—",
            "r": "—",
            "Coverage 50%": round(calibration.coverage["50%"], 3),
            "Coverage 95%": round(calibration.coverage["95%"], 3),
            "Interval width": round(calibration.mean_interval_width, 3),
            "CRPS": round(calibration.crps, 3),
        }
    )

    df = pd.DataFrame(rows).set_index("Model")

    # Improvement row (deterministic metrics only)
    gam_row = df.loc["GAM-LUR (vs EPA daily, no SSM)"]
    ssm_row = df.loc["GAM-SSM (vs EPA daily)"]
    df = pd.concat(
        [
            df.reset_index(),
            pd.DataFrame(
                [
                    {
                        "Model": "Improvement (GAM→SSM)",
                        "RMSE": round(
                            float(gam_row["RMSE"]) - float(ssm_row["RMSE"]), 3
                        ),
                        "MAE": round(float(gam_row["MAE"]) - float(ssm_row["MAE"]), 3),
                        "MBE": round(float(gam_row["MBE"]) - float(ssm_row["MBE"]), 3),
                        "R²": round(float(ssm_row["R²"]) - float(gam_row["R²"]), 3),
                        "r": round(float(ssm_row["r"]) - float(gam_row["r"]), 3),
                        "Coverage 50%": "—",
                        "Coverage 95%": "—",
                        "Interval width": "—",
                        "CRPS": "—",
                    }
                ]
            ),
        ]
    ).set_index("Model")

    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path)
    logger.info("Model comparison table → %s", csv_path)

    # LaTeX version
    # Writing mostly in LaTeX; this format outputs a nice table for the paper.
    # Uses pandas' built-in to_latex() which handles formatting and escaping.
    tex = df.style.to_latex(
        caption=(
            "Comparison of GAM-LUR and GAM-SSM model performance against EPA daily "
            "observations. RMSE, MAE, MBE, and interval width in µg/m³. "
            "Coverage is the empirical proportion of observations inside the nominal "
            "prediction interval. CRPS is the Continuous Ranked Probability Score "
            "(lower = better). Improvement row shows GAM-LUR minus GAM-SSM "
            "(positive = SSM better for RMSE/MAE; positive R²/r = SSM better fit)."
        ),
        label="tab:model_comparison",
        hrules=True,
    )
    tex_path = output_dir / "model_comparison.tex"
    tex_path.write_text(tex)
    logger.info("Model comparison LaTeX → %s", tex_path)
