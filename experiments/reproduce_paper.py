#!/usr/bin/env python
"""
Reproduce Paper Results: GAM-SSM-LUR for Dublin NO₂ Prediction.

This script reproduces the main results from:
    "Hybrid Generalized Additive-State Space Modelling for Urban NO₂ Prediction:
     Integrating Spatial and Temporal Dynamics"

Usage:
    python experiments/reproduce_paper.py --data-dir data/ --output-dir results/

Data Requirements:
    Download from Zenodo: https://zenodo.org/uploads/16534138
    Expected file:
        - data_file.csv: Hourly NO₂ observations
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure local src/ is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from output_utils import make_experiment_dirs
from mapping_utils import (
    create_gridded_comparison,
    create_gridded_residual_map,
    create_uncertainty_surface,
    create_temporal_gridded_sequence,
)
from gam_ssm_lur.data_check import check_data_availability

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default prefixes for spatial/traffic feature columns in merged datasets.
# This list includes the provided columns plus a few common variants to stay backwards compatible.
BASE_FEATURE_PREFIXES = [
    "traffic_volume",
    "scats_distance",
    "motorway",
    "industrial",
    "industria",  # handle possible shortened column name
    "traffic_signals",  # if present
    # Legacy/extra prefixes retained for compatibility
    "commercial",
    "residential",
    "primary",
    "secondary",
    "trunk",
    "tertiary",
]

DEFAULT_DATA_URL = "https://zenodo.org/record/16534138/files/data_table.zip?download=1"
DEFAULT_DATA_FILE = Path("data") / "data_table.csv"


def auto_detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Auto-detect likely column names from CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    dict
        Dictionary with detected column names
    """
    detected = {}
    col_lower = {col: col.lower() for col in df.columns}
    lower_to_orig = {v: k for k, v in col_lower.items()}

    # Timestamp detection
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if datetime_cols:
        detected["timestamp"] = datetime_cols[0]
    else:
        # Look for common names
        for candidate in ["timestamp", "date", "time", "datetime", "dt"]:
            if candidate in lower_to_orig:
                detected["timestamp"] = lower_to_orig[candidate]
                break

    # Target detection (NO2 related) — prioritize high-quality EPA fields
    priority_targets = [
        "epa_no2",
        "epa",
        "epa_value",
        "epa_values",
        "epa_target",
    ]
    for cand in priority_targets:
        if cand in lower_to_orig:
            detected["target"] = lower_to_orig[cand]
            break
    if "target" not in detected:
        for col in df.columns:
            col_l = col.lower()
            if "epa" in col_l and "no2" in col_l:
                detected["target"] = col
                break
    if "target" not in detected:
        for col in df.columns:
            col_l = col.lower()
            if ("no2" in col_l or "no_2" in col_l) and "value" not in col_l:
                detected["target"] = col
                break

    # No automatic fallback: prefer explicit EPA target only

    # Location detection
    for candidate in ["grid_id", "location_id", "location", "site_id", "station_id"]:
        if candidate in lower_to_orig:
            detected["location"] = lower_to_orig[candidate]
            break

    # Coordinate detection
    for candidate in ["latitude", "lat"]:
        if candidate in lower_to_orig:
            detected["lat"] = lower_to_orig[candidate]
            break

    for candidate in ["longitude", "lon", "long", "lng"]:
        if candidate in lower_to_orig:
            detected["lon"] = lower_to_orig[candidate]
            break

    return detected


def validate_and_confirm_columns(
    data_file: Path,
    provided_args: dict[str, str | None],
    interactive: bool = True,
) -> dict[str, str]:
    """Load CSV preview and confirm column mappings.

    Parameters
    ----------
    data_file : Path
        Path to data file
    provided_args : dict
        User-provided column arguments
    interactive : bool
        Whether to prompt user for confirmation

    Returns
    -------
    dict
        Confirmed column mappings
    """
    logger.info(f"Loading preview from {data_file.name}...")

    try:
        # Load preview
        df_preview = pd.read_csv(data_file, nrows=5)
    except Exception as e:
        logger.error(f" Failed to load data file: {data_file}")
        logger.error(f"   Error: {e}")
        logger.info(f"\n Troubleshooting:")
        logger.info(f"   1. Check file exists: ls {data_file}")
        logger.info(f"   2. Check CSV format: head {data_file}")
        sys.exit(1)

    print(f"\n Data Preview from {data_file.name}")
    print(
        f"   Shape: {df_preview.shape[0]} rows (preview), {df_preview.shape[1]} columns"
    )
    print(f"   Columns: {', '.join(df_preview.columns[:10].tolist())}")
    if len(df_preview.columns) > 10:
        print(f"            ... and {len(df_preview.columns) - 10} more")
    print()

    # Auto-detect columns
    detected = auto_detect_columns(df_preview)

    # Merge with user-provided args (user args take precedence)
    final_mapping = {}
    field_map = {
        "timestamp": "timestamp_col",
        "target": "target_col",
        "fallback_target": "fallback_target_col",
        "location": "location_col",
        "lat": "lat_col",
        "lon": "lon_col",
    }

    for key, arg_name in field_map.items():
        # User-provided takes precedence
        if provided_args.get(arg_name):
            final_mapping[arg_name] = provided_args[arg_name]
        elif key in detected:
            final_mapping[arg_name] = detected[key]

    # Display auto-detected or user-provided columns
    print("🔍 Column Mapping:")
    print(
        f"   Timestamp column:       {final_mapping.get('timestamp_col', '❌ NOT FOUND')}"
    )
    print(
        f"   Target column:          {final_mapping.get('target_col', '❌ NOT FOUND')}"
    )
    print(
        f"   Fallback target column: {final_mapping.get('fallback_target_col', '(optional)')}"
    )
    print(
        f"   Location column:        {final_mapping.get('location_col', '❌ NOT FOUND')}"
    )
    print(f"   Latitude column:        {final_mapping.get('lat_col', '❌ NOT FOUND')}")
    print(f"   Longitude column:       {final_mapping.get('lon_col', '❌ NOT FOUND')}")

    # Check for missing required columns
    required = ["timestamp_col", "target_col", "location_col", "lat_col", "lon_col"]
    missing = [f for f in required if f not in final_mapping]

    if missing:
        print(f"\n ERROR: Could not detect the following required columns:")
        for m in missing:
            print(f"   - {m.replace('_col', '')}")
        print(f"\n Please specify them manually using command-line arguments:")
        print(f"   --{missing[0].replace('_', '-')} <column_name>")
        sys.exit(1)

    if interactive:
        print()
        response = input("✓ Use these columns? [Y/n]: ").strip().lower()
        if response not in ["", "y", "yes"]:
            print(
                "\n Aborted. Please specify columns manually using --timestamp-col, --target-col, etc."
            )
            sys.exit(0)
        print()

    return final_mapping


def setup_paths(data_dir: Path, output_base: Path, run_name: str | None) -> dict:
    """Set up file paths with a unique experiment folder to avoid overwriting."""
    output_root, groups = make_experiment_dirs(
        base=output_base,
        groups=["figures", "models", "tables"],
        run_name=run_name,
    )
    paths = {
        "data_dir": data_dir,
        "output_dir": output_root,
        "figures_dir": groups["figures"],
        "models_dir": groups["models"],
        "tables_dir": groups["tables"],
    }
    return paths


def _infer_feature_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    """Infer feature columns based on known prefixes."""
    candidates = {
        col for prefix in prefixes for col in df.columns if col.startswith(prefix)
    }
    return sorted(candidates)


def _spatial_idw_fill(
    df: pd.DataFrame,
    target_col: str,
    lon_col: str,
    lat_col: str,
    timestamp_col: str,
    k: int = 5,
    power: float = 2.0,
    min_neighbors: int = 1,
) -> pd.DataFrame:
    """
    Fill missing targets using 
    inverse-distance weighting 
    per timestamp."""
    if df[target_col].notna().all():
        return df

    filled_frames: list[pd.DataFrame] = []
    for _, frame in df.groupby(timestamp_col):
        frame = frame.copy()
        missing_mask = frame[target_col].isna()
        if not missing_mask.any():
            filled_frames.append(frame)
            continue

        known = frame[~missing_mask]
        if known.empty:
            filled_frames.append(frame)
            continue

        coords_known = known[[lon_col, lat_col]].to_numpy(dtype=float)
        vals_known = known[target_col].to_numpy(dtype=float)
        coords_missing = frame.loc[missing_mask, [lon_col, lat_col]].to_numpy(dtype=float)

        # Distances from each missing to all known
        dist = np.linalg.norm(coords_missing[:, None, :] - coords_known[None, :, :], axis=2)
        order = np.argsort(dist, axis=1)

        for idx, row_idx in enumerate(frame.index[missing_mask]):
            d_sorted = dist[idx, order[idx]]
            v_sorted = vals_known[order[idx]]

            # Exact match (zero distance)
            zero_mask = d_sorted == 0
            if zero_mask.any():
                frame.at[row_idx, target_col] = float(np.mean(v_sorted[zero_mask]))
                continue

            # Nearest neighbors with IDW
            nn = min(k, len(v_sorted))
            d_sel = d_sorted[:nn]
            v_sel = v_sorted[:nn]
            valid = ~np.isnan(v_sel)
            if valid.sum() < min_neighbors:
                continue
            d_sel = d_sel[valid]
            v_sel = v_sel[valid]
            weights = 1.0 / np.power(d_sel + 1e-8, power)
            frame.at[row_idx, target_col] = float(np.dot(weights, v_sel) / weights.sum())

        filled_frames.append(frame)

    return pd.concat(filled_frames, axis=0)


def _clean_merged_dataframe(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    fallback_target_col: str | None,
    location_col: str,
    lat_col: str,
    lon_col: str,
    feature_prefixes: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    """Clean merged dataset to handle missing/inf values and infer features."""
    required = {timestamp_col, target_col, location_col, lat_col, lon_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in merged data: {', '.join(sorted(missing))}"
        )

    df = df.copy()

    # Choose a valid EPA target column if the suggested one is empty/missing
    epa_candidates = [
        target_col,
        *[c for c in ["epa_no2", "epa", "epa_value", "epa_values", "epa_target"] if c != target_col],
    ]
    selected_target: str | None = None
    for cand in epa_candidates:
        if cand in df.columns:
            series = df[cand].replace([np.inf, -np.inf], np.nan)
            if series.notna().any():
                selected_target = cand
                break

    # If no valid values found, check if column exists and warn instead of failing
    if selected_target is None:
        available = [c for c in df.columns if "epa" in c.lower()]
        # Check if target column exists but has no valid values in the loaded subset
        if target_col in df.columns:
            warning_msg = (
                f"\n{'*'*70}\n"
                f"WARNING: Target column '{target_col}' has no valid values\n"
                f"{'*'*70}\n"
                f"Rows loaded: {len(df):,}\n"
                f"This may occur when using --max-records with a small sample size.\n"
                f"The model will attempt spatial/temporal imputation to fill EPA values.\n"
                f"If this fails, you'll need to increase --max-records or use the full dataset.\n"
                f"{'*'*70}\n"
            )
            logger.warning(warning_msg)
            selected_target = target_col
        else:
            raise ValueError(
                f"No EPA target column with valid values found. Tried: {epa_candidates}. "
                f"Available EPA-like columns: {available or 'none'}. "
                "Specify the correct column with --target-col."
            )
    if selected_target != target_col:
        logger.warning(
            f"Switching target column from '{target_col}' to '{selected_target}' "
            "because the original was empty."
        )
        target_col = selected_target

    df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)

    # Spatial IDW fill per timestamp for locations without EPA readings
    df = _spatial_idw_fill(
        df=df,
        target_col=target_col,
        lon_col=lon_col,
        lat_col=lat_col,
        timestamp_col=timestamp_col,
        k=5,
        power=2.0,
        min_neighbors=1,
    )

    # Interpolate EPA target within each location over time, then fill remaining with median/zero
    df = df.sort_values([location_col, timestamp_col])

    def _interp_target(group: pd.DataFrame) -> pd.Series:
        series = group.set_index(timestamp_col)[target_col].sort_index()
        series_interp = series.interpolate(method="time", limit_direction="both")
        return series_interp.reindex(series.index)

    # Avoid pandas FutureWarning about including grouping columns by operating on
    # the target Series after setting timestamp as the index.
    df_idxed = df.set_index(timestamp_col)
    interpolated = (
        df_idxed.groupby(location_col)[target_col]
        .apply(lambda s: s.sort_index().interpolate(method="time", limit_direction="both"))
        .reset_index(level=0, drop=True)
    )
    # Reindex to original timestamp order within the sorted dataframe
    df[target_col] = interpolated.reindex(df_idxed.index)
    # Fill any remaining gaps with global median; if none, fail fast
    median_target = df[target_col].median()
    if pd.isna(median_target):
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: No valid EPA measurements found in loaded data\n"
            f"{'='*70}\n"
            f"Target column: '{target_col}'\n"
            f"Rows loaded:   {len(df):,}\n"
            f"\n"
            f"The problem:\n"
            f"  - EPA measurements don't appear until around row 15,158\n"
            f"  - You loaded only {len(df):,} rows, which contain no EPA data\n"
            f"  - Without EPA values, the model cannot be trained\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Increase --max-records to at least 20,000:\n"
            f"     python experiments/reproduce_paper.py --max-records 20000 --yes\n"
            f"\n"
            f"  2. Use the full dataset (recommended for actual experiments):\n"
            f"     python experiments/reproduce_paper.py --yes\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    df[target_col] = df[target_col].fillna(median_target)

    df = df.dropna(subset=[target_col])
    df = df[~np.isinf(df[target_col])]
    df.loc[df[target_col] < 0, target_col] = 0
    if df.empty:
        raise ValueError(
            f"No rows remaining after cleaning target column '{target_col}'. "
            "Ensure the EPA target column exists and has non-null values."
        )

    # Clean coordinates
    for coord in [lon_col, lat_col]:
        if coord in df.columns:
            df = df.dropna(subset=[coord])
            df = df[~np.isinf(df[coord])]

    # Feature list (based on prefixes); exclude satellite NO2 from GAM features
    feature_cols = _infer_feature_columns(df, feature_prefixes)
    if not feature_cols:
        raise ValueError(
            "No feature columns found using provided prefixes. "
            "Check column names or extend BASE_FEATURE_PREFIXES."
        )

    # Clean features (replace inf/NaN with median or 0)
    clean_features = []
    for f in feature_cols:
        if f not in df.columns:
            continue
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
        median = df[f].median()
        df[f] = df[f].fillna(0 if pd.isna(median) else median)
        clean_features.append(f)

    return df, clean_features, target_col


def load_data(
    data_dir: Path,
    data_file: Path | None,
    timestamp_col: str,
    target_col: str,
    fallback_target_col: str | None,
    location_col: str,
    lat_col: str,
    lon_col: str,
    max_records: int | None = None,
) -> tuple:
    """Load Dublin NO₂ dataset, supporting both merged and separate inputs.

    Parameters
    ----------
    max_records : int, optional
        Maximum number of records to load (for testing). If None, loads all data.

    Returns
    -------
    observations : pd.DataFrame
        NO₂ observations with columns: timestamp, location_id, no2, lat, lon
    features : pd.DataFrame
        Spatial features for each location
    validation_stations : pd.DataFrame
        EPA validation station information (None if not provided)
    """
    logger.info("Loading data...")

    # Path 1: user-provided merged table
    if data_file:
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        logger.info(f"Downloading Data from Zenodo: {data_file}")

        # Load with optional row limit
        def _load_and_clean(nrows: int | None):
            frame = pd.read_csv(
                data_file,
                parse_dates=[timestamp_col],
                nrows=nrows,
            )
            return _clean_merged_dataframe(
                df=frame,
                timestamp_col=timestamp_col,
                target_col=target_col,
                fallback_target_col=fallback_target_col,
                location_col=location_col,
                lat_col=lat_col,
                lon_col=lon_col,
                feature_prefixes=BASE_FEATURE_PREFIXES,
            )

        if max_records:
            logger.info(f"Loading only first {max_records} records for testing")
            df, feature_cols, target_col = _load_and_clean(max_records)
        else:
            df, feature_cols, target_col = _load_and_clean(None)

        # Normalize column names for downstream code
        # df = df.head(100)
        df = df.rename(
            columns={
                timestamp_col: "timestamp",
                target_col: "no2",
                location_col: "location_id",
                lat_col: "lat",
                lon_col: "lon",
            }
        )

        observations = df[["timestamp", "location_id", "no2", "lat", "lon"]].copy()

        # Keep only the inferred/cleaned feature columns
        selected_feature_cols = [
            c for c in feature_cols if c in df.columns and c not in {"timestamp", "no2"}
        ]
        features = (
            df[["location_id"] + selected_feature_cols]
            .drop_duplicates(subset=["location_id"])
            .reset_index(drop=True)
        )

        logger.info(
            f"Loaded merged data: {len(observations)} observations across {len(features)} locations"
        )
        validation_stations = None
        return observations, features, validation_stations

    else:
        raise ValueError(
            "No data file provided and unable to proceed. "
            "Please specify --data-file with the path to your CSV dataset."
        )

    return observations, features, validation_stations


def fit_baseline_gam(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """Fit static GAM-LUR baseline model."""
    from gam_ssm_lur.models.spatial_gam import SpatialGAM

    logger.info("Fitting baseline GAM-LUR model...")

    gam = SpatialGAM(n_splines=10, lam="auto")
    gam.fit(X, y, feature_names=feature_names)

    y_pred = gam.predict(X)

    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    mbe = np.mean(y - y_pred)  # Mean Bias Error (positive = under-prediction)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    corr = np.corrcoef(y, y_pred)[0, 1]

    return {
        "model": gam,
        "predictions": y_pred,
        "rmse": rmse,
        "mae": mae,
        "mbe": mbe,
        "r2": r2,
        "correlation": corr,
    }


def fit_hybrid_model(
    X: np.ndarray,
    y: np.ndarray,
    time_idx: np.ndarray,
    loc_idx: np.ndarray,
    feature_names: list,
    scalability_mode: str = "auto",
) -> dict:
    """Fit hybrid GAM-SSM model."""
    from gam_ssm_lur import HybridGAMSSM

    logger.info("Fitting hybrid GAM-SSM model...")

    model = HybridGAMSSM(
        n_splines=10,
        gam_lam="auto",
        em_max_iter=50,
        em_tol=1e-6,
        scalability_mode=scalability_mode,
        confidence_level=0.95,
        random_state=42,
    )

    model.fit(
        X=pd.DataFrame(X, columns=feature_names),
        y=y,
        time_index=time_idx,
        location_index=loc_idx,
    )

    predictions = model.predict()
    metrics = model.evaluate(
        y_true=model._y_matrix.flatten(),
        y_pred=predictions.total.flatten(),
        y_lower=predictions.lower.flatten(),
        y_upper=predictions.upper.flatten(),
    )

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
    }


def create_comparison_table(
    baseline: dict, hybrid: dict, output_path: Path
) -> pd.DataFrame:
    """Create Table 2: Model comparison (GAM-only vs GAM-SSM)."""

    # Helper function to calculate improvement
    def calc_improvement(baseline_val, hybrid_val, lower_is_better=True):
        if lower_is_better:
            improvement = (baseline_val - hybrid_val) / baseline_val * 100
        else:
            improvement = (hybrid_val - baseline_val) / baseline_val * 100
        return f"{improvement:.1f}%"

    table = pd.DataFrame(
        {
            "Metric": [
                "RMSE (µg/m³)",
                "MAE (µg/m³)",
                "MBE (µg/m³)",
                "R²",
                "Pearson r",
                "CRPS (µg/m³)",
                "Coverage (90%)",
                "Coverage (95%)"
            ],
            "GAM-LUR": [
                f"{baseline['rmse']:.3f}",
                f"{baseline['mae']:.3f}",
                f"{baseline['mbe']:.3f}",
                f"{baseline['r2']:.3f}",
                f"{baseline['correlation']:.3f}",
                "—",  # No uncertainty estimates for GAM-LUR
                "—",
                "—"
            ],
            "GAM-SSM": [
                f"{hybrid['metrics']['rmse']:.3f}",
                f"{hybrid['metrics']['mae']:.3f}",
                f"{hybrid['metrics']['mbe']:.3f}",
                f"{hybrid['metrics']['r2']:.3f}",
                f"{hybrid['metrics']['correlation']:.3f}",
                f"{hybrid['metrics'].get('crps', 0.0):.3f}",
                f"{hybrid['metrics'].get('coverage_90', 0.0):.3f}",
                f"{hybrid['metrics']['coverage_95']:.3f}"
            ],
            "Improvement": [
                calc_improvement(baseline['rmse'], hybrid['metrics']['rmse'], True),
                calc_improvement(baseline['mae'], hybrid['metrics']['mae'], True),
                calc_improvement(abs(baseline['mbe']), abs(hybrid['metrics']['mbe']), True),
                calc_improvement(baseline['r2'], hybrid['metrics']['r2'], False),
                calc_improvement(baseline['correlation'], hybrid['metrics']['correlation'], False),
                "—",
                "—",
                "—"
            ]
        }
    )

    table.to_csv(output_path, index=False)
    logger.info(f"Saved comparison table to {output_path}")

    return table


def save_metrics_as_latex(baseline: dict, hybrid: dict, output_path: Path) -> None:
    """Export model performance metrics as LaTeX table for publication.

    Args:
        baseline: Dictionary with baseline GAM-LUR metrics (rmse, mae, mbe, r2, correlation)
        hybrid: Dictionary with hybrid GAM-SSM results (contains 'metrics' subdictionary)
        output_path: Path where LaTeX table will be saved (.tex file)

    Note:
        Table compares GAM-LUR (baseline) vs GAM-SSM (hybrid model).
    """
    # Build rows with numeric values and improvements
    rows = []
    metrics_config = [
        ("rmse", "RMSE (µg/m³)", True),  # True = lower is better
        ("mae", "MAE (µg/m³)", True),
        ("mbe", "MBE (µg/m³)", True, True),  # Fourth param = use absolute value for improvement
        ("r2", "R²", False),     # False = higher is better
        ("correlation", "Pearson r", False),
        ("crps", "CRPS (µg/m³)", True, False, True),  # Fifth param = GAM-SSM only
        ("coverage_90", "Coverage (90\\%)", False, False, True),
        ("coverage_95", "Coverage (95\\%)", False, False, True)
    ]

    for metric_info in metrics_config:
        metric_key = metric_info[0]
        metric_label = metric_info[1]
        lower_is_better = metric_info[2]
        use_abs_for_improvement = metric_info[3] if len(metric_info) > 3 else False
        gam_ssm_only = metric_info[4] if len(metric_info) > 4 else False

        try:
            # Get values
            if gam_ssm_only:
                gam_val = None  # No value for GAM-only
                hybrid_val = hybrid['metrics'][metric_key]
            else:
                gam_val = baseline[metric_key]
                hybrid_val = hybrid['metrics'][metric_key]

            # Calculate improvement (positive % means hybrid is better)
            if gam_val is not None:
                if use_abs_for_improvement:
                    # For MBE, use absolute values (smaller absolute bias is better)
                    improvement = (abs(gam_val) - abs(hybrid_val)) / abs(gam_val) * 100
                elif lower_is_better:
                    improvement = (gam_val - hybrid_val) / gam_val * 100
                else:
                    improvement = (hybrid_val - gam_val) / gam_val * 100
            else:
                improvement = None  # No improvement for GAM-SSM only metrics

            rows.append({
                "Metric": metric_label,
                "GAM-LUR": gam_val if gam_val is not None else "—",
                "GAM-SSM": hybrid_val,
                "Improvement (%)": improvement if improvement is not None else "—"
            })
        except KeyError as e:
            logger.warning(f"Missing metric {metric_key}: {e}")

    df = pd.DataFrame(rows)

    # Custom formatting function for mixed numeric and string data
    def format_value(val):
        if val == "—":
            return "—"
        elif isinstance(val, (int, float)):
            return f"{val:.3f}"
        else:
            return str(val)

    # Generate LaTeX table manually for better control
    latex_lines = [
        "\\begin{table}",
        "\\caption{Model Performance Comparison: GAM-LUR vs GAM-SSM}",
        "\\label{tab:model_performance}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Metric & GAM-LUR & GAM-SSM & Improvement (\\%) \\\\",
        "\\midrule"
    ]

    for _, row in df.iterrows():
        metric = row["Metric"]
        gam = format_value(row["GAM-LUR"])
        hybrid = format_value(row["GAM-SSM"])
        improvement = format_value(row["Improvement (%)"])
        latex_lines.append(f"{metric} & {gam} & {hybrid} & {improvement} \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    latex_table = "\n".join(latex_lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex_table)

    logger.info(f"Saved LaTeX performance table to {output_path}")


def create_convergence_plot(hybrid_result: dict, output_path: Path):
    """Create Figure 6: EM convergence diagnostics with enhanced styling from visualization_suite."""

    model = hybrid_result["model"]
    em_history = model.get_em_convergence()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    iterations = em_history["iteration"]

    # Log-likelihood convergence (enhanced styling)
    ax = axes[0, 0]
    ax.plot(
        iterations,
        em_history["log_likelihood"],
        "bo-",
        linewidth=2,
        markersize=8
    )
    ax.set_xlabel("EM Iteration", fontsize=10)
    ax.set_ylabel("Log-Likelihood", fontsize=10)
    ax.set_title("Log-Likelihood Convergence", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Parameter traces (enhanced colors from visualization_suite)
    ax = axes[0, 1]
    ax.plot(
        iterations,
        em_history["tr_T"],
        "ro-",
        label="tr(T)",
        linewidth=2,
        markersize=6
    )
    ax.plot(
        iterations,
        em_history["tr_Q"],
        "go-",
        label="tr(Q)",
        linewidth=2,
        markersize=6
    )
    ax.plot(
        iterations,
        em_history["tr_H"],
        "bo-",
        label="tr(H)",
        linewidth=2,
        markersize=6
    )
    ax.set_xlabel("EM Iteration", fontsize=10)
    ax.set_ylabel("Parameter Trace", fontsize=10)
    ax.set_title("Parameter Evolution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Log-likelihood differences (semilogy for convergence rate)
    ax = axes[1, 0]
    ll = em_history["log_likelihood"].values
    ll_change = np.abs(np.diff(ll))
    ax.semilogy(range(1, len(ll)), ll_change, "mo-", linewidth=2, markersize=6)
    ax.axhline(y=1e-6, color="r", linestyle="--", linewidth=2, label="Convergence threshold")
    ax.set_xlabel("EM Iteration", fontsize=10)
    ax.set_ylabel("|ΔLog-Likelihood|", fontsize=10)
    ax.set_title("Convergence Rate", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Final diagnostics text
    ax = axes[1, 1]
    ax.axis("off")
    diagnostics = model.ssm_.get_diagnostics()
    text = (
        f"EM Convergence Summary\n"
        f"{'='*30}\n\n"
        f"Converged: {diagnostics.em_converged}\n"
        f"Iterations: {diagnostics.em_iterations}\n"
        f"Final log-likelihood: {diagnostics.log_likelihood:.2e}\n"
        f"AIC: {diagnostics.aic:.2e}\n"
        f"BIC: {diagnostics.bic:.2e}\n\n"
        f"Process noise var: {diagnostics.process_noise_variance:.4f}\n"
        f"Observation noise var: {diagnostics.observation_noise_variance:.4f}\n"
    )
    ax.text(
        0.1,
        0.9,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax.set_title("(d) Summary")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved convergence plot to {output_path}")
    plt.close()


def create_residual_diagnostics(hybrid_result: dict, output_path: Path):
    """Create Figure 8: Residual diagnostics with enhanced styling from visualization_suite."""
    from scipy import stats

    model = hybrid_result["model"]
    predictions = hybrid_result["predictions"]

    y_true = model._y_matrix.flatten()
    y_pred = predictions.total.flatten()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Residuals vs fitted (enhanced with better visibility)
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=1, color='#3498db')
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Fitted Values", fontsize=10)
    ax.set_ylabel("Residuals", fontsize=10)
    ax.set_title("Residuals vs Fitted", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Q-Q plot (with enhanced grid)
    ax = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Normality Check)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Customize the Q-Q plot styling
    ax.get_lines()[0].set_markerfacecolor('#3498db')
    ax.get_lines()[0].set_markeredgecolor('#2c3e50')
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)

    # Histogram with normal overlay (enhanced styling)
    ax = axes[0, 2]
    ax.hist(
        residuals,
        bins=50,
        density=True,
        alpha=0.7,
        edgecolor="black",
        color='#3498db'
    )
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(
        x,
        stats.norm.pdf(x, residuals.mean(), residuals.std()),
        "r-",
        lw=2,
        label="Normal"
    )
    ax.set_xlabel("Residuals", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Residual Distribution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Temporal residual pattern (enhanced with better colors)
    ax = axes[1, 0]
    residual_matrix = residuals.reshape(model.n_times_, model.n_locations_)
    daily_mean = residual_matrix.mean(axis=1)
    daily_std = residual_matrix.std(axis=1)
    t = np.arange(len(daily_mean))
    ax.plot(t, daily_mean, "b-", lw=2, label="Mean", color='#3498db')
    ax.fill_between(
        t,
        daily_mean - daily_std,
        daily_mean + daily_std,
        alpha=0.3,
        label="±1 SD",
        color='#3498db'
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Residual", fontsize=10)
    ax.set_title("Temporal Residual Pattern", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Spatial residual pattern (enhanced histogram)
    ax = axes[1, 1]
    spatial_rmse = np.sqrt((residual_matrix**2).mean(axis=0))
    ax.hist(spatial_rmse, bins=30, edgecolor="black", alpha=0.7, color='#3498db')
    ax.axvline(
        x=spatial_rmse.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {spatial_rmse.mean():.3f}",
    )
    ax.set_xlabel("Location RMSE", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Spatial RMSE Distribution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ACF of residuals (enhanced styling)
    ax = axes[1, 2]
    # Use mean residuals across space
    try:
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(daily_mean, lags=50, ax=ax, alpha=0.05, color='#3498db')
        ax.set_title("Residual Autocorrelation", fontsize=11, fontweight='bold')
        ax.set_xlabel("Lag", fontsize=10)
        ax.set_ylabel("Autocorrelation", fontsize=10)
        ax.grid(True, alpha=0.3)
    except Exception:
        # Fallback if statsmodels is unavailable (with enhanced styling)
        nlags = min(50, len(daily_mean) // 2)
        autocorr = np.correlate(
            daily_mean - daily_mean.mean(),
            daily_mean - daily_mean.mean(),
            mode="full"
        )
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr[: nlags + 1] / autocorr[0]
        lags = np.arange(len(autocorr))

        ax.plot(lags, autocorr, "b-", linewidth=2, color='#3498db')
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
        ax.axhline(
            y=1.96 / np.sqrt(len(daily_mean)),
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5
        )
        ax.axhline(
            y=-1.96 / np.sqrt(len(daily_mean)),
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5
        )
        ax.set_xlabel("Lag", fontsize=10)
        ax.set_ylabel("Autocorrelation", fontsize=10)
        ax.set_title("Residual Autocorrelation", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved residual diagnostics to {output_path}")
    plt.close()


def plot_temporal_evolution(
    hybrid_result: dict,
    output_path: Path,
    selected_locations: list[int] = None,
    time_labels=None
) -> None:
    """Plot temporal evolution for selected locations (from visualization_suite.py).

    Shows observed vs smoothed data with 95% confidence intervals in a 2x3 grid.
    """
    model = hybrid_result["model"]
    predictions = hybrid_result["predictions"]

    # Select locations
    if selected_locations is None:
        n_locs = min(6, model.n_locations_)
        selected_locations = np.linspace(0, model.n_locations_ - 1, n_locs, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    timesteps = np.arange(model.n_times_)
    observed = model._y_matrix
    smoothed = predictions.total
    lower = predictions.lower
    upper = predictions.upper

    # Setup x-axis
    x_vals = timesteps
    x_label = "Time step"
    if time_labels is not None and len(time_labels) == model.n_times_:
        x_vals = time_labels
        try:
            x_vals = pd.to_datetime(x_vals)
            x_label = "Date"
        except Exception:
            x_label = "Time"

    for i, loc in enumerate(selected_locations):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot uncertainty band
        ax.fill_between(
            x_vals,
            lower[:, loc],
            upper[:, loc],
            alpha=0.2,
            color='#3498db',
            label="95% CI",
            zorder=1
        )

        # Plot observed vs smoothed
        ax.plot(
            x_vals,
            observed[:, loc],
            'o',
            alpha=0.6,
            label="Observed",
            markersize=2.5,
            linewidth=1,
            color='#2c3e50',
            zorder=3
        )
        ax.plot(
            x_vals,
            smoothed[:, loc],
            '-',
            label="Predicted",
            linewidth=1.5,
            color='#3498db',
            zorder=2
        )

        ax.set_title(f"Location {loc}", fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("NO₂ (µg/m³)", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Rotate x-axis labels if dates
        if isinstance(x_vals, pd.DatetimeIndex):
            ax.tick_params(axis='x', rotation=45)

    # Hide empty subplots
    for i in range(len(selected_locations), len(axes)):
        axes[i].axis('off')

    fig.suptitle(
        "Temporal Evolution: Observed vs Predicted with Uncertainty",
        fontsize=13,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved temporal evolution plot to {output_path}")
    plt.close()


def plot_spatial_patterns(
    hybrid_result: dict,
    output_path: Path,
    timesteps: list[int] = None,
    observations_df: pd.DataFrame = None
) -> None:
    """Plot spatial patterns at selected timesteps (from visualization_suite.py).

    Shows observed vs smoothed spatial distributions in a 2-row grid.
    """
    model = hybrid_result["model"]
    predictions = hybrid_result["predictions"]

    # Get coordinates
    if observations_df is not None:
        coords_df = observations_df[["location_id", "lat", "lon"]].drop_duplicates()
        coords_df = coords_df[coords_df["location_id"].isin(model.location_ids_)]
        coords_df = coords_df.set_index("location_id").loc[model.location_ids_]
        coords = coords_df[["lon", "lat"]].values
    else:
        # Fallback: use location indices as coordinates
        coords = np.column_stack([np.arange(model.n_locations_),
                                  np.arange(model.n_locations_)])

    # Select timesteps
    if timesteps is None:
        timesteps = [
            0,
            model.n_times_ // 4,
            model.n_times_ // 2,
            model.n_times_ - 1,
        ]

    n_times = len(timesteps)
    fig, axes = plt.subplots(2, n_times, figsize=(4 * n_times, 8))

    observed = model._y_matrix
    smoothed = predictions.total

    for i, day in enumerate(timesteps):
        # Observed data
        ax = axes[0, i]
        scatter1 = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=observed[day],
            cmap="viridis",
            s=50,
            alpha=0.7,
            vmin=np.nanmin(observed),
            vmax=np.nanmax(observed)
        )
        ax.set_title(f"Observed (Timestep {day})", fontsize=10, fontweight='bold')
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        plt.colorbar(scatter1, ax=ax, shrink=0.8, label="NO₂ (µg/m³)")
        ax.grid(True, alpha=0.3)

        # Smoothed data
        ax = axes[1, i]
        scatter2 = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=smoothed[day],
            cmap="viridis",
            s=50,
            alpha=0.7,
            vmin=np.nanmin(smoothed),
            vmax=np.nanmax(smoothed)
        )
        ax.set_title(f"Predicted (Timestep {day})", fontsize=10, fontweight='bold')
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        plt.colorbar(scatter2, ax=ax, shrink=0.8, label="NO₂ (µg/m³)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Spatial Patterns: Observed vs Predicted",
        fontsize=13,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved spatial patterns plot to {output_path}")
    plt.close()


def create_spatial_comparison_map(
    baseline: dict,
    hybrid_result: dict,
    observations: pd.DataFrame,
    output_path: Path,
) -> None:
    """Side-by-side spatial comparison of GAM-LUR vs GAM-SSM mean predictions."""
    model = hybrid_result["model"]
    preds = hybrid_result["predictions"]

    # Location lookups
    loc_meta = observations[["location_id", "lat", "lon"]].drop_duplicates()

    # Baseline per-location mean
    baseline_df = (
        pd.DataFrame(
            {
                "location_id": observations["location_id"],
                "baseline_pred": baseline["predictions"],
            }
        )
        .groupby("location_id", as_index=False)
        .mean()
    )

    # Hybrid per-location mean over time
    hybrid_means = preds.total.mean(axis=0)
    hybrid_df = pd.DataFrame(
        {
            "location_id": model.location_ids_,
            "hybrid_pred": hybrid_means,
        }
    )

    # Merge with coordinates
    merged = loc_meta.merge(baseline_df, on="location_id").merge(
        hybrid_df, on="location_id"
    )

    vmin = min(merged["baseline_pred"].min(), merged["hybrid_pred"].min())
    vmax = max(merged["baseline_pred"].max(), merged["hybrid_pred"].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, col, title in [
        (axes[0], "baseline_pred", "GAM-LUR (mean over time)"),
        (axes[1], "hybrid_pred", "GAM-SSM (mean over time)"),
    ]:
        sc = ax.scatter(
            merged["lon"],
            merged["lat"],
            c=merged[col],
            cmap="viridis",
            s=20,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("NO₂ (µg/m³)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved spatial comparison map to {output_path}")
    plt.close()


def create_residual_hotspot_map(
    hybrid_result: dict,
    baseline: dict,
    time_idx: np.ndarray,
    loc_idx: np.ndarray,
    observations: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize per-location RMSE for baseline vs hybrid (hotspots)."""
    model = hybrid_result["model"]
    y_matrix = model._y_matrix
    n_times, n_locs = y_matrix.shape

    # Create mappings from original indices to matrix positions
    time_map = {t: i for i, t in enumerate(model.time_ids_)}
    loc_map = {l: i for i, l in enumerate(model.location_ids_)}

    def _reshape(values: np.ndarray) -> np.ndarray:
        mat = np.full((n_times, n_locs), np.nan)
        for v, t, l in zip(values, time_idx, loc_idx):
            t_pos = time_map[t]
            l_pos = loc_map[l]
            mat[t_pos, l_pos] = v
        return mat

    baseline_matrix = _reshape(baseline["predictions"])
    hybrid_matrix = hybrid_result["predictions"].total

    baseline_rmse = np.sqrt(np.nanmean((y_matrix - baseline_matrix) ** 2, axis=0))
    hybrid_rmse = np.sqrt(np.nanmean((y_matrix - hybrid_matrix) ** 2, axis=0))

    loc_meta = observations[["location_id", "lat", "lon"]].drop_duplicates()
    rmse_df = pd.DataFrame(
        {
            "location_id": model.location_ids_,
            "baseline_rmse": baseline_rmse,
            "hybrid_rmse": hybrid_rmse,
        }
    ).merge(loc_meta, on="location_id")

    vmin = min(rmse_df["baseline_rmse"].min(), rmse_df["hybrid_rmse"].min())
    vmax = max(rmse_df["baseline_rmse"].max(), rmse_df["hybrid_rmse"].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, col, title in [
        (axes[0], "baseline_rmse", "GAM-LUR RMSE"),
        (axes[1], "hybrid_rmse", "GAM-SSM RMSE"),
    ]:
        sc = ax.scatter(
            rmse_df["lon"],
            rmse_df["lat"],
            c=rmse_df[col],
            cmap="inferno",
            s=25,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE (µg/m³)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved residual hotspot map to {output_path}")
    plt.close()


def create_shap_importance_plot(
    baseline: dict,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    max_samples: int = 800,
) -> None:
    """Create SHAP summary plot for the baseline GAM-LUR model. Falls back to permutation importance."""
    model = baseline["model"]
    tried_shap = False
    try:
        import shap  # type: ignore

        tried_shap = True
        X_sample = X
        if len(X) > max_samples:
            idx = np.random.choice(len(X), size=max_samples, replace=False)
            X_sample = X[idx]

        def predict_fn(x):
            return model.predict(x)

        explainer = shap.KernelExplainer(
            predict_fn, shap.sample(X_sample, min(200, len(X_sample)))
        )
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_names, show=False
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot to {output_path}")
        return
    except Exception as e:
        if tried_shap:
            logger.warning(
                f"SHAP plot failed ({e}); falling back to permutation importance"
            )
        else:
            logger.warning("shap not installed; falling back to permutation importance")

    # Fallback: permutation importance on RMSE
    import matplotlib.pyplot as plt

    baseline_pred = baseline["predictions"]
    baseline_rmse = np.sqrt(np.mean((y - baseline_pred) ** 2))
    rng = np.random.default_rng(42)
    deltas = []
    for j in range(X.shape[1]):
        X_perm = X.copy()
        rng.shuffle(X_perm[:, j])
        perm_pred = model.predict(X_perm)
        perm_rmse = np.sqrt(np.mean((y - perm_pred) ** 2))
        deltas.append(perm_rmse - baseline_rmse)
    deltas = np.array(deltas)

    sorted_idx = np.argsort(deltas)[::-1]
    plt.figure(figsize=(8, max(4, 0.3 * len(feature_names))))
    plt.barh(np.array(feature_names)[sorted_idx], deltas[sorted_idx], color="slateblue")
    plt.xlabel("RMSE increase when permuted (µg/m³)")
    plt.title("Feature importance (permutation fallback)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved permutation importance plot to {output_path}")


def create_morans_i_plot(
    residuals: np.ndarray, lats: np.ndarray, lons: np.ndarray, output_path: Path
) -> None:
    """Compute and plot Moran's I if esda/libpysal are available."""
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran
    except ImportError:
        logger.warning("libpysal/esda not installed; skipping Moran's I plot")
        return

    coords = np.column_stack([lons, lats])
    w = KNN.from_array(coords, k=8)
    w.transform = "r"
    m = Moran(residuals, w)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(m.sim, bins=30, density=True, alpha=0.7, color="gray")
    ax.axvline(m.I, color="red", linestyle="--", label=f"I = {m.I:.3f}")
    ax.set_title("Moran's I (residual means)")
    ax.set_xlabel("I simulated")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Moran's I plot to {output_path}")


def create_variogram_plot(
    residuals: np.ndarray, lats: np.ndarray, lons: np.ndarray, output_path: Path
) -> None:
    """Create empirical variogram plot if skgstat is available."""
    try:
        from skgstat import Variogram
    except ImportError:
        logger.warning("skgstat not installed; skipping variogram plot")
        return

    # Check if we have enough locations for variogram analysis
    if len(residuals) < 10:
        logger.warning(
            f"Only {len(residuals)} locations available; skipping variogram plot (need at least 10)"
        )
        return

    # Remove NaN values
    valid_mask = ~np.isnan(residuals)
    if valid_mask.sum() < 10:
        logger.warning(
            f"Only {valid_mask.sum()} valid residuals; skipping variogram plot (need at least 10)"
        )
        return

    coords = np.column_stack([lons[valid_mask], lats[valid_mask]])
    residuals_clean = residuals[valid_mask]

    V = Variogram(coords, residuals_clean, normalize=True, n_lags=12, maxlag="median")
    fig = V.plot(show=False)
    fig.set_figwidth(6)
    fig.set_figheight(4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved variogram plot to {output_path}")


def create_observed_vs_predicted_plot(
    baseline: dict, hybrid_result: dict, output_path: Path
):
    """Create Figure 7: Observed vs predicted comparison."""

    model = hybrid_result["model"]
    predictions = hybrid_result["predictions"]

    # Use the original training data for fair comparison
    y_true = model._y_train
    y_pred_baseline = baseline["predictions"]

    # For hybrid, we need to extract predictions at observed locations only
    # Create mapping from (time, location) to prediction
    y_pred_hybrid_full = predictions.total.flatten()

    # Build index mapping for observed data points
    time_map = {t: i for i, t in enumerate(model.time_ids_)}
    loc_map = {l: i for i, l in enumerate(model.location_ids_)}

    y_pred_hybrid = []
    for t_idx, l_idx in zip(model._time_index_train, model._location_index_train):
        t_pos = time_map[t_idx]
        l_pos = loc_map[l_idx]
        flat_idx = t_pos * model.n_locations_ + l_pos
        y_pred_hybrid.append(y_pred_hybrid_full[flat_idx])
    y_pred_hybrid = np.array(y_pred_hybrid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Baseline
    ax = axes[0]
    ax.scatter(y_true, y_pred_baseline, alpha=0.1, s=1)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    ax.set_xlabel("Observed NO₂ (µg/m³)")
    ax.set_ylabel("Predicted NO₂ (µg/m³)")
    ax.set_title(f"(a) GAM-only (R² = {baseline['r2']:.3f})")
    ax.set_aspect("equal")

    # Hybrid
    ax = axes[1]
    ax.scatter(y_true, y_pred_hybrid, alpha=0.1, s=1)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    corr = np.corrcoef(y_true, y_pred_hybrid)[0, 1]
    ax.set_xlabel("Observed NO₂ (µg/m³)")
    ax.set_ylabel("Predicted NO₂ (µg/m³)")
    ax.set_title(
        f"(b) GAM-SSM (R² = {hybrid_result['metrics']['r2']:.3f}, r = {corr:.3f})"
    )
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved observed vs predicted plot to {output_path}")
    plt.close()


def main():
    """Reproduce paper results."""

    parser = argparse.ArgumentParser(
        description="Reproduce GAM-SSM-LUR paper results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with auto-detected columns (interactive)
  python experiments/reproduce_paper.py --data-file data/my_data.csv

  # Quick test with first 100 records
  python experiments/reproduce_paper.py --data-file data/my_data.csv --max-records 100 --yes

  # Run with specific columns (non-interactive)
  python experiments/reproduce_paper.py --data-file data/my_data.csv \\
    --timestamp-col timestamp --target-col epa_no2 --location-col grid_id \\
    --lat-col latitude --lon-col longitude --yes

  # Run with demo data (no file needed)
  python experiments/reproduce_paper.py
        """,
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        required=False,
        help="Path to merged CSV file (if not provided, uses demo data)",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default=None,
        help="Timestamp column in merged data (auto-detected if not provided)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column (e.g., gold-standard EPA NO₂) in merged data (auto-detected if not provided)",
    )
    parser.add_argument(
        "--location-col",
        type=str,
        default=None,
        help="Location ID column in merged data (auto-detected if not provided)",
    )
    parser.add_argument(
        "--lat-col",
        type=str,
        default=None,
        help="Latitude column in merged data (auto-detected if not provided)",
    )
    parser.add_argument(
        "--lon-col",
        type=str,
        default=None,
        help="Longitude column in merged data (auto-detected if not provided)",
    )
    parser.add_argument(
        "--fallback-target-col",
        type=str,
        default=None,
        help="Optional column to fill missing target values (e.g., satellite NO₂) (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Base directory for output files (each run gets a timestamped subfolder)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom experiment folder name (defaults to experiment_<timestamp>)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip interactive confirmation (use auto-detected or provided columns)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records to load for testing (default: load all data)",
    )
    parser.add_argument(
        "--scalability-mode",
        type=str,
        default="auto",
        choices=["auto", "dense", "diagonal", "block"],
        help="Matrix scalability mode for Kalman filter/smoother; use diagonal for large datasets",
    )
    parser.add_argument(
        "--diagonal-threshold",
        type=int,
        default=500,
        help="If scalability-mode is auto and number of locations exceeds this, switch to diagonal",
    )
    parser.add_argument(
        "--skip-gridded-plots",
        action="store_true",
        help="Skip gridded/temporal map outputs (Figures 15-18)",
    )
    parser.add_argument(
        "--uncertainty-locations",
        type=str,
        default=None,
        help="Comma-separated location IDs for uncertainty plot (e.g., '0,5,10,15'). Default: auto-select 4 diverse locations",
    )
    args = parser.parse_args()

    # Ensure data is available (download from Zenodo if missing)
    if args.data_file:
        target_file = args.data_file
    else:
        target_file = DEFAULT_DATA_FILE

    data_dir = target_file.parent

    try:
        check_data_availability(target_file, DEFAULT_DATA_URL, data_dir)
    except Exception as exc:
        message = (
            f"Data file {target_file} is missing and automatic download failed. "
            f"Please download it manually from {DEFAULT_DATA_URL} and place it in {data_dir}."
        )
        logger.error(message)
        raise RuntimeError(message) from exc

    args.data_file = target_file

    # Determine column mappings
    provided_args = {
        "timestamp_col": args.timestamp_col,
        "target_col": args.target_col,
        "location_col": args.location_col,
        "lat_col": args.lat_col,
        "lon_col": args.lon_col,
        "fallback_target_col": args.fallback_target_col,
    }
    column_mapping = validate_and_confirm_columns(
        args.data_file,
        provided_args,
        interactive=not args.yes,
    )
    # Update args with validated columns
    args.timestamp_col = column_mapping["timestamp_col"]
    args.target_col = column_mapping["target_col"]
    args.location_col = column_mapping["location_col"]
    args.lat_col = column_mapping["lat_col"]
    args.lon_col = column_mapping["lon_col"]
    args.fallback_target_col = column_mapping.get("fallback_target_col")

    # Setup
    paths = setup_paths(data_dir, args.output_dir, args.run_name)

    logger.info("=" * 70)
    logger.info("GAM-SSM-LUR Paper Reproduction")
    logger.info("=" * 70)

    # Load data
    observations, features, validation_stations = load_data(
        data_dir=data_dir,
        data_file=args.data_file,
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        fallback_target_col=args.fallback_target_col,
        location_col=args.location_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        max_records=args.max_records,
    )

    # Prepare arrays
    logger.info("Preparing data arrays...")

    # Merge observations with features
    df = observations.merge(features, on="location_id")

    feature_cols = [c for c in features.columns if c != "location_id"]
    X = df[feature_cols].values
    y = df["no2"].values
    time_codes, time_uniques = pd.factorize(df["timestamp"])
    time_idx = time_codes  # integer indices for reshaping
    time_labels = pd.Index(time_uniques)
    loc_idx = df["location_id"].values

    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    n_locations = len(np.unique(loc_idx))
    n_times = len(time_labels)

    # Choose scalability mode; auto-switch to diagonal for large location sets
    scalability_mode = args.scalability_mode
    if scalability_mode == "auto" and n_locations >= args.diagonal_threshold:
        logger.info(
            "Auto-switching scalability mode to 'diagonal' "
            f"(locations={n_locations} >= threshold={args.diagonal_threshold})"
        )
        scalability_mode = "diagonal"
    logger.info(
        f"Using scalability mode: {scalability_mode} (locations={n_locations}, times={n_times})"
    )

    # Feature selection
    logger.info("=" * 70)
    logger.info("Feature Selection")
    logger.info("=" * 70)

    from gam_ssm_lur.features import FeatureSelector

    selector = FeatureSelector(
        correlation_threshold=0.8,
        vif_threshold=10.0,
        n_top_features=30,
        force_keep=(
            ["traffic_volume", "tropomi_no2"]
            if "traffic_volume" in feature_cols
            else None
        ),
        random_state=42,
    )

    X_df = pd.DataFrame(X, columns=feature_cols)
    X_selected = selector.fit_transform(X_df, y)
    selected_features = list(X_selected.columns)

    logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)}")

    # Fit baseline
    logger.info("=" * 70)
    logger.info("Fitting Baseline GAM-LUR")
    logger.info("=" * 70)

    baseline = fit_baseline_gam(X_selected.values, y, selected_features)
    logger.info(f"Baseline RMSE: {baseline['rmse']:.4f}")
    logger.info(f"Baseline R²: {baseline['r2']:.4f}")

    # Fit hybrid
    logger.info("=" * 70)
    logger.info("Fitting Hybrid GAM-SSM")
    logger.info("=" * 70)

    hybrid = fit_hybrid_model(
        X_selected.values,
        y,
        time_idx,
        loc_idx,
        selected_features,
        scalability_mode=scalability_mode,
    )
    logger.info(f"Hybrid RMSE: {hybrid['metrics']['rmse']:.4f}")
    logger.info(f"Hybrid R²: {hybrid['metrics']['r2']:.4f}")
    logger.info(f"95% Coverage: {hybrid['metrics']['coverage_95']:.1%}")

    # Calculate improvement
    rmse_improvement = (
        (baseline["rmse"] - hybrid["metrics"]["rmse"]) / baseline["rmse"] * 100
    )
    logger.info(f"RMSE Improvement: {rmse_improvement:.1f}%")

    # Create outputs
    logger.info("=" * 70)
    logger.info("Creating Outputs")
    logger.info("=" * 70)

    # Table 2: CSV format
    table = create_comparison_table(
        baseline, hybrid, paths["tables_dir"] / "table2_model_comparison.csv"
    )
    print("\nTable 2: Model Comparison")
    print(table.to_string(index=False))

    # Table 2: LaTeX format for publication
    save_metrics_as_latex(
        baseline, hybrid, paths["tables_dir"] / "table2_model_comparison.tex"
    )

    # Figure 6: Convergence
    create_convergence_plot(hybrid, paths["figures_dir"] / "fig6_convergence.png")

    # Figure 7: Observed vs Predicted
    create_observed_vs_predicted_plot(
        baseline, hybrid, paths["figures_dir"] / "fig7_observed_vs_predicted.png"
    )

    # Figure 8: Residual diagnostics
    create_residual_diagnostics(
        hybrid, paths["figures_dir"] / "fig8_residual_diagnostics.png"
    )

    # Figure 9: Prediction intervals for multiple locations (uncertainty)
    def plot_uncertainty_for_locations(
        model, predictions, output_path: Path, loc_ids: list[int] = None, time_labels=None
    ):
        """Plot observed/predicted with uncertainty bands for multiple locations in a 2x2 grid."""
        # Default to first 4 locations if not specified
        if loc_ids is None:
            loc_ids = list(range(min(4, model.n_locations_)))

        # Ensure we have exactly 4 locations for 2x2 grid
        if len(loc_ids) > 4:
            loc_ids = loc_ids[:4]
            logger.warning(f"Only plotting first 4 locations from provided list: {loc_ids}")
        elif len(loc_ids) < 4:
            # Pad with additional locations if needed
            available_locs = list(range(model.n_locations_))
            for loc in available_locs:
                if loc not in loc_ids and len(loc_ids) < 4:
                    loc_ids.append(loc)

        t_range = np.arange(model.n_times_)
        x_vals = t_range
        x_label = "Time step"
        if time_labels is not None and len(time_labels) == model.n_times_:
            x_vals = time_labels
            # Try formatting as dates if possible
            try:
                x_vals = pd.to_datetime(x_vals)
                x_label = "Date"
            except Exception:
                x_label = "Time"

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        # Define colors for better aesthetics
        colors = {
            'observed': '#2c3e50',      # Dark gray-blue
            'predicted': '#3498db',      # Bright blue
            'ci_fill': '#3498db',        # Same blue for CI
            'grid': '#bdc3c7'            # Light gray
        }

        for idx, loc_id in enumerate(loc_ids):
            ax = axes[idx]

            # Calculate statistics for this location
            obs_loc = model._y_matrix[:, loc_id]
            pred_loc = predictions.total[:, loc_id]
            mean_obs = np.nanmean(obs_loc)
            rmse_loc = np.sqrt(np.nanmean((obs_loc - pred_loc) ** 2))

            # Plot uncertainty band first (so it's behind)
            ax.fill_between(
                x_vals,
                predictions.lower[:, loc_id],
                predictions.upper[:, loc_id],
                alpha=0.2,
                color=colors['ci_fill'],
                label=f"{int(model.confidence_level * 100)}% CI",
                zorder=1
            )

            # Plot observed values
            ax.plot(
                x_vals,
                obs_loc,
                'o',
                color=colors['observed'],
                markersize=2.5,
                alpha=0.6,
                label="Observed",
                zorder=3
            )

            # Plot predicted values
            ax.plot(
                x_vals,
                pred_loc,
                '-',
                color=colors['predicted'],
                lw=1.5,
                label="Predicted",
                zorder=2
            )

            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel("NO₂ (µg/m³)", fontsize=10)
            ax.set_title(
                f"Location {loc_id}\nMean: {mean_obs:.1f} µg/m³ | RMSE: {rmse_loc:.2f}",
                fontsize=10,
                pad=10
            )
            ax.legend(loc='best', fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.25, linestyle='--', color=colors['grid'], linewidth=0.5)

            # Rotate x-axis labels if dates
            if isinstance(x_vals, pd.DatetimeIndex):
                ax.tick_params(axis='x', rotation=45)

        # Calculate overall statistics for subtitle
        overall_rmse = np.sqrt(np.nanmean((model._y_matrix - predictions.total) ** 2))
        overall_r2 = 1 - (np.nansum((model._y_matrix - predictions.total) ** 2) /
                         np.nansum((model._y_matrix - np.nanmean(model._y_matrix)) ** 2))

        fig.suptitle(
            f"GAM-SSM Predictions with Uncertainty Intervals\n"
            f"Overall RMSE: {overall_rmse:.2f} µg/m³ | R²: {overall_r2:.3f}",
            fontsize=13,
            fontweight='bold',
            y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved uncertainty plot for {len(loc_ids)} locations to {output_path}")

    # Select locations for uncertainty plot
    if args.uncertainty_locations:
        # Use user-specified locations
        diverse_loc_ids = [int(x.strip()) for x in args.uncertainty_locations.split(',')]
        logger.info(f"Using user-specified locations for uncertainty plot: {diverse_loc_ids}")
    else:
        # Auto-select 4 diverse locations based on mean NO2 levels
        # This gives us locations with different pollution characteristics
        mean_no2_per_location = np.nanmean(hybrid["model"]._y_matrix, axis=0)
        sorted_locs = np.argsort(mean_no2_per_location)

        # Pick locations at 25th, 50th, 75th percentiles and max
        n_locs = len(sorted_locs)
        diverse_loc_ids = [
            sorted_locs[int(n_locs * 0.25)],  # Low pollution
            sorted_locs[int(n_locs * 0.50)],  # Medium-low pollution
            sorted_locs[int(n_locs * 0.75)],  # Medium-high pollution
            sorted_locs[-1],                   # High pollution
        ]
        logger.info(f"Auto-selected diverse locations for uncertainty plot: {diverse_loc_ids}")

    plot_uncertainty_for_locations(
        hybrid["model"],
        hybrid["predictions"],
        paths["figures_dir"] / "fig9_uncertainty_timeseries.png",
        loc_ids=diverse_loc_ids,
        time_labels=time_labels,
    )

    # Figure 9b: Temporal Evolution (6 locations with uncertainty)
    plot_temporal_evolution(
        hybrid,
        paths["figures_dir"] / "fig9b_temporal_evolution.png",
        selected_locations=None,  # Auto-select 6 locations
        time_labels=time_labels
    )

    # Figure 9c: Spatial Patterns (observed vs predicted at key timesteps)
    plot_spatial_patterns(
        hybrid,
        paths["figures_dir"] / "fig9c_spatial_patterns.png",
        timesteps=None,  # Auto-select 4 timesteps
        observations_df=observations
    )

    # Figure 10: Spatial comparison maps (GAM-LUR vs GAM-SSM)
    create_spatial_comparison_map(
        baseline,
        hybrid,
        observations,
        paths["figures_dir"] / "fig10_spatial_comparison.png",
    )

    # Figure 11: SHAP feature importance for GAM-LUR
    create_shap_importance_plot(
        baseline,
        X_selected.values,
        y,
        selected_features,
        paths["figures_dir"] / "fig11_shap_importance.png",
    )

    # Figure 12: Residual hotspots (per-location RMSE)
    create_residual_hotspot_map(
        hybrid,
        baseline,
        time_idx,
        loc_idx,
        observations,
        paths["figures_dir"] / "fig12_residual_hotspots.png",
    )

    # Figure 13: Moran's I of residual means (optional deps)
    residual_means = (hybrid["model"]._y_matrix - hybrid["predictions"].total).mean(
        axis=0
    )
    lats = (
        observations[["location_id", "lat"]]
        .drop_duplicates()
        .set_index("location_id")
        .loc[hybrid["model"].location_ids_, "lat"]
        .values
    )
    lons = (
        observations[["location_id", "lon"]]
        .drop_duplicates()
        .set_index("location_id")
        .loc[hybrid["model"].location_ids_, "lon"]
        .values
    )
    create_morans_i_plot(
        residual_means,
        lats,
        lons,
        paths["figures_dir"] / "fig13_morans_i.png",
    )

    # Figure 14: Variogram of residual means (optional deps)
    create_variogram_plot(
        residual_means,
        lats,
        lons,
        paths["figures_dir"] / "fig14_variogram.png",
    )

    # =========================================================================
    # GRIDDED/INTERPOLATED SURFACE MAPS (FIGURES 15-18)
    # =========================================================================
    if args.skip_gridded_plots:
        logger.info(
            "Skipping gridded/temporal map outputs (Figures 15-18) per --skip-gridded-plots."
        )
        return

    logger.info("Creating gridded interpolated surface maps...")

    # Prepare data for gridded maps
    # Get coordinates for each location
    loc_meta = observations[["location_id", "lat", "lon"]].drop_duplicates()
    loc_meta = loc_meta.set_index("location_id").loc[hybrid["model"].location_ids_]
    coordinates = loc_meta[["lon", "lat"]].values
    unique_coords = np.unique(coordinates, axis=0)
    if unique_coords.shape[0] < 4:
        logger.warning(
            "Skipping gridded interpolation maps (Figures 15-18): "
            f"only {unique_coords.shape[0]} unique locations after subsampling; "
            "increase data volume or rerun without --max-records."
        )
    else:

        # Figure 15: Gridded three-panel comparison (time-averaged)
        # Average over time for spatial comparison
        y_obs_mean = hybrid["model"]._y_matrix.mean(axis=0)
        baseline_pred_matrix = np.full_like(hybrid["model"]._y_matrix, np.nan)

        # Rebuild baseline predictions in matrix form
        time_map = {t: i for i, t in enumerate(hybrid["model"].time_ids_)}
        loc_map = {l: i for i, l in enumerate(hybrid["model"].location_ids_)}
        for v, t, l in zip(
            baseline["predictions"],
            hybrid["model"]._time_index_train,
            hybrid["model"]._location_index_train,
        ):
            t_pos = time_map[t]
            l_pos = loc_map[l]
            baseline_pred_matrix[t_pos, l_pos] = v

        baseline_mean = np.nanmean(baseline_pred_matrix, axis=0)
        hybrid_mean = hybrid["predictions"].total.mean(axis=0)

        create_gridded_comparison(
            coordinates=coordinates,
            observed=y_obs_mean,
            baseline_pred=baseline_mean,
            hybrid_pred=hybrid_mean,
            output_path=paths["figures_dir"] / "fig15_gridded_comparison.png",
            title_suffix=" (Time-Averaged)",
        )
        logger.info(
            f"Saved gridded comparison to {paths['figures_dir'] / 'fig15_gridded_comparison.png'}"
        )

        # Figure 16: Gridded residual maps with performance metrics
        create_gridded_residual_map(
            coordinates=coordinates,
            observed=y_obs_mean,
            baseline_pred=baseline_mean,
            hybrid_pred=hybrid_mean,
            output_path=paths["figures_dir"] / "fig16_gridded_residuals.png",
        )
        logger.info(
            f"Saved gridded residuals to {paths['figures_dir'] / 'fig16_gridded_residuals.png'}"
        )

        # Figure 17: Uncertainty surface map
        # Get prediction uncertainty (averaged over time)
        uncertainty_matrix = (
            hybrid["predictions"].upper - hybrid["predictions"].lower
        ) / (
            2 * 1.96
        )  # Convert to SD
        mean_uncertainty = uncertainty_matrix.mean(axis=0)

        create_uncertainty_surface(
            coordinates=coordinates,
            predictions=hybrid_mean,
            std_dev=mean_uncertainty,
            output_path=paths["figures_dir"] / "fig17_uncertainty_surface.png",
        )
        logger.info(
            f"Saved uncertainty surface to {paths['figures_dir'] / 'fig17_uncertainty_surface.png'}"
        )

        # Figure 18: Temporal sequence of gridded maps
        # Select 5 time points across the dataset
        n_times = hybrid["model"].n_times_
        time_points = [0, n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]

        create_temporal_gridded_sequence(
            coordinates=coordinates,
            observed_matrix=hybrid["model"]._y_matrix,
            predicted_matrix=hybrid["predictions"].total,
            time_points=time_points,
            output_path=paths["figures_dir"] / "fig18_temporal_sequence.png",
        )
        logger.info(
            f"Saved temporal sequence to {paths['figures_dir'] / 'fig18_temporal_sequence.png'}"
        )

    # Save enriched predictions with metadata for downstream comparison
    loc_meta = (
        observations[["location_id", "lat", "lon"]]
        .drop_duplicates()
        .set_index("location_id")
    )
    lat_map = loc_meta["lat"].to_dict()
    lon_map = loc_meta["lon"].to_dict()

    # Align predictions to the observed training pairs (same length as baseline/y)
    time_map = {t: i for i, t in enumerate(hybrid["model"].time_ids_)}
    loc_map = {l: i for i, l in enumerate(hybrid["model"].location_ids_)}
    obs_indices = zip(
        hybrid["model"]._time_index_train, hybrid["model"]._location_index_train
    )
    pred_hybrid_obs = []
    pred_hybrid_lower_obs = []
    pred_hybrid_upper_obs = []
    for t_idx, l_idx in obs_indices:
        t_pos = time_map[t_idx]
        l_pos = loc_map[l_idx]
        pred_hybrid_obs.append(hybrid["predictions"].total[t_pos, l_pos])
        pred_hybrid_lower_obs.append(hybrid["predictions"].lower[t_pos, l_pos])
        pred_hybrid_upper_obs.append(hybrid["predictions"].upper[t_pos, l_pos])

    export_df = pd.DataFrame(
        {
            "timestamp": time_labels[hybrid["model"]._time_index_train],
            "location_id": hybrid["model"]._location_index_train,
            "epa_values": y,
            "pred_baseline": baseline["predictions"],
            "pred_hybrid": pred_hybrid_obs,
            "pred_hybrid_lower": pred_hybrid_lower_obs,
            "pred_hybrid_upper": pred_hybrid_upper_obs,
        }
    )
    export_df["residual_hybrid"] = export_df["epa_values"] - export_df["pred_hybrid"]
    export_df["latitude"] = export_df["location_id"].map(lat_map)
    export_df["longitude"] = export_df["location_id"].map(lon_map)

    if "no2_values" in features.columns:
        no2_map = features.set_index("location_id")["no2_values"].to_dict()
        export_df["no2_values"] = export_df["location_id"].map(no2_map)

    predictions_path = paths["tables_dir"] / "predictions_with_intervals.csv"
    export_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved enriched predictions to {predictions_path}")

    # Save model
    hybrid["model"].save(paths["models_dir"] / "hybrid_gam_ssm")

    logger.info("=" * 70)
    logger.info("Paper reproduction completed!")
    logger.info(f"Outputs saved to: {paths['output_dir']}")
    logger.info("=" * 70)


import time


if __name__ == "__main__":

    # Calculate and display the elapsed time
    # Record the start time
    start_time = time.time()

    print(f"Start Time: {start_time}")

    main()

    end_time = time.time()

    print(f"End Time: {end_time}")
    # Note: The calculation for total minutes is correct, but the unit string is a bit unusual.
    # A cleaner print would be just seconds, or a more precise unit conversion.
    print(f"Total Execution Time: {end_time - start_time} seconds")
    print(f"Total Execution Time: {(end_time - start_time) / 60} minutes")
