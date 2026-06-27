"""
Step 2 — Feature engineering:
clean, aggregate road totals, select/load features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gam_ssm_lur.data import StaticData
from gam_ssm_lur.features import FeatureSelector

logger = logging.getLogger(__name__)

ID_COLS = ["grid_id", "latitude", "longitude"]


def prepare_features(feat_df: pd.DataFrame, y_full, args, output_dir: Path):
    """Clean features, aggregate road_length totals, run or load feature selection.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Raw feature DataFrame including ID columns.
    y_full : array-like
        Target values aligned to feat_df rows.
    args : argparse.Namespace
        CLI arguments (corr_threshold, vif_threshold, n_features,
        skip_feature_selection).
    output_dir : Path
        Run output directory — selected_features.txt is written here.

    Returns
    -------
    X_df : pd.DataFrame
        Selected feature matrix (cells × features).
    static_sel : StaticData
        StaticData rebuilt with only selected features (target=None; set by caller).
    imp : pd.DataFrame or None
        Feature importances from FeatureSelector, or None when skipping selection.
    """
    feat_cols = [c for c in feat_df.columns if c not in ID_COLS]
    X_df = feat_df[feat_cols]

    # Drop non-numeric columns (e.g. geographic ID codes that leaked into features.csv)
    non_numeric = [
        c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])
    ]
    if non_numeric:
        logger.debug(
            "Dropping %d non-numeric columns (geographic IDs / strings): %s",
            len(non_numeric),
            non_numeric,
        )
        X_df = X_df.drop(columns=non_numeric)

    # Impute NaNs with column median
    nan_counts = X_df.isna().sum()
    if (nan_counts > 0).sum():
        logger.info(
            "Imputing NaN in %d feature columns (median fill)", (nan_counts > 0).sum()
        )
        X_df = X_df.fillna(X_df.median())

    # Aggregate sector-specific road_length columns into a single total per buffer.
    # The 8 directional (_s0–_s7) variants each carry ~1/8 of the road signal,
    # causing feature selection to drop all of them. Summing restores the full
    # road-proximity signal so high-NO2 corridors appear in spatial predictions.
    road_buffers = sorted(
        {
            c.rsplit("_s", 1)[0]
            for c in X_df.columns
            if c.startswith("road_length_") and "_s" in c
        }
    )
    totals = {}
    for base in road_buffers:
        sector_cols = [
            f"{base}_s{i}" for i in range(8) if f"{base}_s{i}" in X_df.columns
        ]
        if sector_cols:
            totals[f"{base}_total"] = X_df[sector_cols].sum(axis=1)
    if totals:
        X_df = pd.concat([X_df, pd.DataFrame(totals, index=X_df.index)], axis=1)
    n_road_totals = len(totals)
    logger.info("Added %d aggregated road_length_total columns", n_road_totals)
    logger.info(
        "After filtering: %d cells, %d raw features", len(X_df), len(X_df.columns)
    )

    selected_features_path = output_dir / "selected_features.txt"
    imp = None

    if not args.skip_feature_selection:
        selector = FeatureSelector(
            correlation_threshold=args.corr_threshold,
            vif_threshold=args.vif_threshold,
            importance_threshold=args.importance_threshold,
            random_state=42,
        )
        X_df = selector.fit_transform(X_df, y_full)
        logger.info("Feature selection: %d features selected", len(X_df.columns))
        imp = selector.result_.feature_importances
        imp.to_csv(output_dir / "feature_importance.csv", index=False)
        selected_features_path.write_text("\n".join(X_df.columns.tolist()))
    else:
        # Search both experiments/results/ (new) and results/ (legacy project root)
        project_root = Path(__file__).resolve().parent.parent
        prev_runs = sorted(
            [
                *project_root.glob("experiments/results/run_*/selected_features.txt"),
                *(project_root / "results").glob("run_*/selected_features.txt"),
            ]
        )
        if prev_runs:
            prev_path = prev_runs[-1]
            all_selected = prev_path.read_text().splitlines()
            selected = [f for f in all_selected if f in X_df.columns]
            missing = [f for f in all_selected if f not in X_df.columns]
            if missing:
                logger.warning(
                    "Features in selected_features.txt not found in data: %s", missing
                )
            if selected:
                X_df = X_df[selected]
                logger.info(
                    "Loaded %d selected features from %s", len(selected), prev_path
                )
            else:
                logger.warning(
                    "No matching features found in %s — using all %d features",
                    prev_path,
                    len(X_df.columns),
                )
        else:
            logger.warning(
                "No selected_features.txt found — using all %d features (memory risk!)",
                len(X_df.columns),
            )

    static_sel = StaticData(
        features=pd.concat(
            [feat_df[ID_COLS].reset_index(drop=True), X_df.reset_index(drop=True)],
            axis=1,
        ),
        target=None,  # caller must set static_sel.target = static.target
        grid_ids=list(feat_df["grid_id"].values),
    )
    return X_df, static_sel, imp
