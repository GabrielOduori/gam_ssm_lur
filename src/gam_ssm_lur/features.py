"""Feature selection and extraction utilities for land use regression."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inverse distance transformation
# ---------------------------------------------------------------------------


def inverse_distance_transform(
    df: pd.DataFrame,
    distance_cols: Optional[List[str]] = None,
    add_squared: bool = True,
    min_distance: float = 1.0,
    drop_raw: bool = False,
) -> pd.DataFrame:
    """Transform raw distance columns to 1/d (and optionally 1/d²) predictors.

    1/d is the physically appropriate form: closer sources → higher pollution.
    """
    df = df.copy()

    if distance_cols is None:
        distance_cols = [c for c in df.columns if "distance_to_" in c]

    for col in distance_cols:
        if col not in df.columns:
            logger.warning(
                "Column '%s' not found — skipping inverse distance transform", col
            )
            continue

        # Derive a clean base name: strip "distance_to_" prefix if present
        if col.startswith("distance_to_"):
            base = col[len("distance_to_") :]
        else:
            base = col

        d = df[col].clip(lower=min_distance)
        inv_col = f"{base}_inverse_distance"
        df[inv_col] = 1.0 / d

        if add_squared:
            sq_col = f"{base}_inverse_distance_sq"
            df[sq_col] = 1.0 / (d**2)

    if drop_raw:
        df = df.drop(columns=[c for c in distance_cols if c in df.columns])

    n_new = len(distance_cols) * (2 if add_squared else 1)
    logger.info(
        "inverse_distance_transform: added %d columns from %d distance cols",
        n_new,
        len(distance_cols),
    )
    return df


# ---------------------------------------------------------------------------
# Sparse cell filter
# ---------------------------------------------------------------------------


def filter_sparse_cells(
    features: pd.DataFrame,
    target: pd.Series,
    min_nonzero_features: int = 1,
    drop_zero_target: bool = True,
    id_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove cells with < min_nonzero_features non-zero predictors (OpenLUR style)."""
    if id_cols is None:
        id_cols = ["grid_id", "latitude", "longitude"]

    predictor_cols = [c for c in features.columns if c not in id_cols]
    X_pred = features[predictor_cols].fillna(0)

    mask = pd.Series(True, index=features.index)

    if drop_zero_target:
        zero_target = target.isna() | (target <= 0)
        n_zero = zero_target.sum()
        if n_zero:
            logger.info("filter_sparse_cells: dropping %d zero/NaN target rows", n_zero)
        mask &= ~zero_target

    nonzero_count = (X_pred != 0).sum(axis=1)
    sparse_mask = nonzero_count < min_nonzero_features
    n_sparse = sparse_mask.sum()
    if n_sparse:
        logger.info(
            "filter_sparse_cells: dropping %d cells with < %d non-zero predictors",
            n_sparse,
            min_nonzero_features,
        )
    mask &= ~sparse_mask

    n_in = len(features)
    n_out = mask.sum()
    logger.info(
        "filter_sparse_cells: %d → %d cells retained (%.1f%%)",
        n_in,
        n_out,
        100 * n_out / n_in,
    )
    return features[mask].reset_index(drop=True), target[mask].reset_index(drop=True)


@dataclass
class SelectionResult:
    """Results from the three-stage feature selection pipeline."""

    selected_features: List[str]
    n_original: int
    n_selected: int
    dropped_correlation: List[str]
    dropped_vif: List[str]
    dropped_importance: List[str]
    feature_importances: pd.DataFrame


class FeatureSelector:
    """Three-stage feature selector: correlation → VIF → cumulative RF importance."""

    def __init__(
        self,
        correlation_threshold: float = 0.8,
        vif_threshold: float = 10.0,
        importance_threshold: float = 0.95,
        force_keep: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.importance_threshold = importance_threshold
        self.force_keep = set(force_keep) if force_keep else set()
        self.random_state = random_state

        self.result_: Optional[SelectionResult] = None
        self.selected_columns_: Optional[List[str]] = None
        self._is_fitted = False

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Union[NDArray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> FeatureSelector:
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        elif isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            raise TypeError("X must be a numpy array or pandas DataFrame")

        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y)

        n_original = len(feature_names)
        logger.info(f"Starting feature selection with {n_original} features")

        # Track dropped features
        dropped_corr: List[str] = []
        dropped_vif: List[str] = []
        dropped_imp: List[str] = []

        # Stage 1: Correlation-based removal
        #
        # current_features/dropped_corr are kept as lists ordered by the
        # original feature_names, never a bare set -- set() iteration order
        # for strings is hash-randomised per Python process (PYTHONHASHSEED),
        # so list(some_set) silently gave a different column order on every
        # run. That order then determined which feature among VIF ties got
        # dropped first in Stage 2, cascading into a different *count* of
        # selected features run-to-run (30-32 on identical data), not just
        # different identities. to_remove itself is fine as a set: it's only
        # ever used for membership testing (`in`), never iterated for order.
        logger.info("Stage 1: Correlation-based filtering")
        corr_matrix = X.corr().abs()

        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]

                    # Don't remove force-keep features
                    if col_i in self.force_keep and col_j in self.force_keep:
                        continue
                    elif col_i in self.force_keep:
                        to_remove.add(col_j)
                    elif col_j in self.force_keep:
                        to_remove.add(col_i)
                    else:
                        # Remove feature with lower variance
                        if X[col_i].var() < X[col_j].var():
                            to_remove.add(col_i)
                        else:
                            to_remove.add(col_j)

        dropped_corr = [f for f in feature_names if f in to_remove]
        current_features = [f for f in feature_names if f not in to_remove]
        logger.info(
            "  Removed %d correlated features, %d remaining",
            len(dropped_corr),
            len(current_features),
        )

        # Stage 2: VIF filtering
        logger.info("Stage 2: VIF filtering")
        X_current = X[current_features].copy()

        while True:
            vif_values = self._compute_vif(X_current)
            max_vif_idx = vif_values.argmax()
            max_vif = vif_values[max_vif_idx]

            if max_vif <= self.vif_threshold:
                break

            feature_to_remove = X_current.columns[max_vif_idx]

            # Don't remove force-keep features
            if feature_to_remove in self.force_keep:
                # Remove next highest VIF that's not force-keep
                sorted_idx = np.argsort(vif_values)[::-1]
                for idx in sorted_idx:
                    candidate = X_current.columns[idx]
                    if candidate not in self.force_keep:
                        feature_to_remove = candidate
                        break
                else:
                    # All remaining features are force-keep
                    break

            dropped_vif.append(feature_to_remove)
            X_current = X_current.drop(columns=[feature_to_remove])

            if len(X_current.columns) <= 5:  # safety floor
                break

        current_features = list(X_current.columns)  # order preserved through .drop()
        logger.info(
            "  Removed %d high-VIF features, %d remaining",
            len(dropped_vif),
            len(current_features),
        )

        # Re-add force-keep features if removed, preserving the original
        # feature_names order rather than appending (which would make output
        # order depend on force_keep's own set iteration order)
        current_set = set(current_features)
        for feat in self.force_keep:
            if feat in feature_names and feat not in current_set:
                current_set.add(feat)
                if feat in dropped_vif:
                    dropped_vif.remove(feat)
        current_features = [f for f in feature_names if f in current_set]

        # Stage 3: Importance-based selection
        logger.info("Stage 3: Importance-based selection")
        X_current = X[current_features].copy()

        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_current, y)

        importances = pd.DataFrame(
            {
                "feature": X_current.columns,
                "importance": rf.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Cumulative importance threshold — keep the minimum set of features
        # whose cumulative RF importance reaches importance_threshold.
        # Defensible in publication: "features were retained until cumulative
        # RF importance reached {threshold*100:.0f}% of total explained variance."
        importances["cumulative"] = importances["importance"].cumsum()
        total = importances["importance"].sum()
        importances["cumulative_frac"] = importances["cumulative"] / total

        # Find cutoff index (first row where cumulative fraction >= threshold)
        cutoff = (importances["cumulative_frac"] >= self.importance_threshold).idxmax()
        cutoff_pos = importances.index.get_loc(cutoff)
        selected_by_threshold = set(importances.iloc[: cutoff_pos + 1]["feature"])

        # Always include force-keep features
        top_features = selected_by_threshold | (self.force_keep & set(current_features))

        dropped_imp = [f for f in current_features if f not in top_features]
        selected_features = [f for f in current_features if f in top_features]

        pct = (
            importances.loc[
                importances["feature"].isin(selected_features), "importance"
            ].sum()
            / total
            * 100
        )
        logger.info(
            "  Selected %d features accounting for %.1f%% of total RF importance "
            "(threshold=%.0f%%)",
            len(selected_features),
            pct,
            self.importance_threshold * 100,
        )

        # Store results
        self.selected_columns_ = selected_features
        self.result_ = SelectionResult(
            selected_features=selected_features,
            n_original=n_original,
            n_selected=len(selected_features),
            dropped_correlation=dropped_corr,
            dropped_vif=dropped_vif,
            dropped_importance=dropped_imp,
            feature_importances=importances,
        )

        self._is_fitted = True
        return self

    def transform(
        self,
        X: Union[NDArray, pd.DataFrame],
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.result_.selected_features)
        elif isinstance(X, pd.DataFrame):
            # Check that required columns exist
            missing = set(self.selected_columns_) - set(X.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")

        return X[self.selected_columns_]

    def fit_transform(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Union[NDArray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        self.fit(X, y, feature_names)
        return self.transform(X)

    def _compute_vif(self, X: pd.DataFrame) -> NDArray:
        """Compute Variance Inflation Factor for each feature."""
        from sklearn.linear_model import LinearRegression

        vif = np.zeros(len(X.columns))

        for i, col in enumerate(X.columns):
            # Regress feature i on all other features
            y_i = X[col].values
            X_others = X.drop(columns=[col]).values

            if X_others.shape[1] == 0:
                vif[i] = 1.0
                continue

            lr = LinearRegression()
            lr.fit(X_others, y_i)
            r_squared = lr.score(X_others, y_i)

            # VIF = 1 / (1 - R²)
            if r_squared >= 1.0:
                vif[i] = np.inf
            else:
                vif[i] = 1.0 / (1.0 - r_squared)

        return vif

    def get_summary(self) -> str:
        """Return a text summary of feature selection results."""
        if not self._is_fitted:
            return "Selector not fitted."

        r = self.result_
        lines = [
            "Feature Selection Summary",
            "=" * 40,
            f"Original features: {r.n_original}",
            f"Selected features: {r.n_selected}",
            "",
            "Dropped due to correlation:",
            f"  {len(r.dropped_correlation)} features",
            "",
            "Dropped due to high VIF:",
            f"  {len(r.dropped_vif)} features",
            "",
            "Dropped due to low importance:",
            f"  {len(r.dropped_importance)} features",
            "",
            "Top 10 selected features by importance:",
        ]

        for _, row in r.feature_importances.head(10).iterrows():
            lines.append(f"  {row['feature']}: {row['importance']:.4f}")

        return "\n".join(lines)
