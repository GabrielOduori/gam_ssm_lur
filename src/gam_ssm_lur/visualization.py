"""
Visualization Module for GAM-SSM-LUR.

Produces publication-ready figures using the same visual language as the
Dublin reference implementation:

  - Spatial maps rendered on polygon grid (geopandas) with RdYlGn_r colormap
    and lightgrey fill for missing cells; falls back to scatter when no grid
    GeoDataFrame is available.
  - Residual/diverging maps use RdBu_r.
  - Time series use steelblue (LUR prior / satellite markers) and darkorange
    (SSM line + uncertainty shading) throughout.
  - Alpha = 0.8 on all spatial fills.
  - R² and correlation annotated directly on scatter panels.

Main classes
------------
SpatialVisualizer
    Polygon or scatter maps: NO2 surface, residuals, uncertainty, model
    comparison (OLS / GAM / GAM-SSM three-panel), selected daily snapshots.
TemporalVisualizer
    Station time series (3×3 panel), temporal variance evolution.
ModelComparisonVisualizer
    LOOCV scatter, model-vs-model accuracy table.
DiagnosticsVisualizer
    Residual histogram, Q-Q, ACF, heteroscedasticity panel.

References
----------
.. [1] Naughton, O., et al. (2018). A land use regression model for explaining
       spatial variation in air pollution. Science of the Total Environment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import linregress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

# Colour constants
CMAP_RESIDUAL = "RdBu_r"    # residual / diverging maps
CMAP_ERROR    = "YlOrRd"    # absolute error maps

# NO2 concentration maps — fixed breakpoints matching Atmos-Street 2023
# Dublin range: ~5–25 µg/m³ (Atmos-Street 2023 full range 2–57 µg/m³)
# Colours: light yellow → amber → orange → red → dark red (EEA-style)
NO2_BOUNDS = [0, 5, 8, 10, 12, 15, 18, 22, 28, 40]   # µg/m³ class edges
NO2_COLOURS = [
    "#ffffcc",  # 0–5    very low
    "#c7e9b4",  # 5–8    low
    "#7fcdbb",  # 8–10
    "#41b6c4",  # 10–12
    "#1d91c0",  # 12–15  WHO guideline (10 µg/m³ annual)
    "#225ea8",  # 15–18
    "#253494",  # 18–22
    "#081d58",  # 22–28  EU limit (40 µg/m³ annual)
    "#4d004b",  # 28–40+
]

def _no2_cmap_norm():
    """Return (ListedColormap, BoundaryNorm) for the fixed NO2 colour scheme."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(NO2_COLOURS, name="no2_atmos")
    norm = BoundaryNorm(NO2_BOUNDS, cmap.N)
    return cmap, norm

CMAP_NO2 = "no2_atmos"   # placeholder name — use _no2_cmap_norm() for actual cmap/norm
COL_LUR       = "steelblue"  # GAM LUR prior lines / satellite markers
COL_SSM       = "darkorange" # GAM-SSM lines / uncertainty shading
COL_OBS       = "black"      # observed station measurements
ALPHA_MAP     = 0.8
ALPHA_SHADE   = 0.2
MISSING_KWD   = {"color": "lightgrey", "alpha": ALPHA_MAP}

# Per-station colours for LOOCV scatter
_STATION_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974",
]


def _save(fig: plt.Figure, path: Optional[Union[str, Path]], dpi: int = 150) -> None:
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved → %s", path)


def _annotate_r2(ax: plt.Axes, y_true: NDArray, y_pred: NDArray) -> None:
    """Add R² annotation in the lower-right corner of an axes."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ax.annotate(
        f"R² = {r2:.3f}",
        xy=(0.95, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


# ---------------------------------------------------------------------------
# Spatial Visualizer
# ---------------------------------------------------------------------------

class SpatialVisualizer:
    """Maps of spatial pollution distributions.

    Uses a GeoDataFrame polygon grid when available (preferred), otherwise
    falls back to a scatter plot.

    Parameters
    ----------
    grid_gdf : GeoDataFrame, optional
        Polygon grid with a ``grid_id`` column. Pass the output of
        ``SpatiotemporalDataset.load_grid_geometry()``.
    grid_ids : array-like, optional
        Ordered sequence of grid cell IDs matching the value arrays passed
        to plotting methods.  Required when using polygon rendering.
    """

    def __init__(
        self,
        grid_gdf=None,
        grid_ids: Optional[List[str]] = None,
    ):
        self.grid_gdf = grid_gdf
        self.grid_ids = list(grid_ids) if grid_ids is not None else None

    # ------------------------------------------------------------------ #
    # Core map primitive                                                   #
    # ------------------------------------------------------------------ #

    def _map_ax(
        self,
        ax: plt.Axes,
        values: NDArray,
        grid_ids: Optional[List[str]] = None,
        cmap: str = CMAP_NO2,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        legend: bool = True,
        legend_label: str = "NO₂ (µg/m³)",
    ) -> None:
        """Render values onto *ax* using polygon grid or scatter fallback."""
        values = np.asarray(values)
        gids = grid_ids or self.grid_ids

        # Use the fixed Atmos-Street NO2 colour scheme for all NO2 maps.
        # Diverging/error maps keep their own cmap and a continuous scale.
        if cmap == CMAP_NO2:
            cmap_obj, norm = _no2_cmap_norm()
            vmin = NO2_BOUNDS[0]
            vmax = NO2_BOUNDS[-1]
        else:
            cmap_obj = cmap
            norm = None

        if self.grid_gdf is not None and gids is not None:
            data = pd.DataFrame({"grid_id": gids, "_val": values})
            merged = self.grid_gdf.merge(data, on="grid_id", how="left")
            legend_kwds = {"label": legend_label, "shrink": 0.85} if legend else {}
            if norm is not None:
                legend_kwds["extend"] = "max"
            merged.plot(
                column="_val", ax=ax,
                cmap=cmap_obj, vmin=vmin, vmax=vmax,
                norm=norm,
                alpha=ALPHA_MAP, legend=legend,
                legend_kwds=legend_kwds,
                missing_kwds=MISSING_KWD,
                edgecolor="none",
            )
        else:
            # Scatter fallback — requires grid_gdf with lat/lon or external coords
            if self.grid_gdf is not None and "geometry" in self.grid_gdf.columns:
                cx = self.grid_gdf.geometry.centroid.x.values
                cy = self.grid_gdf.geometry.centroid.y.values
            else:
                logger.warning("No grid geometry available — cannot render map")
                return
            sc = ax.scatter(cx, cy, c=values, cmap=cmap_obj, s=3,
                            alpha=ALPHA_MAP, vmin=vmin, vmax=vmax, norm=norm)
            if legend:
                plt.colorbar(sc, ax=ax, label=legend_label, shrink=0.85,
                             extend="max" if norm is not None else "neither")

        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # Single surface map                                                   #
    # ------------------------------------------------------------------ #

    def plot_surface(
        self,
        values: NDArray,
        title: str = "",
        cmap: str = "turbo",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        legend_label: str = "NO₂ (µg/m³)",
        ax: Optional[plt.Axes] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Axes:
        """Plot a single spatial surface.

        Parameters
        ----------
        values : array-like, shape (n_cells,)
            Values at each grid cell.
        title : str
            Axes title.
        cmap : str
            Matplotlib colormap name.
        vmin, vmax : float, optional
            Colorbar limits.  Defaults to 2nd/98th percentile.
        legend_label : str
            Colorbar label.
        ax : Axes, optional
            Existing axes to plot on.
        save_path : str or Path, optional
            Path to save figure.
        """
        values = np.asarray(values)
        # Anchor vmin=0 so background (low-NO2) cells render dark, making
        # road corridors glow against a near-black field (paper Fig 5/6 style).
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = float(np.nanpercentile(values, 98))

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 7))

        self._map_ax(ax, values, cmap=cmap, vmin=vmin, vmax=vmax,
                     legend_label=legend_label)
        if title:
            ax.set_title(title, fontsize=11, fontweight="bold")

        if standalone:
            plt.tight_layout()
            _save(ax.figure, save_path)

        return ax

    # ------------------------------------------------------------------ #
    # Three-panel static comparison                                        #
    # ------------------------------------------------------------------ #

    def plot_static_comparison(
        self,
        panels: List[Tuple[NDArray, str]],
        suptitle: str = "",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Side-by-side comparison of up to 4 spatial fields.

        Designed for OLS LUR | GAM LUR | GAM-SSM mean (3 panels), but
        accepts any number ≤ 4.

        Parameters
        ----------
        panels : list of (values, label) tuples
            Each tuple is an array of shape (n_cells,) and a panel title.
        suptitle : str
            Figure super-title.
        save_path : str or Path, optional
            Path to save figure.

        Returns
        -------
        fig : Figure
        """
        n = len(panels)
        all_vals = np.concatenate([p[0] for p in panels])
        vmin = 0.0
        vmax = float(np.nanpercentile(all_vals, 98))
        vmax = round(vmax / 5) * 5 or vmax

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]

        for ax, (vals, label) in zip(axes, panels):
            self._map_ax(ax, vals, cmap="turbo", vmin=vmin, vmax=vmax,
                         legend=True, legend_label="NO₂ (µg/m³)")
            ax.set_title(label, fontsize=11, fontweight="bold")

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=1.01)
        plt.tight_layout()
        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------ #
    # Daily snapshot maps                                                  #
    # ------------------------------------------------------------------ #

    def plot_daily_snapshots(
        self,
        ssm_df: pd.DataFrame,
        dates: List,
        date_labels: Optional[Dict] = None,
        date_col: str = "date",
        value_col: str = "no2",
        grid_id_col: str = "grid_id",
        forcing_col: Optional[str] = "delta_traffic",
        satellite_col: Optional[str] = "has_satellite",
        suptitle: str = "",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """2×N grid of daily SSM maps for selected dates.

        Title of each panel shows the date label, traffic forcing Δ, and a
        ▲SAT tag when a satellite overpass updated the filter.

        Parameters
        ----------
        ssm_df : pd.DataFrame
            SSM output with at least ``grid_id``, ``date``, value column.
        dates : list
            Dates to display (up to 4).
        date_labels : dict, optional
            Mapping from date to display label string.
        forcing_col : str, optional
            Column name for the traffic anomaly scalar.  Set to None to omit.
        satellite_col : str, optional
            Boolean column indicating satellite update days.  Set to None to omit.
        save_path : str or Path, optional
            Path to save figure.

        Returns
        -------
        fig : Figure
        """
        n = len(dates)
        ncols = min(n, 2)
        nrows = (n + 1) // 2

        vmin = 0.0
        vmax = float(ssm_df[value_col].quantile(0.98))
        vmax = round(vmax / 5) * 5 or vmax

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for ax, d in zip(axes, dates):
            day = ssm_df[ssm_df[date_col] == d]
            if day.empty:
                ax.set_visible(False)
                continue

            vals = day.set_index(grid_id_col)[value_col]
            gids = list(vals.index)
            self._map_ax(ax, vals.values, grid_ids=gids,
                         cmap="turbo", vmin=vmin, vmax=vmax)

            # Build title
            base = date_labels.get(d, str(d)) if date_labels else str(d)
            extras = ""
            if forcing_col and forcing_col in day.columns:
                dt = float(day[forcing_col].iloc[0])
                extras += f"  Δtraffic={dt:+.2f}"
            if satellite_col and satellite_col in day.columns:
                if bool(day[satellite_col].iloc[0]):
                    extras += "  ▲SAT"
            ax.set_title(f"GAM-SSM  {base}{extras}", fontsize=9, fontweight="bold")

        # Hide leftover axes
        for ax in axes[n:]:
            ax.set_visible(False)

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=1.01)
        plt.tight_layout()
        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------ #
    # Spatial residuals                                                    #
    # ------------------------------------------------------------------ #

    def plot_residuals(
        self,
        residuals: NDArray,
        predictions: Optional[NDArray] = None,
        grid_ids: Optional[List[str]] = None,
        title: str = "Spatial Residuals",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """GAM prediction map + signed residual map + absolute error map.

        If ``predictions`` is supplied the figure has three panels:
        (a) GAM prediction, (b) signed residuals, (c) absolute error.
        Otherwise only panels (b) and (c) are shown.
        """
        res = np.asarray(residuals)
        lim = float(np.nanpercentile(np.abs(res), 95))

        ncols = 3 if predictions is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6),
                                 constrained_layout=True)

        col = 0
        if predictions is not None:
            pred = np.asarray(predictions)
            vmax_pred = float(np.nanpercentile(pred, 98))
            self._map_ax(axes[col], pred, grid_ids=grid_ids,
                         cmap="turbo", vmin=0, vmax=vmax_pred,
                         legend_label="NO₂ (µg/m³)")
            axes[col].set_title("(a) GAM Spatial Prediction",
                                fontsize=11, fontweight="bold")
            col += 1

        label_b = "(b)" if predictions is not None else "(a)"
        label_c = "(c)" if predictions is not None else "(b)"

        self._map_ax(axes[col], res, grid_ids=grid_ids,
                     cmap=CMAP_RESIDUAL, vmin=-lim, vmax=lim,
                     legend_label="Residual (µg/m³)")
        axes[col].set_title(f"{label_b} Signed Residuals  (obs − predicted)",
                            fontsize=11, fontweight="bold")
        col += 1

        self._map_ax(axes[col], np.abs(res), grid_ids=grid_ids,
                     cmap=CMAP_ERROR, vmin=0, vmax=lim,
                     legend_label="|Residual| (µg/m³)")
        axes[col].set_title(f"{label_c} Absolute Error",
                            fontsize=11, fontweight="bold")

        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")
        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------ #
    # Feature importance                                                   #
    # ------------------------------------------------------------------ #

    def plot_partial_dependence(
        self,
        gam_model,
        n_top: int = 9,
        ncols: int = 3,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Grid of partial response curves for top GAM smooth terms.

        Shows the marginal effect of each predictor on NO₂ with 95%
        confidence intervals, as returned by pygam's partial_dependence().

        Parameters
        ----------
        gam_model : SpatialGAM
            Fitted SpatialGAM instance.
        n_top : int
            Number of features to plot (by GAM feature index order).
        ncols : int
            Number of columns in the panel grid.
        save_path : str or Path, optional
            Path to save figure.
        """
        n_features = len(gam_model.feature_names_)
        n_plot = min(n_top, n_features)
        nrows = (n_plot + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 3.5 * nrows),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()

        for i in range(n_plot):
            ax = axes[i]
            pd_result = gam_model.partial_dependence(i)
            ax.plot(pd_result.grid, pd_result.response,
                    color=COL_LUR, lw=2)
            ax.fill_between(pd_result.grid,
                            pd_result.confidence_lower,
                            pd_result.confidence_upper,
                            alpha=ALPHA_SHADE, color=COL_LUR)
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_title(pd_result.feature_name, fontsize=8, fontweight="bold")
            ax.set_xlabel("Feature value", fontsize=7)
            ax.set_ylabel("Partial effect", fontsize=7)
            ax.tick_params(labelsize=7)

        for ax in axes[n_plot:]:
            ax.set_visible(False)

        fig.suptitle("GAM Partial Response Curves", fontsize=12, fontweight="bold")
        _save(fig, save_path)
        return fig

    def plot_feature_importance(
        self,
        importances: pd.DataFrame,
        n_top: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Horizontal bar chart of top feature importances.

        Parameters
        ----------
        importances : pd.DataFrame
            DataFrame with columns ``feature`` and ``importance``, sorted
            descending (as returned by ``SpatialGAM.get_feature_importance``).
        """
        top = importances.head(n_top).copy()
        top = top.sort_values("importance")   # ascending for horizontal bar

        fig, ax = plt.subplots(figsize=(8, 0.35 * n_top + 1.5))
        ax.barh(top["feature"], top["importance"], color=COL_LUR, alpha=0.8)
        ax.set_xlabel("Importance", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        plt.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_shap_summary(
        self,
        gam_model,
        X_df: pd.DataFrame,
        n_top: int = 15,
        n_background: int = 100,
        n_explain: int = 500,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """SHAP beeswarm + bar summary for the GAM spatial component.

        Uses ``shap.KernelExplainer`` on the GAM predict function.  A random
        background sample sets the baseline; a random explain sample provides
        per-cell SHAP values.  Two panels are produced:

        - **(a) Beeswarm** — each dot is one grid cell; x-axis = SHAP value
          (impact on NO₂ prediction); colour = feature value (blue=low,
          red=high).  Features ranked by mean |SHAP|.
        - **(b) Bar** — mean |SHAP| per feature; publication-ready importance
          chart.

        Parameters
        ----------
        gam_model : SpatialGAM
            Fitted model with ``feature_names_`` and a ``predict()`` method.
        X_df : pd.DataFrame
            Feature matrix (columns = ``gam_model.feature_names_``).
        n_top : int
            Number of top features to show.
        n_background : int
            Size of background sample for KernelExplainer baseline.
        n_explain : int
            Number of cells to compute SHAP values for (sub-sampled for speed).
        save_path : str or Path, optional
            Path to save the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import shap
            logging.getLogger("shap").setLevel(logging.WARNING)
        except ImportError:
            logger.warning("shap not installed — skipping SHAP plot")
            return None

        feat_cols = list(gam_model.feature_names_)
        X = X_df[feat_cols].values.astype(float)

        rng = np.random.default_rng(42)
        bg_idx  = rng.choice(len(X), size=min(n_background, len(X)), replace=False)
        exp_idx = rng.choice(len(X), size=min(n_explain,    len(X)), replace=False)

        X_bg  = X[bg_idx]
        X_exp = X[exp_idx]

        explainer   = shap.KernelExplainer(gam_model.predict, X_bg)
        shap_values = explainer.shap_values(X_exp, silent=True)   # (n_exp, n_feat)

        shap_df   = pd.DataFrame(shap_values, columns=feat_cols)
        mean_abs  = shap_df.abs().mean().sort_values(ascending=False)
        top_feats = mean_abs.head(n_top).index.tolist()

        shap_top = shap_df[top_feats].values
        X_top    = pd.DataFrame(X_exp, columns=feat_cols)[top_feats].values

        # Human-readable feature labels
        _SECTOR_NAMES = {
            "_s0": " (N)", "_s1": " (NE)", "_s2": " (E)", "_s3": " (SE)",
            "_s4": " (S)", "_s5": " (SW)", "_s6": " (W)", "_s7": " (NW)",
        }
        _REPLACEMENTS = [
            ("road_length_",           "Road length "),
            ("_total",                 " – all sectors"),
            ("scats_intensity",        "Traffic intensity"),
            ("scats_inverse_distance", "Traffic proximity"),
            ("scats_inverse_distance_sq", "Traffic proximity²"),
            ("nearest_scats_distance", "Nearest traffic sensor"),
            ("distance_to_scats",      "Distance to traffic sensor"),
            ("distance_to_motorway",   "Distance to motorway"),
            ("motorway_inverse_distance", "Motorway proximity"),
            ("distance_to_industrial", "Distance to industrial"),
            ("traffic_signals_inverse_distance_sq", "Traffic signal proximity²"),
            ("traffic_signals_inverse_distance",    "Traffic signal proximity"),
            ("landuse_commercial_area_", "Commercial area "),
            ("landuse_industrial_area_", "Industrial area "),
            ("building_commercial_area_", "Commercial buildings "),
            ("population_density_km2", "Population density"),
            ("elevation_m",            "Elevation"),
            ("m_s",                    "m "),
        ]

        def _fmt(name: str) -> str:
            label = name
            for suffix, direction in _SECTOR_NAMES.items():
                if label.endswith(suffix):
                    label = label[: -len(suffix)] + direction
                    break
            for old, new in _REPLACEMENTS:
                label = label.replace(old, new)
            return label.strip()

        labels = [_fmt(f) for f in top_feats]

        fig, ax = plt.subplots(figsize=(9, 0.45 * n_top + 2), constrained_layout=True)
        cmap_shap   = plt.cm.RdBu_r
        y_positions = np.arange(n_top)

        for i in range(n_top):
            sv   = shap_top[:, i]
            fv   = X_top[:, i]
            norm = plt.Normalize(vmin=np.percentile(fv, 5),
                                 vmax=np.percentile(fv, 95))
            colours = cmap_shap(norm(fv))
            jitter  = rng.uniform(-0.3, 0.3, size=len(sv))
            ax.scatter(sv, y_positions[i] + jitter,
                       c=colours, s=6, alpha=0.6, linewidths=0)

        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("SHAP value — contribution to predicted NO₂ (µg/m³)", fontsize=9)
        ax.set_title("SHAP Feature Importance — GAM Spatial Component",
                     fontsize=10, fontweight="bold")

        sm = plt.cm.ScalarMappable(cmap=cmap_shap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01)
        cbar.set_label("Feature value", fontsize=7)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"], fontsize=7)

        _save(fig, save_path)
        return fig

    def plot_wind_sector_map(
        self,
        gam_model,
        X_df: pd.DataFrame,
        sector_names: Optional[List[str]] = None,
        n_sectors: int = 8,
        ncols: int = 4,
        shared_scale: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Panel of GAM spatial NO₂ maps — one per dominant wind sector.

        For each of the *n_sectors* wind directions the map shows the GAM
        prediction when only that sector's directional OSM features are
        active (columns ending ``_sN`` for the current sector N are kept at
        their observed values; all other ``_sK`` columns are replaced by
        their column mean so they contribute only their average effect).
        Non-sectored features (elevation, population, etc.) are always kept
        at observed values.

        This illustrates how the spatial NO₂ pattern shifts as the dominant
        wind direction changes — e.g. a SW wind activates upwind industrial
        and road features to the SW of each cell, pushing the high-NO₂
        footprint north-eastward.

        Parameters
        ----------
        gam_model : SpatialGAM
            Fitted model (must have ``feature_names_`` and ``_X_train``).
        X_df : pd.DataFrame
            Feature matrix as a DataFrame whose column names match
            ``gam_model.feature_names_``.  Typically ``model._X_train``
            wrapped in a DataFrame.
        sector_names : list of str, optional
            Labels for sectors 0–7.  Defaults to
            ``['N','NE','E','SE','S','SW','W','NW']``.
        n_sectors : int
            Number of sectors (must match the ``_sN`` suffixes in the data).
        ncols : int
            Columns in the figure panel.
        shared_scale : bool
            If True all panels share the same colour axis (easier comparison).
        save_path : str or Path, optional
            Path to save figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if sector_names is None:
            _defaults = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            sector_names = _defaults[:n_sectors]

        feat_cols = list(gam_model.feature_names_)
        X_base = X_df[feat_cols].copy()

        # Identify which columns belong to each sector suffix
        sector_cols: Dict[int, List[str]] = {s: [] for s in range(n_sectors)}
        for col in feat_cols:
            for s in range(n_sectors):
                if col.endswith(f"_s{s}"):
                    sector_cols[s].append(col)

        # Pre-compute column means for all sector columns (used as neutral value)
        all_sector_feat = [c for s in sector_cols.values() for c in s]
        col_means = X_base[all_sector_feat].mean() if all_sector_feat else pd.Series(dtype=float)

        # Predict a surface per sector
        sector_preds: List[NDArray] = []
        for s in range(n_sectors):
            X_s = X_base.copy()
            # Zero out (mean-replace) all other sectors
            for other_s, other_cols in sector_cols.items():
                if other_s != s:
                    for col in other_cols:
                        if col in col_means.index:
                            X_s[col] = col_means[col]
            sector_preds.append(gam_model.predict(X_s.values))

        # Use data-driven colour range across all sector predictions
        all_vals = np.concatenate(sector_preds)
        vmin = 0.0
        vmax = float(np.nanpercentile(all_vals, 98))

        nrows = (n_sectors + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3 * ncols, 3 * nrows),  # 12×6 for 2×4 — fits on one page
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).flatten()

        for s in range(n_sectors):
            ax = axes[s]
            self._map_ax(
                ax, sector_preds[s],
                cmap="turbo", vmin=vmin, vmax=vmax,
                legend=(s == n_sectors - 1),  # colorbar on last panel only
            )
            ax.set_title(
                f"Dominant wind: {sector_names[s]}",
                fontsize=9, fontweight="bold",
            )

        for ax in axes[n_sectors:]:
            ax.set_visible(False)

        fig.suptitle(
            "GAM Spatial NO₂ Prediction by Dominant Wind Sector",
            fontsize=12, fontweight="bold",
        )
        _save(fig, save_path)
        return fig


    def plot_gam_with_wind_rose(
        self,
        gam_values: NDArray,
        wind_df: pd.DataFrame,
        n_sectors: int = 8,
        title: str = "GAM-LUR Annual Mean NO₂ with ERA5 Wind Rose",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """GAM spatial NO₂ map with an ERA5 wind rose inset.

        The main panel shows the GAM spatial prediction across the grid.
        An inset polar axes in the upper-right corner shows the wind rose:
        bar length = sector frequency, bar colour = mean wind speed.

        This figure pairs the spatial NO₂ distribution with the prevailing
        wind climatology, explaining which upwind sectors drive concentrations
        in different parts of Dublin.

        Parameters
        ----------
        gam_values : NDArray
            Per-cell GAM predictions (same ordering as ``self.grid_ids``).
        wind_df : pd.DataFrame
            ERA5 wind sector data with columns
            ``wind_sector_N_freq`` and ``wind_sector_N_mean_speed``
            for N in 0..n_sectors-1.
        n_sectors : int
            Number of wind sectors (default 8).
        title : str
            Figure title.
        save_path : str or Path, optional
            Path to save figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        sector_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][:n_sectors]

        # Aggregate wind stats across all cells and dates
        freq_cols  = [f"wind_sector_{s}_freq"       for s in range(n_sectors)]
        speed_cols = [f"wind_sector_{s}_mean_speed"  for s in range(n_sectors)]
        freq  = wind_df[freq_cols].mean().values
        speed = wind_df[speed_cols].mean().values

        # --- Main figure + inset axes ---
        fig = plt.figure(figsize=(10, 8))
        ax_map = fig.add_axes([0.05, 0.05, 0.88, 0.88])   # main map
        ax_rose = fig.add_axes([0.68, 0.62, 0.28, 0.30],   # inset wind rose
                               projection="polar")

        # Draw main map
        self._map_ax(ax_map, gam_values, cmap="turbo", legend=True)
        ax_map.set_title(title, fontsize=12, fontweight="bold", pad=12)

        # --- Wind rose (polar bar chart) ---
        # Meteorological convention: 0° = N (top), clockwise
        # matplotlib polar: 0° = E, anti-clockwise → convert
        angles_met  = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
        # Rotate so N is at top and direction is clockwise
        angles_plot = np.pi / 2 - angles_met

        bar_width = (2 * np.pi) / n_sectors * 0.85

        # Colour bars by mean wind speed
        norm  = plt.Normalize(vmin=speed.min(), vmax=speed.max())
        cmap_rose = plt.cm.Blues
        colours = cmap_rose(norm(speed))

        bars = ax_rose.bar(
            angles_plot, freq,
            width=bar_width,
            color=colours,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            align="center",
        )

        # Direction labels
        for angle, label in zip(angles_plot, sector_labels):
            r_label = freq.max() * 1.25
            ax_rose.text(
                angle, r_label, label,
                ha="center", va="center",
                fontsize=7, fontweight="bold",
            )

        ax_rose.set_theta_zero_location("N")
        ax_rose.set_theta_direction(-1)          # clockwise
        ax_rose.set_yticklabels([])
        ax_rose.set_xticklabels([])
        ax_rose.spines["polar"].set_visible(False)
        ax_rose.set_facecolor("whitesmoke")
        ax_rose.set_title("Wind rose\n(ERA5 freq)", fontsize=7, pad=4)

        # Colourbar for wind speed inside rose
        sm = plt.cm.ScalarMappable(cmap=cmap_rose, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_rose, orientation="horizontal",
                            pad=0.12, fraction=0.06, shrink=0.8)
        cbar.set_label("Mean speed (m/s)", fontsize=6)
        cbar.ax.tick_params(labelsize=6)

        _save(fig, save_path)
        return fig


# ---------------------------------------------------------------------------
# Temporal Visualizer
# ---------------------------------------------------------------------------

class TemporalVisualizer:
    """Time series visualization for SSM output.

    Colour convention:
      - steelblue  : GAM LUR static prior (horizontal dashed line)
      - black dots : EPA observed values
      - darkorange : GAM-SSM prediction line + ±1σ shading
      - steelblue triangles : days with satellite Kalman update
    """

    def plot_station_timeseries(
        self,
        preds_df: pd.DataFrame,
        date_col: str = "date",
        station_col: str = "station_id",
        obs_col: str = "obs_no2",
        lur_col: str = "lur_prior",
        ssm_col: str = "no2",
        uncertainty_col: Optional[str] = "pred_uncertainty",
        satellite_col: Optional[str] = "has_satellite",
        ncols: int = 3,
        suptitle: str = "GAM-SSM: Daily NO₂ at Monitoring Stations",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """3×3 (or n×ncols) panel of per-station time series.

        Each panel shows:
        - EPA observed (black dots)
        - GAM LUR prior (steelblue dashed horizontal line)
        - GAM-SSM prediction (darkorange line) with ±1σ uncertainty shading
        - Satellite update days (steelblue upward triangles)

        Parameters
        ----------
        preds_df : pd.DataFrame
            One row per (station, date) with all required columns.
        save_path : str or Path, optional
            Path to save figure.
        """
        stations = sorted(preds_df[station_col].unique())
        n = len(stations)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                 sharex=False)
        axes = np.atleast_1d(axes).flatten()

        for ax, sid in zip(axes, stations):
            grp = preds_df[preds_df[station_col] == sid].sort_values(date_col)
            dates = grp[date_col].values
            obs   = grp[obs_col].values
            lur   = float(grp[lur_col].iloc[0])
            ssm   = grp[ssm_col].values

            # Observed
            ax.plot(dates, obs, "k.", ms=4, lw=0, label="Observed", zorder=4)

            # LUR prior
            ax.axhline(lur, color=COL_LUR, ls="--", lw=1.5, label="GAM LUR prior")

            # SSM line
            ax.plot(dates, ssm, color=COL_SSM, lw=1.5, label="GAM-SSM", zorder=3)

            # Uncertainty shading
            if uncertainty_col and uncertainty_col in grp.columns:
                unc = grp[uncertainty_col].values
                ax.fill_between(dates, ssm - unc, ssm + unc,
                                alpha=ALPHA_SHADE, color=COL_SSM, label="±1σ")

            # Satellite update markers
            if satellite_col and satellite_col in grp.columns:
                sat_rows = grp[grp[satellite_col] == True]
                if len(sat_rows):
                    ax.scatter(sat_rows[date_col].values,
                               sat_rows[ssm_col].values,
                               marker="^", s=20, color=COL_LUR, zorder=5,
                               label="SAT update")

            ax.set_title(sid, fontsize=8, fontweight="bold")
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d"))
            ax.set_xlabel("Day", fontsize=7)
            ax.tick_params(axis="x", rotation=0, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)

        # Legend on first panel only
        if len(axes) > 0:
            axes[0].legend(fontsize=6, loc="upper right", ncol=1)

        # Hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(
            suptitle + "\n(observed = validation only, not in Kalman filter)",
            fontsize=10, y=1.01,
        )
        plt.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_epa_vs_predicted_timeseries(
        self,
        station_preds: pd.DataFrame,
        date_col: str = "date",
        obs_col: str = "obs_no2",
        pred_col: str = "no2",
        lur_col: str = "lur_prior",
        uncertainty_col: str = "pred_uncertainty",
        station_col: str = "station_id",
        ncols: int = 3,
        title: str = "Daily NO₂: EPA Observed vs GAM-SSM Predicted",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Per-station panels comparing EPA observations vs GAM-SSM predictions.

        One panel per station showing:
        - Black dots: EPA observed NO₂
        - Orange line: GAM-SSM smoothed prediction
        - Blue shading: 95% prediction interval (±1.96σ)
        - Grey dashed line: static GAM-LUR prior (constant per station)

        A final summary panel shows the daily mean across all stations.
        """
        station_preds = station_preds.copy()
        station_preds[date_col] = pd.to_datetime(station_preds[date_col])
        stations = sorted(station_preds[station_col].unique())
        n = len(stations)

        # +1 for the summary panel
        total_panels = n + 1
        nrows = (total_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()

        z95 = 1.96
        has_lur = lur_col in station_preds.columns

        for i, sid in enumerate(stations):
            ax = axes[i]
            grp = station_preds[station_preds[station_col] == sid].sort_values(date_col)

            # 95% prediction interval shading
            ax.fill_between(
                grp[date_col],
                grp[pred_col] - z95 * grp[uncertainty_col],
                grp[pred_col] + z95 * grp[uncertainty_col],
                alpha=ALPHA_SHADE, color=COL_SSM, label="95% PI",
            )
            # GAM-SSM smoothed prediction
            ax.plot(grp[date_col], grp[pred_col],
                    color=COL_SSM, lw=2, label="GAM-SSM")
            # EPA observed dots
            ax.plot(grp[date_col], grp[obs_col],
                    "k.", ms=5, alpha=0.85, label="EPA observed")
            # Static GAM prior — horizontal dashed reference line
            if has_lur and not grp[lur_col].isna().all():
                gam_val = float(grp[lur_col].iloc[0])
                ax.axhline(gam_val, color="grey", lw=1.2, ls="--",
                           alpha=0.7, label=f"GAM-LUR prior ({gam_val:.1f})")

            ax.set_title(sid, fontsize=9, fontweight="bold")
            ax.set_ylabel("NO₂ (µg/m³)", fontsize=8)
            ax.set_xlabel("Day", fontsize=8)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d"))
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")

        # Summary panel — daily mean across all stations
        ax2 = axes[n]
        daily = station_preds.groupby(date_col).agg(
            obs_mean=(obs_col, "mean"),
            pred_mean=(pred_col, "mean"),
            pred_std=(uncertainty_col, "mean"),
        ).reset_index().sort_values(date_col)

        ax2.fill_between(daily[date_col],
                         daily["pred_mean"] - z95 * daily["pred_std"],
                         daily["pred_mean"] + z95 * daily["pred_std"],
                         alpha=ALPHA_SHADE, color=COL_SSM, label="95% PI")
        ax2.plot(daily[date_col], daily["pred_mean"],
                 color=COL_SSM, lw=2, label="GAM-SSM mean")
        ax2.plot(daily[date_col], daily["obs_mean"],
                 "k.-", ms=6, lw=1.2, label="EPA observed mean")
        ax2.set_title("All stations — daily mean", fontsize=9, fontweight="bold")
        ax2.set_ylabel("NO₂ (µg/m³)", fontsize=8)
        ax2.set_xlabel("Day", fontsize=8)
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-d"))
        ax2.tick_params(labelsize=7)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.grid(True, alpha=0.2)

        # Hide any leftover axes
        for ax in axes[total_panels:]:
            ax.set_visible(False)

        fig.suptitle(title, fontsize=11, fontweight="bold")
        _save(fig, save_path)
        return fig

    def plot_daily_mean_barchart(
        self,
        ssm_df: pd.DataFrame,
        highlighted_dates: Optional[List] = None,
        date_col: str = "date",
        value_col: str = "no2",
        title: str = "Daily Area-Mean NO₂",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Bar chart of daily area-mean NO₂ with highlighted dates.

        Parameters
        ----------
        ssm_df : pd.DataFrame
            SSM output with ``date`` and ``no2`` columns.
        highlighted_dates : list, optional
            Dates to highlight in a contrasting colour (e.g. the 4 map days).
        """
        daily = ssm_df.groupby(date_col)[value_col].mean().reset_index()
        daily[date_col] = pd.to_datetime(daily[date_col])
        daily = daily.sort_values(date_col)

        highlighted = set(pd.to_datetime(d) for d in (highlighted_dates or []))
        colours = [
            COL_SSM if d in highlighted else "#cccccc"
            for d in daily[date_col]
        ]

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(daily[date_col], daily[value_col], 
            color=COL_SSM, lw=2, marker="o", ms=5, label="Daily mean")

        # Annotate highlighted bars
        for d in highlighted:
            ax.axvline(d, color=COL_SSM, linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("NO₂ (µg/m³)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)

        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(color=COL_SSM, label="Map days"),
                Patch(color="#cccccc", label="Other days"),
            ],
            fontsize=8,
        )
        _save(fig, save_path)
        return fig

    def plot_temporal_variance(
        self,
        dates: List,
        ssm_mean: NDArray,
        ssm_std: NDArray,
        obs: Optional[NDArray] = None,
        title: str = "Temporal Dynamics",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot area-averaged SSM mean ± uncertainty over time.

        Parameters
        ----------
        dates : list
            Date sequence.
        ssm_mean : array-like, shape (T,)
            Area-averaged SSM NO2 estimate per day.
        ssm_std : array-like, shape (T,)
            Area-averaged prediction standard deviation.
        obs : array-like, shape (T,), optional
            Area-averaged observed values for overlay.
        """
        dates = np.asarray(dates)
        ssm_mean = np.asarray(ssm_mean)
        ssm_std  = np.asarray(ssm_std)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, ssm_mean - ssm_std, ssm_mean + ssm_std,
                        alpha=ALPHA_SHADE, color=COL_SSM, label="±1σ")
        ax.plot(dates, ssm_mean, color=COL_SSM, lw=2, label="GAM-SSM mean")
        if obs is not None:
            ax.plot(dates, obs, "k.", ms=5, label="Observed")

        ax.set_xlabel("Date")
        ax.set_ylabel("NO₂ (µg/m³)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, save_path)
        return fig


# ---------------------------------------------------------------------------
# Model Comparison Visualizer
# ---------------------------------------------------------------------------

class ModelComparisonVisualizer:
    """Compare model outputs and cross-validation results."""

    def plot_loocv_scatter(
        self,
        cv_df: pd.DataFrame,
        obs_col: str = "obs_no2",
        pred_col: str = "no2",
        station_col: str = "station_id",
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Leave-one-station-out cross-validation scatter.

        Each station is plotted in a distinct colour.  An OLS regression line
        and the 1:1 line are overlaid, with CV R² and Pearson r in the title.

        Parameters
        ----------
        cv_df : pd.DataFrame
            One row per (station, date) with observed and LOOCV-predicted NO2.
        """
        y_obs  = cv_df[obs_col].values
        y_pred = cv_df[pred_col].values

        slope, intercept, r_val, _, _ = linregress(y_pred, y_obs)
        x_line = np.linspace(y_pred.min(), y_pred.max(), 200)

        ss_res = np.sum((y_obs - y_pred) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        cv_r2  = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fig, ax = plt.subplots(figsize=(6, 6))

        stations = sorted(cv_df[station_col].unique())
        for i, sid in enumerate(stations):
            grp = cv_df[cv_df[station_col] == sid]
            colour = _STATION_PALETTE[i % len(_STATION_PALETTE)]
            ax.scatter(grp[pred_col], grp[obs_col],
                       s=18, alpha=0.75, color=colour,
                       edgecolor="white", linewidth=0.5, label=sid)

        ax.plot(x_line, intercept + slope * x_line,
                color=COL_LUR, lw=1.5, label="Trend line", zorder=5)
        lim_lo = min(y_obs.min(), y_pred.min()) * 0.9
        lim_hi = max(y_obs.max(), y_pred.max()) * 1.05
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                "r--", lw=1.2, label="1:1", zorder=4)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)

        ax.set_xlabel("LOOCV Predicted NO₂ (µg/m³)")
        ax.set_ylabel("Measured NO₂ (µg/m³)")
        _annotate_r2(ax, y_obs, y_pred)

        t = title or f"LOOCV: Modelled vs Measured  (CV R²={cv_r2:.3f}, r={r_val:.3f})"
        ax.set_title(t, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_obs_vs_pred(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        labels: Optional[List[str]] = None,
        title: str = "Observed vs Predicted",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Scatter plot of observed vs predicted with 1:1 line and R²."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        fig, ax = plt.subplots(figsize=(6, 6))

        if labels is not None:
            unique_labels = sorted(set(labels))
            for i, lbl in enumerate(unique_labels):
                mask = np.array(labels) == lbl
                ax.scatter(y_pred[mask], y_true[mask], s=20, alpha=0.75,
                           color=_STATION_PALETTE[i % len(_STATION_PALETTE)],
                           edgecolor="white", linewidth=0.4, label=lbl)
        else:
            ax.scatter(y_pred, y_true, s=20, alpha=0.75,
                       color=COL_LUR, edgecolor="white", linewidth=0.4)

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1")

        _annotate_r2(ax, y_true, y_pred)
        ax.set_xlabel("Predicted NO₂ (µg/m³)")
        ax.set_ylabel("Observed NO₂ (µg/m³)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        if labels:
            ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        _save(fig, save_path)
        return fig

    def compare_accuracy_table(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Grouped bar chart of RMSE / MAE / R² across multiple models.

        Parameters
        ----------
        results : dict
            ``{model_name: {metric: value}}``.  Expected metrics:
            ``rmse``, ``mae``, ``r2``.
        """
        models  = list(results.keys())
        metrics = ["rmse", "mae", "r2"]
        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, model in enumerate(models):
            vals = [results[model].get(m, 0.0) for m in metrics]
            ax.bar(x + i * width, vals, width, label=model, alpha=0.85)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(["RMSE (µg/m³)", "MAE (µg/m³)", "R²"])
        ax.set_title("Model Comparison", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, save_path)
        return fig


# ---------------------------------------------------------------------------
# Diagnostics Visualizer
# ---------------------------------------------------------------------------

class DiagnosticsVisualizer:
    """Residual and convergence diagnostic plots."""

    def plot_residual_panel(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        title: str = "Residual Diagnostics",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """2×3 diagnostic panel.

        Panels: (a) Observed vs Predicted, (b) Residuals histogram with mean
        line, (c) Residuals vs Predicted, (d) Q-Q plot, (e) ACF,
        (f) Temporal residual pattern.
        """
        from scipy import stats

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        res    = y_true - y_pred

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # (a) Observed vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, c=COL_LUR,
                   edgecolor="white", linewidth=0.4)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1")
        ax.set_xlabel("Observed NO₂ (µg/m³)")
        ax.set_ylabel("Predicted NO₂ (µg/m³)")
        ax.set_title("(a) Observed vs Predicted", fontsize=11, fontweight="bold")
        _annotate_r2(ax, y_true, y_pred)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (b) Residuals histogram
        ax = axes[0, 1]
        ax.hist(res, bins=30, color=COL_LUR, edgecolor="white", alpha=0.7)
        ax.axvline(0, color="red", ls="--", lw=1.5)
        ax.axvline(res.mean(), color=COL_SSM, lw=1.5,
                   label=f"Mean: {res.mean():.2f}")
        ax.set_xlabel("Residual (µg/m³)")
        ax.set_ylabel("Frequency")
        ax.set_title("(b) Residual Distribution", fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # (c) Residuals vs Predicted
        ax = axes[0, 2]
        ax.scatter(y_pred, res, alpha=0.6, s=20, c=COL_LUR,
                   edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="red", ls="--", lw=1.5)
        ax.set_xlabel("Predicted NO₂ (µg/m³)")
        ax.set_ylabel("Residual (µg/m³)")
        ax.set_title("(c) Residuals vs Fitted", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # (d) Q-Q plot
        ax = axes[1, 0]
        stats.probplot(res, dist="norm", plot=ax)
        ax.set_title("(d) Normal Q-Q", fontsize=11, fontweight="bold")
        ax.get_lines()[0].set(markersize=3, alpha=0.5, color=COL_LUR)

        # (e) ACF
        ax = axes[1, 1]
        n = len(res)
        max_lag = min(40, n // 4)
        mean_r  = np.mean(res)
        var_r   = np.var(res) + 1e-12
        acf = np.array([
            np.mean((res[:n - k] - mean_r) * (res[k:] - mean_r)) / var_r
            for k in range(max_lag + 1)
        ])
        ax.bar(range(len(acf)), acf, color=COL_LUR, edgecolor="white")
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci,  color="red", ls="--", lw=1)
        ax.axhline(-ci, color="red", ls="--", lw=1)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title("(e) Residual Autocorrelation", fontsize=11, fontweight="bold")

        # (f) Temporal pattern (rolling mean ± std)
        ax = axes[1, 2]
        window = max(len(res) // 50, 5)
        rs = pd.Series(res)
        roll_mean = rs.rolling(window, center=True).mean()
        roll_std  = rs.rolling(window, center=True).std()
        idx = np.arange(len(res))
        ax.fill_between(idx, roll_mean - roll_std, roll_mean + roll_std,
                        alpha=0.3, color=COL_LUR, label="±1 SD")
        ax.plot(idx, roll_mean, color=COL_LUR, lw=1.5, label="Rolling mean")
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xlabel("Index")
        ax.set_ylabel("Residual")
        ax.set_title("(f) Temporal Residual Pattern", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_convergence(
        self,
        log_likelihoods: List[float],
        title: str = "EM Convergence",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Log-likelihood trace and increment plot."""
        ll   = np.asarray(log_likelihoods)
        iters = np.arange(len(ll))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(iters, ll, "o-", color=COL_LUR, ms=4)
        axes[0].set_xlabel("EM Iteration")
        axes[0].set_ylabel("Log-Likelihood")
        axes[0].set_title("(a) Log-Likelihood", fontsize=11, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        if len(ll) > 1:
            diff = np.abs(np.diff(ll))
            axes[1].semilogy(iters[1:], diff, "o-", color=COL_SSM, ms=4)
            axes[1].axhline(1e-6, color="red", ls="--", lw=1, label="Tolerance 1e-6")
            axes[1].set_xlabel("EM Iteration")
            axes[1].set_ylabel("|ΔLog-Likelihood|")
            axes[1].set_title("(b) Increment", fontsize=11, fontweight="bold")
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_reliability_diagram(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        y_std: NDArray,
        title: str = "Probabilistic Calibration — Reliability Diagram",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Reliability diagram + interval width + CRPS for probabilistic evaluation.

        Three panels:
        (a) Reliability diagram — nominal vs empirical coverage. A perfectly
            calibrated model lies on the diagonal.
        (b) Prediction interval widths at each nominal level — shows sharpness.
        (c) Interval skill score (ISS) at each level — penalises both
            under-coverage and unnecessarily wide intervals.

        Parameters
        ----------
        y_true : array-like
            Observed values.
        y_pred : array-like
            Predicted mean values.
        y_std : array-like
            Predicted standard deviations (from SSM).
        """
        from scipy import stats

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_std  = np.asarray(y_std)

        nominal_levels = [0.50, 0.80, 0.90, 0.95, 0.99]
        empirical_coverages = []
        interval_widths     = []
        iss_scores          = []

        for alpha in nominal_levels:
            z = stats.norm.ppf(0.5 + alpha / 2)
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            width = 2 * z * y_std

            covered  = ((y_true >= lower) & (y_true <= upper)).mean()
            penalty  = (2 / (1 - alpha)) * np.where(
                y_true < lower, lower - y_true,
                np.where(y_true > upper, y_true - upper, 0)
            ).mean()
            iss = width.mean() + penalty

            empirical_coverages.append(covered)
            interval_widths.append(width.mean())
            iss_scores.append(iss)

        crps = np.mean(
            y_std * (
                (y_true - y_pred) / y_std * (
                    2 * stats.norm.cdf((y_true - y_pred) / y_std) - 1
                )
                + 2 * stats.norm.pdf((y_true - y_pred) / y_std)
                - 1 / np.sqrt(np.pi)
            )
        )

        pct_labels = [f"{int(l*100)}%" for l in nominal_levels]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

        # (a) Reliability diagram
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(nominal_levels, empirical_coverages, "o-",
                color=COL_LUR, lw=2, ms=7, label="Model")
        ax.fill_between(nominal_levels, nominal_levels, empirical_coverages,
                        alpha=0.15, color=COL_LUR)
        ax.set_xlabel("Nominal coverage", fontsize=9)
        ax.set_ylabel("Empirical coverage", fontsize=9)
        ax.set_title("(a) Reliability Diagram", fontsize=10, fontweight="bold")
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.45, 1.02)
        ax.set_xticks(nominal_levels)
        ax.set_xticklabels(pct_labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate over/under confidence
        mid_nom = nominal_levels[len(nominal_levels)//2]
        mid_emp = empirical_coverages[len(nominal_levels)//2]
        if mid_emp < mid_nom - 0.02:
            ax.text(0.05, 0.92, "Overconfident\n(intervals too narrow)",
                    transform=ax.transAxes, fontsize=7, color="red", alpha=0.7)
        elif mid_emp > mid_nom + 0.02:
            ax.text(0.05, 0.92, "Underconfident\n(intervals too wide)",
                    transform=ax.transAxes, fontsize=7, color="orange", alpha=0.7)

        # (b) Interval widths
        ax2 = axes[1]
        ax2.bar(pct_labels, interval_widths, color=COL_SSM, alpha=0.8)
        ax2.set_xlabel("Nominal coverage level", fontsize=9)
        ax2.set_ylabel("Mean interval width (µg/m³)", fontsize=9)
        ax2.set_title("(b) Interval Width (Sharpness)", fontsize=10, fontweight="bold")
        ax2.tick_params(axis="x", labelsize=8)
        ax2.grid(True, alpha=0.3, axis="y")

        # (c) ISS
        ax3 = axes[2]
        ax3.bar(pct_labels, iss_scores, color=COL_LUR, alpha=0.8)
        ax3.set_xlabel("Nominal coverage level", fontsize=9)
        ax3.set_ylabel("Interval Skill Score (lower = better)", fontsize=9)
        ax3.set_title(f"(c) Interval Skill Score  |  CRPS={crps:.3f}",
                      fontsize=10, fontweight="bold")
        ax3.tick_params(axis="x", labelsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

        fig.suptitle(title, fontsize=11, fontweight="bold")
        _save(fig, save_path)
        return fig


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_publication_figure_set(
    model,
    output_dir: Union[str, Path],
    grid_gdf=None,
    cv_df: Optional[pd.DataFrame] = None,
    ssm_df: Optional[pd.DataFrame] = None,
    station_preds: Optional[pd.DataFrame] = None,
    selected_dates: Optional[List] = None,
    X_train_df: Optional[pd.DataFrame] = None,
    wind_df: Optional[pd.DataFrame] = None,
) -> None:
    """Generate the full set of publication figures.

    Writes the following files to *output_dir*:

    - ``static_lur_prior.png``       — GAM LUR spatial prior
    - ``spatial_residuals.png``      — signed and absolute residuals
    - ``residual_diagnostics.png``   — 2×3 diagnostic panel
    - ``ssm_selected_days.png``      — daily snapshot maps (if ssm_df provided)
    - ``station_timeseries.png``     — per-station time series (if station_preds provided)
    - ``loocv_scatter.png``          — LOOCV scatter (if cv_df provided)
    - ``em_convergence.png``         — EM convergence (if model has ssm_)
    - ``wind_sector_map.png``        — GAM NO₂ map per dominant wind sector (if X_train_df provided)

    Parameters
    ----------
    model : HybridGAMSSM
        Fitted model.
    output_dir : str or Path
        Directory to write figures.
    grid_gdf : GeoDataFrame, optional
        Polygon grid geometry.
    cv_df : pd.DataFrame, optional
        LOOCV results from ``ModelEvaluator.loocv_stations()``.
    ssm_df : pd.DataFrame, optional
        SSM daily output for daily snapshot maps.
    station_preds : pd.DataFrame, optional
        Per-station predictions for time series panel.
    selected_dates : list, optional
        Dates for daily snapshot maps.  If None and ssm_df is provided,
        four evenly spaced dates are chosen automatically.
    X_train_df : pd.DataFrame, optional
        Training feature matrix as a DataFrame with named columns.  Required
        for the wind sector map.  Pass ``pd.DataFrame(model._X_train,
        columns=model.gam_.feature_names_)`` or the raw features DataFrame.
    wind_df : pd.DataFrame, optional
        ERA5 wind sector daily data with columns ``wind_sector_N_freq`` and
        ``wind_sector_N_mean_speed``.  Used for the GAM + wind rose map.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gids = list(model.location_ids_) if hasattr(model, "location_ids_") else None
    sv   = SpatialVisualizer(grid_gdf=grid_gdf, grid_ids=gids)
    tv   = TemporalVisualizer()
    cv   = ModelComparisonVisualizer()
    dv   = DiagnosticsVisualizer()

    # GAM LUR prior map
    if hasattr(model, "gam_") and model.gam_ is not None:
        lur_pred = model.gam_.predict(model._X_train)
        sv.plot_surface(
            lur_pred, title="GAM LUR — Annual Mean NO₂",
            save_path=output_dir / "static_lur_prior.png",
        )

        # Residuals
        res = model._y_train - lur_pred
        sv.plot_residuals(
            res,
            predictions=lur_pred,
            title="GAM LUR — Spatial Prediction and Residuals",
            save_path=output_dir / "spatial_residuals.png",
        )
        dv.plot_residual_panel(
            model._y_train, lur_pred,
            title="GAM LUR — Residual Diagnostics",
            save_path=output_dir / "residual_diagnostics.png",
        )

    # EM convergence
    if (hasattr(model, "ssm_") and model.ssm_ is not None
            and model.ssm_.em_result_ is not None):
        dv.plot_convergence(
            model.ssm_.em_result_.log_likelihoods,
            save_path=output_dir / "em_convergence.png",
        )

    # Daily snapshots — pick 4 days spanning min/low/high/max daily-mean NO₂
    if ssm_df is not None:
        dates = selected_dates
        if dates is None:
            daily_mean = ssm_df.groupby("date")["no2"].mean().sort_values()
            n = len(daily_mean)
            idx = sorted({0, n // 3, 2 * n // 3, n - 1})
            dates = [daily_mean.index[i] for i in idx]
        sv.plot_daily_snapshots(
            ssm_df, dates=dates,
            suptitle="GAM-SSM Daily NO₂ — low → high pollution days",
            save_path=output_dir / "ssm_selected_days.png",
        )
        tv.plot_daily_mean_barchart(
            ssm_df, highlighted_dates=dates,
            title=(
                "Daily Area-Mean NO₂\n"
                "Highlighted days selected to span the observed temporal range "
                "(minimum, lower-tercile, upper-tercile, maximum)"
            ),
            save_path=output_dir / "ssm_daily_mean_barchart.png",
        )

    # Station time series
    if station_preds is not None:
        tv.plot_station_timeseries(
            station_preds,
            save_path=output_dir / "station_timeseries.png",
        )

    # LOOCV scatter
    if cv_df is not None:
        cv.plot_loocv_scatter(
            cv_df, save_path=output_dir / "loocv_scatter.png",
        )

    # Wind sector map (requires fitted GAM + feature DataFrame)
    if (X_train_df is not None
            and hasattr(model, "gam_") and model.gam_ is not None):
        sv.plot_wind_sector_map(
            model.gam_,
            X_train_df,
            save_path=output_dir / "wind_sector_map.png",
        )

    # GAM map + wind rose inset (requires ERA5 wind data)
    if (wind_df is not None
            and hasattr(model, "gam_") and model.gam_ is not None):
        lur_vals = model.gam_.predict(model._X_train)
        sv.plot_gam_with_wind_rose(
            lur_vals,
            wind_df,
            save_path=output_dir / "gam_wind_rose.png",
        )

    logger.info("Figure set written to %s", output_dir)
