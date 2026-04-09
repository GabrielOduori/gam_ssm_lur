"""
Generic Spatiotemporal Data Loader for GAM-SSM-LUR.

Defines a standard input contract that any city's data can satisfy:

  features.csv         — static spatial features per grid cell
  target.csv           — annual/background pollutant concentration per grid cell
  time_series/
    <dense_obs>        — high-coverage gridded observations (e.g. satellite)
                         → used to UPDATE the Kalman filter
    <point_obs>        — sparse station observations
                         → held out for VALIDATION only, never enters filter
    <activity>         — city-wide activity proxy time series (e.g. traffic)
                         → used as transition forcing Δ(t)
    <met_forcing>      — gridded meteorological forcing (e.g. wind sectors)
                         → used as spatially varying transition forcing W(s,t)
  grid.geojson         — polygon grid geometry (optional, for mapping)

Dublin reference data maps onto this contract as:
  dense_obs  → satellite_retreavals.csv  (TROPOMI column tropomi_no2)
  point_obs  → epa_timeseries.csv        (epa_no2)
  activity   → traffic_timeseries.csv    (traffic_volume)
  met_forcing→ wind_sector_features_era5land_2023-06_daily.csv

Column names are fully configurable so any city's data can be loaded
without modifying this class.

References
----------
.. [1] Naughton, O., et al. (2018). A land use regression model for explaining
       spatial variation in air pollution. Science of the Total Environment.
.. [2] Hoek, G., et al. (2008). A review of land-use regression models to assess
       spatial variation of outdoor air pollution. Atmospheric Environment, 42(33).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import lstsq
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StaticData:
    """Static (time-invariant) inputs for the GAM-LUR component.

    Attributes
    ----------
    features : pd.DataFrame
        Spatial feature matrix. Columns include ``grid_id``, ``latitude``,
        ``longitude``, and all predictor features. Shape (n_cells, n_features+3).
    target : pd.DataFrame
        Background pollutant concentration per cell. Columns: ``grid_id``,
        ``latitude``, ``longitude``, ``<target_col>``.
    grid_ids : list of str
        Ordered list of grid cell identifiers.
    """
    features: pd.DataFrame
    target: pd.DataFrame
    grid_ids: List[str]


@dataclass
class TemporalData:
    """Time-varying inputs for the SSM component.

    Attributes
    ----------
    dense_obs : pd.DataFrame
        Gridded observations used to update the Kalman filter. Columns:
        ``grid_id``, ``date``, ``obs_value``.
    point_obs : pd.DataFrame
        Sparse station observations held out for validation. Columns:
        ``station_id``, ``grid_id``, ``date``, ``obs_value``.
    activity_forcing : pd.DataFrame
        City-wide daily activity anomaly. Columns: ``date``, ``activity_mean``,
        ``delta_activity`` (normalised anomaly, mean-centred and scaled).
    met_forcing : pd.DataFrame
        Per-cell daily meteorological forcing scalar. Columns: ``grid_id``,
        ``date``, ``met_forcing``.
    dates : list
        Sorted list of unique dates covered by the time series.
    """
    dense_obs: pd.DataFrame
    point_obs: pd.DataFrame
    activity_forcing: pd.DataFrame
    met_forcing: pd.DataFrame
    dates: List


@dataclass
class CalibrationResult:
    """Coefficients from satellite-to-surface bias correction.

    Attributes
    ----------
    beta0 : float
        Intercept of OLS fit: surface = beta0 + beta1 * satellite.
    beta1 : float
        Slope of OLS fit.
    sigma2_obs : float
        Estimated observation noise variance (used as Kalman R matrix).
    n_collocated : int
        Number of collocated station–satellite pairs used for fitting.
    r : float
        Pearson correlation of fit.
    """
    beta0: float
    beta1: float
    sigma2_obs: float
    n_collocated: int
    r: float

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Apply calibration: raw satellite → surface-equivalent."""
        return self.beta0 + self.beta1 * np.asarray(raw)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class SpatiotemporalDataset:
    """Generic loader for spatiotemporal LUR datasets.

    Reads a directory organised as::

        data_dir/
          features.csv
          target.csv
          time_series/
            <dense_obs_file>
            <point_obs_file>
            <activity_file>
            <met_forcing_file>
          grid.geojson          (optional)

    All file names and column names are configurable so the same class works
    for any city. Dublin defaults are provided for every parameter.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory (contains ``features.csv`` and ``target.csv``).
    time_series_dir : str or Path, optional
        Sub-directory containing temporal CSVs. Defaults to
        ``data_dir/time_series``.
    features_file : str
        Filename of the spatial features CSV. Default ``features.csv``.
    target_file : str
        Filename of the target CSV. Default ``target.csv``.
    target_col : str
        Column name for the pollutant target in ``target_file``.
        Default ``atmos_no2``.
    dense_obs_file : str
        Filename of the gridded observation CSV (e.g. satellite).
        Default ``satellite_retreavals.csv``.
    dense_obs_timestamp_col : str
        Timestamp column in the dense obs file. Default ``timestamp``.
    dense_obs_value_col : str
        Value column in the dense obs file. Default ``tropomi_no2``.
    point_obs_file : str
        Filename of the station observation CSV.
        Default ``epa_timeseries.csv``.
    point_obs_station_col : str
        Station identifier column. Default ``station_id``.
    point_obs_timestamp_col : str
        Timestamp column in the station file. Default ``timestamp_utc``.
    point_obs_value_col : str
        Value column in the station file. Default ``epa_no2``.
    activity_file : str
        Filename of the activity/traffic CSV.
        Default ``traffic_timeseries.csv``.
    activity_timestamp_col : str
        Timestamp column in the activity file. Default ``traffic_end_time``.
    activity_value_col : str
        Volume column in the activity file. Default ``traffic_volume``.
    met_forcing_file : str
        Filename of the meteorological forcing CSV.
        Default ``wind_sector_features_era5land_2023-06_daily.csv``.
    met_date_col : str
        Date column in the met forcing file. Default ``date``.
    met_n_sectors : int
        Number of wind/met sectors. Default 8.
    met_freq_prefix : str
        Prefix for sector frequency columns. Default ``wind_sector_``.
    met_speed_prefix : str
        Prefix for sector mean-speed columns. Default ``wind_sector_``.
    met_freq_suffix : str
        Suffix for frequency columns. Default ``_freq``.
    met_speed_suffix : str
        Suffix for speed columns. Default ``_mean_speed``.
    grid_geojson : str, optional
        Path (relative to ``data_dir``) to grid GeoJSON. Default ``grid.geojson``.

    Examples
    --------
    Load with Dublin defaults:

    >>> ds = SpatiotemporalDataset("/path/to/lur_data")
    >>> static = ds.load_static()
    >>> temporal = ds.load_temporal()

    Load with custom column names for a different city:

    >>> ds = SpatiotemporalDataset(
    ...     "/path/to/city_data",
    ...     dense_obs_file="sentinel5p.csv",
    ...     dense_obs_value_col="no2_molm2",
    ...     activity_file="road_counts.csv",
    ...     activity_value_col="vehicle_count",
    ... )
    """

    def __init__(
        self,
        data_dir: str | Path,
        time_series_dir: Optional[str | Path] = None,
        # Static files
        features_file: str = "features.csv",
        target_file: str = "target.csv",
        target_col: str = "atmos_no2",
        # Dense gridded observations (Kalman update source)
        dense_obs_file: str = "satellite_retreavals.csv",
        dense_obs_timestamp_col: str = "timestamp",
        dense_obs_value_col: str = "tropomi_no2",
        # Sparse point observations (validation only)
        point_obs_file: str = "epa_timeseries.csv",
        point_obs_station_col: str = "station_id",
        point_obs_timestamp_col: str = "timestamp_utc",
        point_obs_value_col: str = "epa_no2",
        # Activity forcing (city-wide transition covariate)
        activity_file: str = "traffic_timeseries.csv",
        activity_timestamp_col: str = "traffic_end_time",
        activity_value_col: str = "traffic_volume",
        # Meteorological forcing (spatially varying transition covariate)
        met_forcing_file: str = "wind_sector_features_era5land_2023-06_daily.csv",
        met_date_col: str = "date",
        met_n_sectors: int = 8,
        met_freq_prefix: str = "wind_sector_",
        met_speed_prefix: str = "wind_sector_",
        met_freq_suffix: str = "_freq",
        met_speed_suffix: str = "_mean_speed",
        # Grid geometry
        grid_geojson: Optional[str] = "grid/grid.geojson",
    ):
        self.data_dir = Path(data_dir)
        self.ts_dir = Path(time_series_dir) if time_series_dir else self.data_dir / "time_series"

        self.features_file = features_file
        self.target_file = target_file
        self.target_col = target_col

        self.dense_obs_file = dense_obs_file
        self.dense_obs_timestamp_col = dense_obs_timestamp_col
        self.dense_obs_value_col = dense_obs_value_col

        self.point_obs_file = point_obs_file
        self.point_obs_station_col = point_obs_station_col
        self.point_obs_timestamp_col = point_obs_timestamp_col
        self.point_obs_value_col = point_obs_value_col

        self.activity_file = activity_file
        self.activity_timestamp_col = activity_timestamp_col
        self.activity_value_col = activity_value_col

        self.met_forcing_file = met_forcing_file
        self.met_date_col = met_date_col
        self.met_n_sectors = met_n_sectors
        self.met_freq_prefix = met_freq_prefix
        self.met_speed_prefix = met_speed_prefix
        self.met_freq_suffix = met_freq_suffix
        self.met_speed_suffix = met_speed_suffix

        self._grid_geojson = (
            self.data_dir / grid_geojson if grid_geojson else None
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load_static(self) -> StaticData:
        """Load features and target for the static GAM-LUR component.

        Returns
        -------
        StaticData
            Features DataFrame, target DataFrame, and ordered grid IDs.
        """
        features = self._load_features()
        target = self._load_target()
        grid_ids = list(features["grid_id"].values)
        logger.info(
            "Static data: %d grid cells, %d feature columns",
            len(grid_ids),
            len(features.columns) - 3,  # exclude grid_id, lat, lon
        )
        return StaticData(features=features, target=target, grid_ids=grid_ids)

    def load_temporal(self) -> TemporalData:
        """Load all time-varying data for the SSM component.

        Returns
        -------
        TemporalData
            Dense observations, point observations, activity forcing,
            meteorological forcing, and sorted date list.
        """
        dense_obs = self._load_dense_obs()
        point_obs = self._load_point_obs()
        activity = self._load_activity_forcing()
        met = self._load_met_forcing()

        # Union of dates across dense obs and activity
        dates_dense = set(dense_obs["date"].unique()) if len(dense_obs) else set()
        dates_activity = set(activity["date"].unique()) if len(activity) else set()
        dates = sorted(dates_dense | dates_activity)

        logger.info(
            "Temporal data: %d dates, %d dense obs rows, %d station obs rows",
            len(dates),
            len(dense_obs),
            len(point_obs),
        )
        return TemporalData(
            dense_obs=dense_obs,
            point_obs=point_obs,
            activity_forcing=activity,
            met_forcing=met,
            dates=dates,
        )

    def calibrate_dense_obs(
        self,
        temporal: TemporalData,
        static: StaticData,
        min_collocated: int = 10,
    ) -> CalibrationResult:
        """Calibrate dense observations against point observations via OLS.

        Fits: ``point_obs = beta0 + beta1 * dense_obs`` at the grid cells
        that contain monitoring stations on days where both sources exist.

        This corrects for systematic bias between the two observation types
        (e.g. TROPOMI column retrievals vs. surface-level EPA measurements).

        Parameters
        ----------
        temporal : TemporalData
            Loaded temporal data (dense + point obs).
        static : StaticData
            Loaded static data (used for grid cell lookup).
        min_collocated : int
            Minimum collocated samples required. Falls back to identity
            calibration (beta0=0, beta1=1) if fewer are available.

        Returns
        -------
        CalibrationResult
            Fitted calibration coefficients and observation noise estimate.
        """
        sta_grids = (
            temporal.point_obs[["station_id", "grid_id"]]
            .drop_duplicates()
        )
        merged = (
            temporal.dense_obs
            .merge(sta_grids, on="grid_id", how="inner")
            .merge(
                temporal.point_obs[["station_id", "date", "obs_value"]],
                on=["station_id", "date"],
                how="inner",
            )
            .dropna(subset=["obs_value", "obs_dense"])
        )
        n = len(merged)

        if n < min_collocated:
            logger.warning(
                "Only %d collocated obs for calibration (need %d). "
                "Using identity calibration (beta0=0, beta1=1).",
                n, min_collocated,
            )
            return CalibrationResult(
                beta0=0.0, beta1=1.0, sigma2_obs=36.0,
                n_collocated=n, r=float("nan"),
            )

        X = np.column_stack([np.ones(n), merged["obs_dense"].values])
        y = merged["obs_value"].values
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        beta0, beta1 = float(coeffs[0]), float(coeffs[1])

        resid = y - (beta0 + beta1 * merged["obs_dense"].values)
        sigma2_obs = float(max(np.var(resid) * 2.0, 25.0))
        r = float(np.corrcoef(y, merged["obs_dense"].values)[0, 1])

        logger.info(
            "Calibration: N=%d  surface = %.2f + %.2f × dense  r=%.3f  σ²=%.1f",
            n, beta0, beta1, r, sigma2_obs,
        )
        return CalibrationResult(
            beta0=round(beta0, 4),
            beta1=round(beta1, 4),
            sigma2_obs=round(sigma2_obs, 2),
            n_collocated=n,
            r=round(r, 4),
        )

    def load_grid_geometry(self) -> Optional["gpd.GeoDataFrame"]:
        """Load polygon grid GeoJSON for spatial mapping.

        Returns
        -------
        GeoDataFrame or None
            Grid polygons with ``grid_id`` column, or None if file not found.
        """
        if self._grid_geojson is None or not self._grid_geojson.exists():
            logger.warning("Grid GeoJSON not found at %s", self._grid_geojson)
            return None
        try:
            import geopandas as gpd
            gdf = gpd.read_file(self._grid_geojson)
            logger.info("Grid geometry loaded: %d polygons", len(gdf))
            return gdf
        except ImportError:
            logger.warning("geopandas not installed — grid geometry unavailable")
            return None

    # -----------------------------------------------------------------------
    # Private loading methods
    # -----------------------------------------------------------------------

    def _load_features(self) -> pd.DataFrame:
        path = self.data_dir / self.features_file
        logger.info("Loading features from %s", path)
        df = pd.read_csv(path, low_memory=False)
        if "grid_id" not in df.columns:
            raise ValueError(f"'grid_id' column missing from {path}")
        return df

    def _load_target(self) -> pd.DataFrame:
        path = self.data_dir / self.target_file
        logger.info("Loading target from %s", path)
        df = pd.read_csv(path)
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in {path}. "
                f"Available: {list(df.columns)}"
            )
        return df

    def _load_dense_obs(self) -> pd.DataFrame:
        """Load gridded observations, aggregate to daily, rename to standard cols."""
        path = self.ts_dir / self.dense_obs_file
        logger.info("Loading dense observations from %s", path)
        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(df[self.dense_obs_timestamp_col]).dt.date
        df = df.dropna(subset=[self.dense_obs_value_col])
        df = df[df[self.dense_obs_value_col] > 0]

        daily = (
            df.groupby(["grid_id", "date"])[self.dense_obs_value_col]
            .mean()
            .reset_index()
            .rename(columns={self.dense_obs_value_col: "obs_dense"})
        )
        logger.info(
            "Dense obs: %d rows, %d unique dates, %d grid cells",
            len(daily),
            daily["date"].nunique(),
            daily["grid_id"].nunique(),
        )
        return daily

    def _load_point_obs(self) -> pd.DataFrame:
        """Load station observations, aggregate to daily, rename to standard cols."""
        path = self.ts_dir / self.point_obs_file
        logger.info("Loading point observations from %s", path)
        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(
            df[self.point_obs_timestamp_col], utc=True, errors="coerce"
        ).dt.date
        df = df.dropna(subset=["date", self.point_obs_value_col])
        df = df[df[self.point_obs_value_col] > 0]

        daily = (
            df.groupby([self.point_obs_station_col, "grid_id", "date"])[self.point_obs_value_col]
            .mean()
            .reset_index()
            .rename(columns={
                self.point_obs_station_col: "station_id",
                self.point_obs_value_col: "obs_value",
            })
        )
        logger.info(
            "Point obs: %d rows, %d stations",
            len(daily),
            daily["station_id"].nunique(),
        )
        return daily

    def _load_activity_forcing(self) -> pd.DataFrame:
        """Load activity time series and compute normalised daily anomaly.

        Anomaly: Δ(t) = (daily_mean − period_mean) / period_mean
        Positive Δ → above-average activity → positive forcing on pollutant.
        """
        path = self.ts_dir / self.activity_file
        logger.info("Loading activity forcing from %s", path)
        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(
            df[self.activity_timestamp_col], errors="coerce"
        ).dt.date
        df = df.dropna(subset=["date", self.activity_value_col])

        daily = (
            df.groupby("date")[self.activity_value_col]
            .mean()
            .reset_index()
            .rename(columns={self.activity_value_col: "activity_mean"})
        )
        period_mean = daily["activity_mean"].mean()
        daily["delta_activity"] = (
            (daily["activity_mean"] - period_mean) / (period_mean + 1e-9)
        )
        logger.info(
            "Activity forcing: %d days, period mean=%.1f",
            len(daily), period_mean,
        )
        return daily[["date", "activity_mean", "delta_activity"]]

    def _load_met_forcing(self) -> pd.DataFrame:
        """Load meteorological forcing and compute per-cell scalar W(s,t).

        For wind data with sector structure:
          W(s,t) = Σ_k freq_k(s,t) × speed_k(s,t)

        This is the frequency-weighted mean wind speed across all sectors,
        giving a single scalar that represents how strongly wind is flushing
        each cell on each day. Any met forcing data with the same sector
        structure (freq + speed per sector) is supported.
        """
        path = self.ts_dir / self.met_forcing_file
        logger.info("Loading meteorological forcing from %s", path)
        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(df[self.met_date_col]).dt.date

        freq_cols = [
            f"{self.met_freq_prefix}{k}{self.met_freq_suffix}"
            for k in range(self.met_n_sectors)
        ]
        speed_cols = [
            f"{self.met_speed_prefix}{k}{self.met_speed_suffix}"
            for k in range(self.met_n_sectors)
        ]

        # Validate columns exist
        missing = [c for c in freq_cols + speed_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Met forcing columns missing from {path}: {missing[:5]}..."
            )

        freq = df[freq_cols].values
        speed = np.nan_to_num(df[speed_cols].values, nan=0.0)

        # Normalise frequencies to sum to 1 per row
        freq_sum = freq.sum(axis=1, keepdims=True)
        freq_norm = np.where(freq_sum > 0, freq / (freq_sum + 1e-9), 1.0 / self.met_n_sectors)

        df["met_forcing"] = (freq_norm * speed).sum(axis=1)

        if "grid_id" in df.columns:
            result = df[["grid_id", "date", "met_forcing"]]
            logger.info(
                "Met forcing: %d rows, %d unique dates, %d grid cells",
                len(result), result["date"].nunique(), result["grid_id"].nunique(),
            )
        else:
            result = df[["date", "met_forcing"]]
            logger.info(
                "Met forcing (city-wide): %d rows, %d unique dates",
                len(result), result["date"].nunique(),
            )
        return result

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def summary(self) -> str:
        """Print a summary of what data is available in this dataset."""
        lines = [
            "SpatiotemporalDataset",
            "=" * 50,
            f"  data_dir        : {self.data_dir}",
            f"  time_series_dir : {self.ts_dir}",
            "",
            "  Static files:",
            f"    features  → {self.features_file}  (target col: {self.target_col})",
            f"    target    → {self.target_file}",
            "",
            "  Temporal files:",
            f"    dense obs → {self.dense_obs_file}",
            f"               col: {self.dense_obs_value_col}",
            f"    point obs → {self.point_obs_file}",
            f"               col: {self.point_obs_value_col}  (validation only)",
            f"    activity  → {self.activity_file}",
            f"               col: {self.activity_value_col}",
            f"    met forc. → {self.met_forcing_file}",
            f"               {self.met_n_sectors} sectors",
            "",
            "  Grid geometry:",
            f"    {self._grid_geojson or 'not configured'}",
        ]
        report = "\n".join(lines)
        print(report)
        return report
