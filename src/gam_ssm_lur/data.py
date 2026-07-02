"""Spatiotemporal data loader for GAM-SSM-LUR.

Input contract: features.csv, target.csv, and time_series/ sub-directory
containing dense gridded observations (Kalman update), sparse point observations
(validation), city-wide activity forcing, and meteorological forcing. All file
and column names are configurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)


# Data containers
# ---------------


@dataclass
class StaticData:
    """Static (time-invariant) GAM-LUR inputs."""

    features: pd.DataFrame
    target: pd.DataFrame
    grid_ids: List[str]


@dataclass
class TemporalData:
    """Time-varying SSM inputs."""

    dense_obs: pd.DataFrame
    point_obs: pd.DataFrame
    activity_forcing: pd.DataFrame
    met_forcing: pd.DataFrame
    dates: List


@dataclass
class CalibrationResult:
    """Coefficients from OLS satellite-to-surface bias correction.

    OLS on collocated satellite and station observations — fits
    surface = beta0 + beta1 * satellite.
    """

    beta0: float
    beta1: float
    sigma2_obs: float
    n_collocated: int
    r: float
    collocated: Optional[pd.DataFrame] = None

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Apply calibration: raw satellite → surface-equivalent."""
        return self.beta0 + self.beta1 * np.asarray(raw)


# Main loader
# -----------


class SpatiotemporalDataset:
    """Loader for spatiotemporal LUR data. All file/column names are configurable."""

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
        # Overpass window: hours (inclusive) over which satellite, EPA, and
        # traffic observations are averaged before entering the model.
        overpass_window_start: int = 11,
        overpass_window_end: int = 14,
        # Grid geometry
        grid_geojson: Optional[str] = "grid/grid.geojson",
    ):
        self.data_dir = Path(data_dir)
        self.ts_dir = (
            Path(time_series_dir) if time_series_dir else self.data_dir / "time_series"
        )

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

        self.overpass_window_start = overpass_window_start
        self.overpass_window_end = overpass_window_end

        self._grid_geojson = self.data_dir / grid_geojson if grid_geojson else None

    # Public API
    # ----------

    def load_static(self) -> StaticData:
        """Load features and target for the static GAM-LUR component."""
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
        """Load all time-varying data for the SSM component."""
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
        """OLS fit of dense obs to point obs: surface = β₀ + β₁·satellite."""
        sta_grids = temporal.point_obs[["station_id", "grid_id"]].drop_duplicates()
        merged = (
            temporal.dense_obs.merge(sta_grids, on="grid_id", how="inner")
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
                n,
                min_collocated,
            )
            return CalibrationResult(
                beta0=0.0,
                beta1=1.0,
                sigma2_obs=36.0,
                n_collocated=n,
                r=float("nan"),
                collocated=merged,
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
            n,
            beta0,
            beta1,
            r,
            sigma2_obs,
        )
        return CalibrationResult(
            beta0=round(beta0, 4),
            beta1=round(beta1, 4),
            sigma2_obs=round(sigma2_obs, 2),
            n_collocated=n,
            r=round(r, 4),
            collocated=merged,
        )

    def load_grid_geometry(self) -> Optional[gpd.GeoDataFrame]:
        """Load polygon grid GeoJSON; returns None if file not found."""
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

    # Private loading methods
    # -----------------------

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
        """Load satellite obs averaged over the overpass window per (grid_id, date)."""
        path = self.ts_dir / self.dense_obs_file
        logger.info("Loading dense observations from %s", path)
        df = pd.read_csv(path)

        ts = pd.to_datetime(df[self.dense_obs_timestamp_col]).dt.floor("h")
        df["date"] = ts.dt.date
        df["hour"] = ts.dt.hour
        df = df.dropna(subset=[self.dense_obs_value_col])
        df = df[df[self.dense_obs_value_col] > 0]

        # Keep only observations within the overpass window
        df = df[
            (df["hour"] >= self.overpass_window_start)
            & (df["hour"] <= self.overpass_window_end)
        ]

        out = (
            df.groupby(["grid_id", "date"])
            .agg(obs_dense=(self.dense_obs_value_col, "mean"))
            .reset_index()
        )
        logger.info(
            "Dense obs: %d rows, %d unique dates, %d grid cells "
            "(window %02d:00–%02d:00)",
            len(out),
            out["date"].nunique(),
            out["grid_id"].nunique(),
            self.overpass_window_start,
            self.overpass_window_end,
        )
        return out

    def _load_point_obs(self) -> pd.DataFrame:
        """Load station obs averaged over the overpass window per (station_id, date)."""
        path = self.ts_dir / self.point_obs_file
        logger.info("Loading point observations from %s", path)
        df = pd.read_csv(path)

        ts = pd.to_datetime(df[self.point_obs_timestamp_col], utc=True, errors="coerce")
        df["date"] = ts.dt.date
        df["hour"] = ts.dt.hour
        df = df.dropna(subset=["date", "hour", self.point_obs_value_col])
        df = df[df[self.point_obs_value_col] > 0]

        # Keep only readings within the overpass window
        df = df[
            (df["hour"] >= self.overpass_window_start)
            & (df["hour"] <= self.overpass_window_end)
        ]

        out = (
            df.groupby([self.point_obs_station_col, "grid_id", "date"])
            .agg(obs_value=(self.point_obs_value_col, "mean"))
            .reset_index()
            .rename(columns={self.point_obs_station_col: "station_id"})
        )
        logger.info(
            "Point obs (window %02d:00–%02d:00 mean): %d rows, %d stations",
            self.overpass_window_start,
            self.overpass_window_end,
            len(out),
            out["station_id"].nunique(),
        )
        return out

    def _load_activity_forcing(self) -> pd.DataFrame:
        """Load overpass-window traffic volume; compute day-to-day anomaly Δ(t)."""
        path = self.ts_dir / self.activity_file
        logger.info("Loading activity forcing from %s", path)
        df = pd.read_csv(path)

        ts = pd.to_datetime(df[self.activity_timestamp_col], errors="coerce")
        df["date"] = ts.dt.date
        df["hour"] = ts.dt.hour
        df = df.dropna(subset=["date", "hour", self.activity_value_col])

        # Keep only readings within the overpass window
        df = df[
            (df["hour"] >= self.overpass_window_start)
            & (df["hour"] <= self.overpass_window_end)
        ]

        hourly = (
            df.groupby("date")
            .agg(activity_mean=(self.activity_value_col, "mean"))
            .reset_index()
        )
        period_mean = hourly["activity_mean"].mean()
        hourly["delta_activity"] = (hourly["activity_mean"] - period_mean) / (
            period_mean + 1e-9
        )
        logger.info(
            "Activity forcing (window %02d:00–%02d:00 mean): %d days, period mean=%.1f",
            self.overpass_window_start,
            self.overpass_window_end,
            len(hourly),
            period_mean,
        )
        return hourly[["date", "activity_mean", "delta_activity"]]

    def _load_met_forcing(self) -> pd.DataFrame:
        """Met forcing: W(s,t) = Σ_k freq_k × speed_k (frequency-weighted wind)."""
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
        freq_norm = np.where(
            freq_sum > 0, freq / (freq_sum + 1e-9), 1.0 / self.met_n_sectors
        )

        df["met_forcing"] = (freq_norm * speed).sum(axis=1)

        if "grid_id" in df.columns:
            result = df[["grid_id", "date", "met_forcing"]]
            logger.info(
                "Met forcing: %d rows, %d unique dates, %d grid cells",
                len(result),
                result["date"].nunique(),
                result["grid_id"].nunique(),
            )
        else:
            result = df[["date", "met_forcing"]]
            logger.info(
                "Met forcing (city-wide): %d rows, %d unique dates",
                len(result),
                result["date"].nunique(),
            )
        return result

    # Convenience
    # -----------

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
