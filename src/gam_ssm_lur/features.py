"""
Feature Selection and Extraction for Land Use Regression.

This module provides utilities for:
1. Multi-stage feature selection (correlation, VIF, importance-based)
2. Spatial feature extraction from OpenStreetMap data
3. Traffic and satellite data processing

References
----------
.. [1] Hoek, G., et al. (2008). A review of land-use regression models.
.. [2] Lautenschlager, F., et al. (2020). OpenLUR: Off-the-shelf air pollution
       modeling with open features and machine learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Results from feature selection pipeline.
    
    Attributes
    ----------
    selected_features : List[str]
        Names of selected features
    n_original : int
        Number of original features
    n_selected : int
        Number of selected features
    dropped_correlation : List[str]
        Features dropped due to high correlation
    dropped_vif : List[str]
        Features dropped due to high VIF
    dropped_importance : List[str]
        Features dropped due to low importance
    feature_importances : pd.DataFrame
        Importance scores for retained features
    """
    selected_features: List[str]
    n_original: int
    n_selected: int
    dropped_correlation: List[str]
    dropped_vif: List[str]
    dropped_importance: List[str]
    feature_importances: pd.DataFrame


class FeatureSelector:
    """Multi-stage feature selection pipeline for LUR models.
    
    Implements a three-stage pipeline:
    1. Correlation-based removal: Remove highly correlated features
    2. VIF filtering: Remove features with high variance inflation
    3. Importance-based selection: Select top features by RF importance
    
    Parameters
    ----------
    correlation_threshold : float
        Maximum allowed pairwise correlation (default 0.8)
    vif_threshold : float
        Maximum allowed VIF value (default 10.0)
    n_top_features : int
        Number of top features to select (default 30)
    force_keep : List[str], optional
        Features to always keep regardless of selection criteria
    random_state : int, optional
        Random seed for reproducibility
        
    Examples
    --------
    >>> selector = FeatureSelector(
    ...     correlation_threshold=0.8,
    ...     vif_threshold=10.0,
    ...     n_top_features=30,
    ...     force_keep=['traffic_volume', 'motorway_distance']
    ... )
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(f"Selected {selector.result_.n_selected} of {selector.result_.n_original} features")
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.8,
        vif_threshold: float = 10.0,
        n_top_features: int = 30,
        force_keep: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.n_top_features = n_top_features
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
    ) -> "FeatureSelector":
        """Fit the feature selector.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target values
        feature_names : list of str, optional
            Feature names. Inferred from DataFrame if available.
            
        Returns
        -------
        self : FeatureSelector
        """
        # Convert to DataFrame for easier handling
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
        logger.info("Stage 1: Correlation-based filtering")
        current_features = set(feature_names)
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
                            
        dropped_corr = list(to_remove)
        current_features -= to_remove
        logger.info(f"  Removed {len(dropped_corr)} correlated features, {len(current_features)} remaining")
        
        # Stage 2: VIF filtering
        logger.info("Stage 2: VIF filtering")
        X_current = X[list(current_features)].copy()
        
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
            
            if len(X_current.columns) <= self.n_top_features:
                break
                
        current_features = set(X_current.columns)
        logger.info(f"  Removed {len(dropped_vif)} high-VIF features, {len(current_features)} remaining")
        
        # Re-add force-keep features if removed
        for feat in self.force_keep:
            if feat in feature_names and feat not in current_features:
                current_features.add(feat)
                if feat in dropped_vif:
                    dropped_vif.remove(feat)
                    
        # Stage 3: Importance-based selection
        logger.info("Stage 3: Importance-based selection")
        X_current = X[list(current_features)].copy()
        
        from sklearn.ensemble import RandomForestRegressor
        
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_current, y)
        
        importances = pd.DataFrame({
            'feature': X_current.columns,
            'importance': rf.feature_importances_,
        }).sort_values('importance', ascending=False)
        
        # Select top features + force-keep
        top_features = set(importances.head(self.n_top_features)['feature'])
        top_features |= self.force_keep & current_features
        
        dropped_imp = [f for f in current_features if f not in top_features]
        selected_features = list(top_features)
        
        logger.info(f"  Selected {len(selected_features)} features")
        
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
        """Transform features using fitted selector.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
            
        Returns
        -------
        X_selected : pd.DataFrame
            Selected features
        """
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
        """Fit selector and transform features.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target values
        feature_names : list of str, optional
            Feature names
            
        Returns
        -------
        X_selected : pd.DataFrame
            Selected features
        """
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
        """Get human-readable summary of selection results."""
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


class FeatureExtractor:
    """Extract spatial features from OpenStreetMap and other sources.
    
    Provides utilities for computing LUR-style features:
    - Road network density within buffers
    - Land use composition
    - Distance to features
    - Traffic proximity
    
    Parameters
    ----------
    buffer_distances : List[int]
        Buffer distances in meters for density calculations
    road_types : List[str]
        OSM road types to include
    land_use_types : List[str]
        OSM land use types to include
        
    Examples
    --------
    >>> extractor = FeatureExtractor(
    ...     buffer_distances=[50, 100, 200, 500, 1000],
    ...     road_types=['motorway', 'primary', 'secondary', 'tertiary'],
    ... )
    >>> features = extractor.extract(points_gdf, road_network_gdf, land_use_gdf)
    """
    
    DEFAULT_BUFFER_DISTANCES = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
    DEFAULT_ROAD_TYPES = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential']
    DEFAULT_LAND_USE_TYPES = ['industrial', 'commercial', 'residential', 'retail']
    
    def __init__(
        self,
        buffer_distances: Optional[List[int]] = None,
        road_types: Optional[List[str]] = None,
        land_use_types: Optional[List[str]] = None,
    ):
        self.buffer_distances = buffer_distances or self.DEFAULT_BUFFER_DISTANCES
        self.road_types = road_types or self.DEFAULT_ROAD_TYPES
        self.land_use_types = land_use_types or self.DEFAULT_LAND_USE_TYPES
        
    def extract(
        self,
        points: "gpd.GeoDataFrame",
        roads: Optional["gpd.GeoDataFrame"] = None,
        land_use: Optional["gpd.GeoDataFrame"] = None,
        traffic_points: Optional["gpd.GeoDataFrame"] = None,
    ) -> pd.DataFrame:
        """Extract features for given points.
        
        Parameters
        ----------
        points : GeoDataFrame
            Points to extract features for
        roads : GeoDataFrame, optional
            Road network with 'highway' column for road type
        land_use : GeoDataFrame, optional
            Land use polygons with 'landuse' column
        traffic_points : GeoDataFrame, optional
            Traffic monitoring points
            
        Returns
        -------
        pd.DataFrame
            Feature matrix with one row per point
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            raise ImportError("geopandas and shapely required for feature extraction")
            
        features = pd.DataFrame(index=points.index)
        
        # Road density features
        if roads is not None:
            logger.info("Extracting road network features")
            road_features = self._extract_road_features(points, roads)
            features = pd.concat([features, road_features], axis=1)
            
        # Land use features
        if land_use is not None:
            logger.info("Extracting land use features")
            lu_features = self._extract_land_use_features(points, land_use)
            features = pd.concat([features, lu_features], axis=1)
            
        # Traffic features
        if traffic_points is not None:
            logger.info("Extracting traffic features")
            traffic_features = self._extract_traffic_features(points, traffic_points)
            features = pd.concat([features, traffic_features], axis=1)
            
        return features
        
    def _extract_road_features(
        self,
        points: "gpd.GeoDataFrame",
        roads: "gpd.GeoDataFrame",
    ) -> pd.DataFrame:
        """Extract road network density and distance features."""
        import geopandas as gpd
        
        features = {}
        
        for road_type in self.road_types:
            # Filter roads by type
            if 'highway' in roads.columns:
                roads_filtered = roads[roads['highway'] == road_type]
            else:
                roads_filtered = roads
                
            if len(roads_filtered) == 0:
                continue
                
            # Distance to nearest road
            dist_col = f"{road_type}_distance"
            features[dist_col] = points.geometry.apply(
                lambda p: roads_filtered.distance(p).min()
            )
            
            # Road length within buffers
            for buffer_dist in self.buffer_distances:
                col_name = f"{road_type}_{buffer_dist}m"
                
                def calc_road_length(point, buffer_dist=buffer_dist):
                    buffer = point.buffer(buffer_dist)
                    clipped = gpd.clip(roads_filtered, buffer)
                    return clipped.length.sum() if len(clipped) > 0 else 0
                    
                features[col_name] = points.geometry.apply(calc_road_length)
                
        return pd.DataFrame(features, index=points.index)
        
    def _extract_land_use_features(
        self,
        points: "gpd.GeoDataFrame",
        land_use: "gpd.GeoDataFrame",
    ) -> pd.DataFrame:
        """Extract land use composition features."""
        import geopandas as gpd
        
        features = {}
        
        for lu_type in self.land_use_types:
            # Filter land use
            if 'landuse' in land_use.columns:
                lu_filtered = land_use[land_use['landuse'] == lu_type]
            else:
                lu_filtered = land_use
                
            if len(lu_filtered) == 0:
                continue
                
            # Distance to nearest land use
            dist_col = f"{lu_type}_distance"
            features[dist_col] = points.geometry.apply(
                lambda p: lu_filtered.distance(p).min() if len(lu_filtered) > 0 else np.inf
            )
            
            # Area within buffers
            for buffer_dist in self.buffer_distances[:6]:  # Use smaller set for land use
                col_name = f"{lu_type}_{buffer_dist}m"
                
                def calc_area(point, buffer_dist=buffer_dist):
                    buffer = point.buffer(buffer_dist)
                    clipped = gpd.clip(lu_filtered, buffer)
                    return clipped.area.sum() if len(clipped) > 0 else 0
                    
                features[col_name] = points.geometry.apply(calc_area)
                
        return pd.DataFrame(features, index=points.index)
        
    def _extract_traffic_features(
        self,
        points: "gpd.GeoDataFrame",
        traffic_points: "gpd.GeoDataFrame",
    ) -> pd.DataFrame:
        """Extract traffic-related features."""
        features = {}
        
        # Distance to nearest traffic monitoring point
        features['traffic_distance'] = points.geometry.apply(
            lambda p: traffic_points.distance(p).min()
        )
        
        # Find nearest traffic point for each location
        nearest_idx = points.geometry.apply(
            lambda p: traffic_points.distance(p).idxmin()
        )
        features['nearest_traffic_id'] = nearest_idx
        
        return pd.DataFrame(features, index=points.index)
        
    @staticmethod
    def from_osm(
        bbox: Tuple[float, float, float, float],
        buffer_distances: Optional[List[int]] = None,
    ) -> Tuple["FeatureExtractor", "gpd.GeoDataFrame", "gpd.GeoDataFrame"]:
        """Create extractor and download data from OpenStreetMap.
        
        Parameters
        ----------
        bbox : tuple
            Bounding box as (west, south, east, north)
        buffer_distances : list of int, optional
            Buffer distances for feature extraction
            
        Returns
        -------
        extractor : FeatureExtractor
            Configured feature extractor
        roads : GeoDataFrame
            Road network
        land_use : GeoDataFrame
            Land use polygons
        """
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("osmnx required for OSM data download. Install with: pip install osmnx")
            
        logger.info(f"Downloading OSM data for bbox: {bbox}")
        
        # Download road network
        west, south, east, north = bbox
        roads = ox.features_from_bbox(
            north=north, south=south, east=east, west=west,
            tags={'highway': True}
        )
        
        # Download land use
        land_use = ox.features_from_bbox(
            north=north, south=south, east=east, west=west,
            tags={'landuse': True}
        )
        
        extractor = FeatureExtractor(buffer_distances=buffer_distances)
        
        return extractor, roads, land_use
