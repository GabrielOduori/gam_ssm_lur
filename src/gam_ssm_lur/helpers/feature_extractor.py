
"""
Spatial feature extraction from OpenStreetMap data
Ideally, features are extracted and are ready when running 
this module.
Features extracted here are also used in other projects so
makes sense to keep them independent.
"""

# class FeatureExtractor:
#     """Extract spatial features from OpenStreetMap and other sources.
    
#     Provides utilities for computing LUR-style features:
#     - Road network density within buffers
#     - Land use composition
#     - Distance to features
#     - Traffic proximity
    
#     Parameters
#     ----------
#     buffer_distances : List[int]
#         Buffer distances in meters for density calculations
#     road_types : List[str]
#         OSM road types to include
#     land_use_types : List[str]
#         OSM land use types to include
        
#     Examples
#     --------
#     >>> extractor = FeatureExtractor(
#     ...     buffer_distances=[50, 100, 200, 500, 1000],
#     ...     road_types=['motorway', 'primary', 'secondary', 'tertiary'],
#     ... )
#     >>> features = extractor.extract(points_gdf, road_network_gdf, land_use_gdf)
#     """
    
#     DEFAULT_BUFFER_DISTANCES = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
#     DEFAULT_ROAD_TYPES = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential']
#     DEFAULT_LAND_USE_TYPES = ['industrial', 'commercial', 'residential', 'retail']
    
#     def __init__(
#         self,
#         buffer_distances: Optional[List[int]] = None,
#         road_types: Optional[List[str]] = None,
#         land_use_types: Optional[List[str]] = None,
#     ):
#         self.buffer_distances = buffer_distances or self.DEFAULT_BUFFER_DISTANCES
#         self.road_types = road_types or self.DEFAULT_ROAD_TYPES
#         self.land_use_types = land_use_types or self.DEFAULT_LAND_USE_TYPES
        
#     def extract(
#         self,
#         points: "gpd.GeoDataFrame",
#         roads: Optional["gpd.GeoDataFrame"] = None,
#         land_use: Optional["gpd.GeoDataFrame"] = None,
#         traffic_points: Optional["gpd.GeoDataFrame"] = None,
#     ) -> pd.DataFrame:
#         """Extract features for given points.
        
#         Parameters
#         ----------
#         points : GeoDataFrame
#             Points to extract features for
#         roads : GeoDataFrame, optional
#             Road network with 'highway' column for road type
#         land_use : GeoDataFrame, optional
#             Land use polygons with 'landuse' column
#         traffic_points : GeoDataFrame, optional
#             Traffic monitoring points
            
#         Returns
#         -------
#         pd.DataFrame
#             Feature matrix with one row per point
#         """
#         try:
#             import geopandas as gpd
#             from shapely.geometry import Point
#         except ImportError:
#             raise ImportError("geopandas and shapely required for feature extraction")
            
#         features = pd.DataFrame(index=points.index)
        
#         # Road density features
#         if roads is not None:
#             logger.info("Extracting road network features")
#             road_features = self._extract_road_features(points, roads)
#             features = pd.concat([features, road_features], axis=1)
            
#         # Land use features
#         if land_use is not None:
#             logger.info("Extracting land use features")
#             lu_features = self._extract_land_use_features(points, land_use)
#             features = pd.concat([features, lu_features], axis=1)
            
#         # Traffic features
#         if traffic_points is not None:
#             logger.info("Extracting traffic features")
#             traffic_features = self._extract_traffic_features(points, traffic_points)
#             features = pd.concat([features, traffic_features], axis=1)
            
#         return features
        
    # def _extract_road_features(
    #     self,
    #     points: "gpd.GeoDataFrame",
    #     roads: "gpd.GeoDataFrame",
    # ) -> pd.DataFrame:
    #     """Extract road network density and distance features."""
    #     import geopandas as gpd
        
    #     features = {}
        
    #     for road_type in self.road_types:
    #         # Filter roads by type
    #         if 'highway' in roads.columns:
    #             roads_filtered = roads[roads['highway'] == road_type]
    #         else:
    #             roads_filtered = roads
                
    #         if len(roads_filtered) == 0:
    #             continue
                
    #         # Distance to nearest road
    #         dist_col = f"{road_type}_distance"
    #         features[dist_col] = points.geometry.apply(
    #             lambda p: roads_filtered.distance(p).min()
    #         )
            
    #         # Road length within buffers
    #         for buffer_dist in self.buffer_distances:
    #             col_name = f"{road_type}_{buffer_dist}m"
                
    #             def calc_road_length(point, buffer_dist=buffer_dist):
    #                 buffer = point.buffer(buffer_dist)
    #                 clipped = gpd.clip(roads_filtered, buffer)
    #                 return clipped.length.sum() if len(clipped) > 0 else 0
                    
    #             features[col_name] = points.geometry.apply(calc_road_length)
                
    #     return pd.DataFrame(features, index=points.index)
        
    # def _extract_land_use_features(
    #     self,
    #     points: "gpd.GeoDataFrame",
    #     land_use: "gpd.GeoDataFrame",
    # ) -> pd.DataFrame:
    #     """Extract land use composition features."""
    #     import geopandas as gpd
        
    #     features = {}
        
    #     for lu_type in self.land_use_types:
    #         # Filter land use
    #         if 'landuse' in land_use.columns:
    #             lu_filtered = land_use[land_use['landuse'] == lu_type]
    #         else:
    #             lu_filtered = land_use
                
    #         if len(lu_filtered) == 0:
    #             continue
                
    #         # Distance to nearest land use
    #         dist_col = f"{lu_type}_distance"
    #         features[dist_col] = points.geometry.apply(
    #             lambda p: lu_filtered.distance(p).min() if len(lu_filtered) > 0 else np.inf
    #         )
            
    #         # Area within buffers
    #         for buffer_dist in self.buffer_distances[:6]:  # Use smaller set for land use
    #             col_name = f"{lu_type}_{buffer_dist}m"
                
    #             def calc_area(point, buffer_dist=buffer_dist):
    #                 buffer = point.buffer(buffer_dist)
    #                 clipped = gpd.clip(lu_filtered, buffer)
    #                 return clipped.area.sum() if len(clipped) > 0 else 0
                    
    #             features[col_name] = points.geometry.apply(calc_area)
                
    #     return pd.DataFrame(features, index=points.index)
        
    # def _extract_traffic_features(
    #     self,
    #     points: "gpd.GeoDataFrame",
    #     traffic_points: "gpd.GeoDataFrame",
    # ) -> pd.DataFrame:
    #     """Extract traffic-related features."""
    #     features = {}
        
    #     # Distance to nearest traffic monitoring point
    #     features['traffic_distance'] = points.geometry.apply(
    #         lambda p: traffic_points.distance(p).min()
    #     )
        
    #     # Find nearest traffic point for each location
    #     nearest_idx = points.geometry.apply(
    #         lambda p: traffic_points.distance(p).idxmin()
    #     )
    #     features['nearest_traffic_id'] = nearest_idx
        
    #     return pd.DataFrame(features, index=points.index)
        
    # @staticmethod
    # def from_osm(
    #     bbox: Tuple[float, float, float, float],
    #     buffer_distances: Optional[List[int]] = None,
    # ) -> Tuple["FeatureExtractor", "gpd.GeoDataFrame", "gpd.GeoDataFrame"]:
    #     """Create extractor and download data from OpenStreetMap.
        
    #     Parameters
    #     ----------
    #     bbox : tuple
    #         Bounding box as (west, south, east, north)
    #     buffer_distances : list of int, optional
    #         Buffer distances for feature extraction
            
    #     Returns
    #     -------
    #     extractor : FeatureExtractor
    #         Configured feature extractor
    #     roads : GeoDataFrame
    #         Road network
    #     land_use : GeoDataFrame
    #         Land use polygons
    #     """
    #     try:
    #         import osmnx as ox
    #     except ImportError:
    #         raise ImportError("osmnx required for OSM data download. Install with: pip install osmnx")
            
    #     logger.info(f"Downloading OSM data for bbox: {bbox}")
        
    #     # Download road network
    #     west, south, east, north = bbox
    #     roads = ox.features_from_bbox(
    #         north=north, south=south, east=east, west=west,
    #         tags={'highway': True}
    #     )
        
    #     # Download land use
    #     land_use = ox.features_from_bbox(
    #         north=north, south=south, east=east, west=west,
    #         tags={'landuse': True}
    #     )
        
    #     extractor = FeatureExtractor(buffer_distances=buffer_distances)
        
    #     return extractor, roads, land_use
