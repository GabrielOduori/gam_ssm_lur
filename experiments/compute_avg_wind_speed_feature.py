#!/usr/bin/env python
"""Compute a per-cell average wind speed feature (11:00-14:00 UTC window,
June 2023) from raw ERA5-Land wind data, bilinear-interpolated onto the
100m model grid, for testing as a candidate GAM predictor.

Motivated by Naughton et al. (2018), whose final wind-sector LUR model
retained "average wind speed" as one of only 5 selected predictors
(coefficient -0.83, p<0.001) -- a feature absent from this project's
5 feature types. The domain-wide daily wind_sector CSV cannot be used
directly for this (it has no per-cell resolution, so would be a
zero-variance column); this instead goes back to the raw ERA5-Land
NetCDF, which retains coarse (3x5 point) spatial resolution across
the domain.

Usage:
    python experiments/compute_avg_wind_speed_feature.py \\
        --era5-nc /path/to/era5_land_wind_2023-06.nc \\
        --features-csv data/features.csv \\
        --out-csv avg_wind_speed_percell.csv

Note: some CDS API downloads of this file are actually ZIP archives
despite the .nc extension -- unzip first if `file` reports "Zip archive".
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


def compute_avg_wind_speed(
    era5_nc: str, window_start: int = 11, window_end: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(era5_nc)
    hours = pd.to_datetime(ds["valid_time"].values).hour
    window = (hours >= window_start) & (hours <= window_end)
    u = ds["u10"].values[window]
    v = ds["v10"].values[window]
    speed = np.sqrt(u**2 + v**2)
    mean_speed_grid = speed.mean(axis=0)  # (lat, lon), NaN over open water

    # ERA5-Land has no data over sea; nearest-neighbour fill so coastal
    # land cells don't inherit NaN from a diagonal sea-facing corner.
    mask = np.isnan(mean_speed_grid)
    if mask.any():
        ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
        mean_speed_grid = mean_speed_grid[tuple(ind)]

    return mean_speed_grid, ds["latitude"].values, ds["longitude"].values


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era5-nc", required=True)
    p.add_argument("--features-csv", required=True)
    p.add_argument("--out-csv", default="avg_wind_speed_percell.csv")
    args = p.parse_args()

    grid, lats, lons = compute_avg_wind_speed(args.era5_nc)

    lat_asc = lats[::-1]
    grid_asc = grid[::-1, :]
    interp = RegularGridInterpolator(
        (lat_asc, lons), grid_asc, bounds_error=False, fill_value=None
    )

    feat = pd.read_csv(args.features_csv, usecols=["grid_id", "latitude", "longitude"])
    pts = np.column_stack([feat["latitude"].values, feat["longitude"].values])
    feat["avg_wind_speed"] = interp(pts)

    print(feat["avg_wind_speed"].describe())
    feat[["grid_id", "avg_wind_speed"]].to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
