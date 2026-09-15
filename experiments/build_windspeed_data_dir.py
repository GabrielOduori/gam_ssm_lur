#!/usr/bin/env python
"""Build a data/ directory with average wind speed merged into features.csv,
for re-running reproduce_paper.py with the wind-speed GAM feature adopted
(see compute_avg_wind_speed_feature.py for how that column is derived).

Everything except features.csv is symlinked from the original data
directory, so this doesn't duplicate the large satellite/OSM files.

Usage:
    python experiments/build_windspeed_data_dir.py \\
        --era5-nc /path/to/era5_land_wind_2023-06.nc \\
        --source-data-dir data \\
        --out-data-dir data_windspeed

Note: some CDS API downloads of the ERA5-Land NetCDF are actually ZIP
archives despite the .nc extension -- if `file <path>` reports "Zip
archive", unzip it first and point --era5-nc at the extracted data_0.nc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_avg_wind_speed_feature import compute_avg_wind_speed
from scipy.interpolate import RegularGridInterpolator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era5-nc", required=True)
    p.add_argument("--source-data-dir", default="data")
    p.add_argument("--out-data-dir", required=True)
    args = p.parse_args()

    src = Path(args.source_data_dir).resolve()
    out = Path(args.out_data_dir)
    out.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name == "features.csv":
            continue
        link = out / item.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(item)

    grid, lats, lons = compute_avg_wind_speed(args.era5_nc)
    lat_asc, grid_asc = lats[::-1], grid[::-1, :]
    interp = RegularGridInterpolator(
        (lat_asc, lons), grid_asc, bounds_error=False, fill_value=None
    )

    feat = pd.read_csv(src / "features.csv", low_memory=False)
    pts = feat[["latitude", "longitude"]].values
    feat["avg_wind_speed"] = interp(pts)
    feat.to_csv(out / "features.csv", index=False)

    print(f"Wrote {out / 'features.csv'} ({len(feat)} rows, avg_wind_speed added)")
    print(
        f"\nRun with:\n  python experiments/reproduce_paper.py --data-dir {out} --output-dir experiments/results/<name>"
    )


if __name__ == "__main__":
    main()
