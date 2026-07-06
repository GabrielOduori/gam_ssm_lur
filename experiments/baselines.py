#!/usr/bin/env python
"""Naive temporal baselines against EPA station-day observations.

Persistence, station climatology, and domain climatology, all jackknifed to
avoid the target day informing its own baseline.

Usage:
    python experiments/baselines.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gam_ssm_lur.evaluation import ModelEvaluator

EXPERIMENTS = Path(__file__).resolve().parent


def gaussian_crps(y, mu, sigma):
    from scipy.stats import norm

    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    z = (np.asarray(y) - np.asarray(mu)) / sigma
    return float(
        np.mean(
            sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        )
    )


def metrics_row(name, y, yp, crps=None):
    ev = ModelEvaluator()
    a = ev.compute_accuracy(np.asarray(y), np.asarray(yp))
    return {
        "model": name,
        "rmse": a.rmse,
        "mae": a.mae,
        "mbe": a.mbe,
        "r2": a.r2,
        "r": a.correlation,
        "crps": crps,
        "n": len(y),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    p.add_argument(
        "--output", default=str(EXPERIMENTS / "results" / "baseline_metrics.csv")
    )
    args = p.parse_args()

    # use the pipeline's own loader so observations carry the same
    # overpass-window (11:00-14:00 UTC) aggregation as Table 6
    from gam_ssm_lur.data import SpatiotemporalDataset

    ds = SpatiotemporalDataset(
        data_dir=args.data_dir,
        target_col="atmos_no2",
        dense_obs_file="satellite_retreavals.csv",
        dense_obs_value_col="tropomi_no2",
        point_obs_file="epa_timeseries.csv",
        point_obs_value_col="epa_no2",
        activity_file="traffic_timeseries.csv",
        activity_value_col="traffic_volume",
        met_forcing_file="wind_sector_2023-06_daily.csv",
        grid_geojson="grid/grid.geojson",
    )
    ds.load_static()
    temporal = ds.load_temporal()
    df = temporal.point_obs[["station_id", "date", "obs_value"]].rename(
        columns={"obs_value": "obs"}
    )
    df = df.dropna(subset=["obs"])
    df = df.groupby(["station_id", "date"], as_index=False)["obs"].mean()
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    rows = []

    # Persistence
    # -----------
    df["obs_prev"] = df.groupby("station_id")["obs"].shift(1)
    pers = df.dropna(subset=["obs_prev"])
    rows.append(
        metrics_row(
            "persistence (today = yesterday)",
            pers["obs"],
            pers["obs_prev"],
            crps=float(np.mean(np.abs(pers["obs"] - pers["obs_prev"]))),
        )
    )

    # Station climatology
    # -------------------
    g = df.groupby("station_id")["obs"]
    n_s = g.transform("count")
    sum_s = g.transform("sum")
    mean_jack = (sum_s - df["obs"]) / (n_s - 1)
    # jackknifed std: std of other days (approximate via full std of others)
    sq_sum = g.transform(lambda x: (x**2).sum())
    var_jack = (sq_sum - df["obs"] ** 2) / (n_s - 1) - mean_jack**2
    std_jack = np.sqrt(np.maximum(var_jack, 1e-9))
    ok = n_s > 2
    rows.append(
        metrics_row(
            "station climatology (jackknifed mean)",
            df.loc[ok, "obs"],
            mean_jack[ok],
            crps=gaussian_crps(df.loc[ok, "obs"], mean_jack[ok], std_jack[ok]),
        )
    )

    # Domain climatology
    # ------------------
    N = len(df)
    total = df["obs"].sum()
    dom_mean = (total - df["obs"]) / (N - 1)
    dom_var = ((df["obs"] ** 2).sum() - df["obs"] ** 2) / (N - 1) - dom_mean**2
    dom_row = metrics_row(
        "domain climatology (jackknifed mean)",
        df["obs"],
        dom_mean,
        crps=gaussian_crps(df["obs"], dom_mean, np.sqrt(np.maximum(dom_var, 1e-9))),
    )
    # jackknifed constant is a decreasing linear function of the target, so its
    # correlation is -1 by construction and carries no information
    dom_row["r"] = np.nan
    rows.append(dom_row)

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
