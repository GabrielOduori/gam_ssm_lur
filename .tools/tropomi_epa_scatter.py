"""
tropomi_epa_scatter.py
======================
Colocation scatter plot: TROPOMI VCD (mol/m²) vs EPA surface NO₂ (µg/m³).

Replicates the OLS calibration step used in the pipeline and visualises the
station-level relationship. Each point is one station–day pair collocated in
the same grid cell during the 11:00–13:00 UTC overpass window.

Output
------
.tools/outputs/tropomi_epa_calibration_scatter.png
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import lstsq

os.chdir(Path(__file__).resolve().parent.parent)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_STATIONS = [
    "EPA-11",
    "EPA-17",
    "EPA-22",
    "EPA-29",
    "EPA-49",
    "EPA-50",
    "EPA-57",
    "EPA-69",
    "EPA-76",
]
OVERPASS_START = 11  # UTC hour (inclusive)
OVERPASS_END = 13  # UTC hour (inclusive)
SCALE = 1e4  # display TROPOMI as ×10⁻⁴ mol/m²

# ── Load data ──────────────────────────────────────────────────────────────────
sat = pd.read_csv(
    "data/time_series/satellite_retreavals.csv",
    parse_dates=["timestamp"],
    usecols=["grid_id", "timestamp", "tropomi_no2"],
)
epa = pd.read_csv(
    "data/time_series/epa_timeseries.csv",
    parse_dates=["timestamp_utc"],
    usecols=["station_id", "grid_id", "timestamp_utc", "epa_no2"],
)
epa = epa[epa["station_id"].isin(MODEL_STATIONS)].copy()

# ── TROPOMI: daily mean per grid cell (one overpass per day) ───────────────────
sat["date"] = sat["timestamp"].dt.date
sat_daily = sat.groupby(["grid_id", "date"])["tropomi_no2"].mean().reset_index()

# ── EPA: window-mean 11:00–13:00 UTC per station–day ──────────────────────────
epa["hour"] = epa["timestamp_utc"].dt.hour
epa["date"] = epa["timestamp_utc"].dt.date
epa_window = epa[(epa["hour"] >= OVERPASS_START) & (epa["hour"] <= OVERPASS_END)]
epa_daily = (
    epa_window.groupby(["station_id", "grid_id", "date"])["epa_no2"]
    .mean()
    .reset_index()
)

# ── Collocate: merge TROPOMI and EPA at the same grid cell and date ────────────
merged = epa_daily.merge(sat_daily, on=["grid_id", "date"], how="inner")
merged = merged.dropna(subset=["epa_no2", "tropomi_no2"])
n = len(merged)

# ── OLS: EPA (µg/m³) = β₀ + β₁ · TROPOMI_scaled (×10⁻⁴ mol/m²) ──────────────
x_raw = merged["tropomi_no2"].values  # mol/m²
x_scaled = x_raw * SCALE  # ×10⁻⁴ mol/m² (for display)
y = merged["epa_no2"].values  # µg/m³

X = np.column_stack([np.ones(n), x_scaled])
coeffs, _, _, _ = lstsq(X, y, cond=None)
beta0, beta1 = float(coeffs[0]), float(coeffs[1])

y_pred = beta0 + beta1 * x_scaled
r = float(np.corrcoef(y, x_scaled)[0, 1])
r2 = r**2

print(f"Collocated pairs : {n}")
print(f"β₀               : {beta0:.3f} µg/m³")
print(f"β₁               : {beta1:.3f} µg/m³ per (×10⁻⁴ mol/m²)")
print(f"r                : {r:.3f}")
print(f"r²               : {r2:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

MARKERS = ["o", "x", "+", "s", "^", "D", "v", "P", "*"]
for i, stn in enumerate(sorted(MODEL_STATIONS)):
    mask = merged["station_id"] == stn
    if mask.sum() == 0:
        continue
    ax.scatter(
        merged.loc[mask, "tropomi_no2"] * SCALE,
        merged.loc[mask, "epa_no2"],
        label=stn,
        marker=MARKERS[i],
        s=45,
        alpha=0.85,
        color="steelblue",
        zorder=3,
    )

# OLS fit line
x_line = np.linspace(x_scaled.min(), x_scaled.max(), 200)
ax.plot(
    x_line, beta0 + beta1 * x_line, color="black", lw=1.5, zorder=4, label="OLS fit"
)

# Annotation box
sign = "+" if beta0 >= 0 else "-"
eq_text = (
    f"$C_{{\\mathrm{{EPA}}}} = {beta1:.2f}\\,C_{{\\mathrm{{TROP}}}} {sign} {abs(beta0):.2f}$\n"
    f"$r^2 = {r2:.3f}$,  $N = {n}$"
)
ax.text(
    0.04,
    0.96,
    eq_text,
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    ha="left",
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "grey",
        "alpha": 0.85,
    },
)

ax.set_xlabel(
    r"TROPOMI VCD, $C_{\mathrm{TROP}}$ ($\times 10^{-4}$ mol/m$^{2}$)", fontsize=10
)
ax.set_ylabel(r"EPA surface NO$_2$ ($\mu$g/m$^{3}$)", fontsize=10)
ax.set_title("TROPOMI–EPA colocation calibration", fontsize=11)
ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.85)
ax.grid(True, linestyle="--", alpha=0.35)

plt.tight_layout()

out = ".tools/outputs/tropomi_epa_calibration_scatter.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
