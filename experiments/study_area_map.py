"""
Study area map — Dublin NO2 monitoring network.

Shows: EPA stations, SCATS detectors, study area extent,
       Ireland inset with study area bounding box.

Output: experiments/results/study_area_map.png
"""

import os

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pyproj import Transformer

os.chdir("/media/gabriel-oduori/SERVER/dev_space/gam_ssm_lur")

# ── Station data ──────────────────────────────────────────────
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

epa = pd.read_csv("data/time_series/epa_timeseries.csv")
epa_pts = epa[epa["station_id"].isin(MODEL_STATIONS)].drop_duplicates("station_id")[
    ["station_id", "latitude", "longitude"]
]

scats = pd.read_csv("data/time_series/traffic_timeseries.csv")
scats_pts = scats.drop_duplicates("site_id")[["site_id", "latitude", "longitude"]]

# ── GeoDataFrames → Web Mercator ─────────────────────────────
epa_gdf = gpd.GeoDataFrame(
    epa_pts,
    geometry=gpd.points_from_xy(epa_pts.longitude, epa_pts.latitude),
    crs="EPSG:4326",
).to_crs("EPSG:3857")

scats_gdf = gpd.GeoDataFrame(
    scats_pts,
    geometry=gpd.points_from_xy(scats_pts.longitude, scats_pts.latitude),
    crs="EPSG:4326",
).to_crs("EPSG:3857")

# ── Study area extent (WGS84 → Web Mercator) ──────────────────
LON_MIN, LON_MAX = -6.42, -6.13
LAT_MIN, LAT_MAX = 53.29, 53.42

tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = tf.transform(LON_MIN, LAT_MIN)
xmax, ymax = tf.transform(LON_MAX, LAT_MAX)

# ── Main figure ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))

# SCATS detectors
scats_gdf.plot(ax=ax, color="steelblue", markersize=5, alpha=0.6, zorder=3)

# EPA stations
epa_gdf.plot(ax=ax, color="#DC143C", marker="^", markersize=100, zorder=5)

# EPA labels
for _, row in epa_gdf.iterrows():
    ax.annotate(
        row["station_id"],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold",
        color="#DC143C",
        zorder=6,
    )

# Study area extent box
extent_rect = Rectangle(
    (xmin, ymin),
    xmax - xmin,
    ymax - ymin,
    linewidth=2,
    edgecolor="black",
    facecolor="none",
    linestyle="--",
    zorder=4,
)
ax.add_patch(extent_rect)

ax.set_xlim(xmin - 300, xmax + 300)
ax.set_ylim(ymin - 300, ymax + 300)

# Basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager, zoom=13, attribution=False)

# Degree tick labels (convert Web Mercator ticks back to WGS84)
tf_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
lon_ticks = [-6.40, -6.35, -6.30, -6.25, -6.20, -6.15]
lat_ticks = [53.30, 53.32, 53.34, 53.36, 53.38, 53.40, 53.42]
xtick_wm = [tf.transform(lon, 53.35)[0] for lon in lon_ticks]
ytick_wm = [tf.transform(-6.27, lat)[1] for lat in lat_ticks]
ax.set_xticks(xtick_wm)
ax.set_xticklabels([f"{lon:.2f}°" for lon in lon_ticks], fontsize=8)
ax.set_yticks(ytick_wm)
ax.set_yticklabels([f"{lat:.2f}°" for lat in lat_ticks], fontsize=8)

# Attribution
ax.text(
    0.01,
    0.01,
    "© OpenStreetMap contributors © CARTO",
    transform=ax.transAxes,
    fontsize=5,
    color="grey",
    va="bottom",
)

# ── Inset: Ireland overview ───────────────────────────────────
axins = ax.inset_axes([0.70, 0.02, 0.27, 0.27])

LON_INS_MIN, LON_INS_MAX = -10.8, -5.3
LAT_INS_MIN, LAT_INS_MAX = 51.2, 55.6
xins_min, yins_min = tf.transform(LON_INS_MIN, LAT_INS_MIN)
xins_max, yins_max = tf.transform(LON_INS_MAX, LAT_INS_MAX)

axins.set_xlim(xins_min, xins_max)
axins.set_ylim(yins_min, yins_max)
ctx.add_basemap(axins, source=ctx.providers.CartoDB.Voyager, zoom=6, attribution=False)

# Study area bounding box on inset
study_box = Rectangle(
    (xmin, ymin),
    xmax - xmin,
    ymax - ymin,
    linewidth=2,
    edgecolor="#DC143C",
    facecolor="none",
    zorder=5,
)
axins.add_patch(study_box)

axins.set_xticks([])
axins.set_yticks([])
axins.set_title("Ireland", fontsize=8, pad=2)
for spine in axins.spines.values():
    spine.set_linewidth(1.2)

# ── Legend ────────────────────────────────────────────────────
n_scats = len(scats_gdf)
n_epa = len(epa_gdf)
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="steelblue",
        markersize=7,
        alpha=0.7,
        label=f"SCATS detector (n={n_scats})",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor="#DC143C",
        markersize=9,
        label=f"EPA station (n={n_epa})",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Study area extent",
    ),
]
# ── Shared horizontal centre for scale bar / arrow / legend ───
cx = 0.14  # axes fraction — centre of the cluster

# ── Legend (bottom) ───────────────────────────────────────────
ax.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(cx, 0.02),
    borderaxespad=0,
    fontsize=8,
    framealpha=0.9,
    edgecolor="grey",
)

# ── Scale bar (~5 km) above legend ────────────────────────────
scale_m = 5000
total_x = (xmax + 300) - (xmin - 300)
scale_frac = scale_m / total_x
ax.plot(
    [cx - scale_frac / 2, cx + scale_frac / 2],
    [0.155, 0.155],
    "k-",
    lw=2.5,
    zorder=6,
    transform=ax.transAxes,
)
ax.text(cx, 0.140, "5 km", ha="center", fontsize=7, zorder=6, transform=ax.transAxes)

# ── North arrow above scale bar ───────────────────────────────
ax.annotate(
    "",
    xy=(cx, 0.22),
    xytext=(cx, 0.175),
    xycoords="axes fraction",
    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5},
)
ax.text(
    cx, 0.235, "N", ha="center", fontsize=9, fontweight="bold", transform=ax.transAxes
)

ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)

# ── Save ──────────────────────────────────────────────────────
out = "experiments/results/study_area_map.png"
plt.tight_layout()
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
