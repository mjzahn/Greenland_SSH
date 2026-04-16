"""
plot_ar7w_transects.py
──────────────────────
Open ar7w_density_transects.nc and produce a large multipanel figure
(one panel per year) showing sigma0 density contours with dark-gray
GEBCO bathymetry shading.  Output is sized for a PowerPoint slide
(widescreen 16:9, 200 dpi → ~2560 × 1440 px at 13.33 × 7.5 in).

Usage
─────
    python plot_ar7w_transects.py

Outputs
───────
    ar7w_density_transects_multipanel.png   (next to the input file)

Requirements
────────────
    pip install xarray numpy matplotlib cmocean
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cmocean                         # pip install cmocean  (optional but nice)

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
IN_FILE  = "~/efs-mount-point/mzahn/data/ar7w/ar7w_density_transects.nc"
OUT_FILE = "~/git_repos/Greenland_SSH/ctd_satellite_analyses/dyn_height_calcs/ar7w_density_transects_multipanel.png"

PLOT_VAR     = "sigma0"          # variable to contour-fill
BATHY_COLOR  = "#3a3a3a"         # dark gray for seafloor
BATHY_ALPHA  = 1.0
CMAP         = cmocean.cm.dense  # swap to plt.cm.viridis if cmocean missing
N_CONTOURS   = 20                # number of filled contour levels

# Pressure / depth axis limits (dbar, positive down)
P_MIN, P_MAX = 0, 3500

# y-axis label every N dbar
P_YTICK = 500

# Slide layout  (inches × dpi = pixels)
FIG_W_IN = 13.33
FIG_H_IN =  7.50
FIG_DPI  = 200
# ─────────────────────────────────────────────────────────────────────────────


def fill_bathy(ax, dist_km, bottom_depth, p_max):
    """
    Fill the region below the bathymetry profile with BATHY_COLOR as a
    solid polygon, from the seafloor down to p_max.
    """
    x = np.concatenate([[dist_km[0]], dist_km, [dist_km[-1]]])
    y = np.concatenate([[p_max], bottom_depth, [p_max]])
    ax.fill(x, y, color=BATHY_COLOR, alpha=BATHY_ALPHA, zorder=5)


def main():
    # ── Load dataset ──────────────────────────────────────────────────────────
    in_path = os.path.expanduser(IN_FILE)
    ds = xr.open_dataset(in_path)
    years        = ds["year"].values
    dist_km      = ds["distance_km"].values
    pressure     = ds["pressure"].values
    bottom_depth = ds["bottom_depth"].values   # (distance_km,)
    data_all     = ds[PLOT_VAR].values         # (year, distance_km, pressure)
    ds.close()

    # Clip pressure axis
    p_mask   = pressure <= P_MAX
    pressure = pressure[p_mask]
    data_all = data_all[:, :, p_mask]

    # ── Global colour range (ignore NaN) ──────────────────────────────────────
    vmin   = np.nanpercentile(data_all, 1)
    vmax   = np.nanpercentile(data_all, 99)
    levels = np.linspace(vmin, vmax, N_CONTOURS + 1)

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_years = len(years)
    n_cols  = int(np.ceil(np.sqrt(n_years * (FIG_W_IN / FIG_H_IN))))
    n_rows  = int(np.ceil(n_years / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_W_IN, FIG_H_IN),
        sharey=True, sharex=True,
        facecolor="white",
    )
    axes = np.array(axes).flatten()   # always 1-D

    fig.subplots_adjust(
        left=0.06, right=0.92,
        bottom=0.08, top=0.92,
        wspace=0.04, hspace=0.25,
    )

    cf_last = None  # keep last contourf handle for the shared colorbar

    for i, year in enumerate(years):
        ax   = axes[i]
        data = data_all[i]           # (distance_km, pressure)

        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

        # Contour-fill density
        cf = ax.contourf(
            dist_km, pressure, data.T,
            levels=levels, cmap=CMAP,
            vmin=vmin, vmax=vmax,
            extend="both", zorder=2,
        )
        # Thin density contour lines for readability
        ax.contour(
            dist_km, pressure, data.T,
            levels=levels[::2], colors="black",
            linewidths=0.25, alpha=0.25, zorder=3,
        )

        # Bathymetry shading
        bathy_plot = np.clip(bottom_depth, 0, P_MAX)
        fill_bathy(ax, dist_km, bathy_plot, P_MAX)

        # Year label inside panel
        ax.text(
            0.02, 0.13, str(year),
            transform=ax.transAxes,
            color="white", fontsize=8, fontweight="bold",
            va="top", ha="left", zorder=10,
        )

        # Y-axis: pressure increasing downward
        ax.set_ylim(P_MAX, P_MIN)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(P_YTICK))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))

        cf_last = cf

    # ── Hide unused panels ────────────────────────────────────────────────────
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # ── Shared axis labels ────────────────────────────────────────────────────
    fig.text(0.50, 0.01, "Distance along AR7W transect (km)",
             ha="center", va="bottom", color="black", fontsize=9)
    fig.text(0.01, 0.50, "Pressure (dbar)",
             ha="left", va="center", color="black", fontsize=9,
             rotation=90)

    # ── Colourbar ─────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.93, 0.10, 0.015, 0.80])   # [left, bottom, w, h]
    cbar    = fig.colorbar(cf_last, cax=cbar_ax, extend="both")
    cbar.set_label(
        rf"$\sigma_0$  (kg m$^{{-3}}$)",
        color="black", fontsize=8, labelpad=6,
    )
    cbar.ax.yaxis.set_tick_params(color="black", labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")
    cbar.outline.set_edgecolor("#cccccc")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        "AR7W Labrador Sea — Annual Density Transects",
        color="white", fontsize=11, fontweight="bold", y=0.97,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.expanduser(OUT_FILE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
