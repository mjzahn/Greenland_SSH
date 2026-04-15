"""
Step 2: Interpolate AR7W density profiles onto a consistent transect
grid and apply a GEBCO bathymetry mask.

Workflow:
  1. Define a fixed SW→NE reference transect, sampled every DX_KM km.
  2. Extract GEBCO bathymetry along the reference transect nodes and save a
     lightweight transect-bathy file (done once; reused on subsequent runs).
  3. For each year:
       a. Project CTD casts onto the reference transect (distance_km).
       b. Interpolate every variable at every pressure level across the full
          range of valid casts for that year (continuous field, no gap masking).
       c. Nodes outside that year's cast range → NaN.
  4. Apply the bathymetry mask: set values to NaN wherever pressure exceeds
     the local seafloor depth (1 dbar ≈ 1 m, so pressure > |bathy_depth_m|).
  5. Save final dataset with bathy included as a variable.

Output:
    ar7w_density_transects.nc
    Dimensions : (year, distance_km, pressure)
    Coordinates: year, distance_km, pressure,
                 latitude(distance_km), longitude(distance_km)
    Variables  : density, sigma0, sigma1, temperature, salinity,
                 absolute_salinity, conservative_temperature,
                 bottom_depth(distance_km)

Requirements:
    pip install gsw xarray numpy scipy
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
IN_FILE   = "~/efs-mount-point/mzahn/data/ar7w/ar7w_density_profiles.nc"
OUT_FILE  = "~/efs-mount-point/mzahn/data/ar7w/ar7w_density_transects.nc"
BATHY_FILE = "~/efs-mount-point/mzahn/data/bathy/gebco_2021_sub_ice_n90.0_s30.0_w-120.0_e30.0.nc"
BATHY_TRANSECT_FILE = "~/efs-mount-point/mzahn/data/ar7w/ar7w_bathy_transect.nc"

DX_KM  = 10      # along-transect spacing in km (adjust as needed)

# AR7W reference end-points  (Labrador SW → Greenland NE)
LAT_SW, LON_SW = 53.56, -55.66   # Labrador shelf end
LAT_NE, LON_NE = 60.62, -48.17   # Greenland shelf end

# Buffer around the transect for GEBCO subsetting (degrees)
BATHY_BUFFER_DEG = 0.5

# Variables to interpolate onto the standard grid
VARS_TO_INTERP = [
    "density", "sigma0", "sigma1",
    "temperature", "salinity",
    "absolute_salinity", "conservative_temperature",
]
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_reference_transect(lat_sw, lon_sw, lat_ne, lon_ne, dx_km):
    """
    Build a great-circle reference transect sampled every dx_km.
    Returns arrays of (lat, lon, distance_km).
    """
    total_km = haversine_km(lat_sw, lon_sw, lat_ne, lon_ne)
    n_pts    = int(np.ceil(total_km / dx_km)) + 1
    fracs    = np.linspace(0, 1, n_pts)
    ref_lat  = lat_sw + fracs * (lat_ne - lat_sw)
    ref_lon  = lon_sw + fracs * (lon_ne - lon_sw)
    ref_dist = fracs * total_km
    return ref_lat, ref_lon, ref_dist


def project_casts_to_transect(cast_lats, cast_lons, ref_lat, ref_lon, ref_dist):
    """
    Snap each CTD cast to the nearest reference-transect node.
    Returns the transect distance (km) of each cast.
    """
    ref_xy  = np.column_stack([ref_lat, ref_lon])
    cast_xy = np.column_stack([cast_lats, cast_lons])
    tree    = cKDTree(ref_xy)
    _, idx  = tree.query(cast_xy)
    return ref_dist[idx]


def extract_bathy_along_transect(bathy_file, ref_lat, ref_lon,
                                  buffer_deg, out_file):
    """
    Subset GEBCO to a small box around the transect, save that subset,
    then extract depth at each reference node via nearest-neighbour lookup.
    Returns depth array (positive = metres below sea surface).
    """
    print("\nExtracting bathymetry along transect …")
    lat_min = ref_lat.min() - buffer_deg
    lat_max = ref_lat.max() + buffer_deg
    lon_min = ref_lon.min() - buffer_deg
    lon_max = ref_lon.max() + buffer_deg

    print(f"  Subsetting GEBCO to "
          f"lat [{lat_min:.2f}, {lat_max:.2f}], "
          f"lon [{lon_min:.2f}, {lon_max:.2f}] …")

    bathy_ds  = xr.open_dataset(bathy_file)
    lat_name  = "lat"       if "lat"       in bathy_ds.dims else "latitude"
    lon_name  = "lon"       if "lon"       in bathy_ds.dims else "longitude"
    elev_name = "elevation"   # standard GEBCO variable name

    bathy_sub = bathy_ds.sel(
        {lat_name: slice(lat_min, lat_max),
         lon_name: slice(lon_min, lon_max)}
    )

    # Save lightweight subset for reuse on subsequent runs
    bathy_sub.to_netcdf(out_file)
    print(f"  Saved bathy transect subset → {out_file}")

    depth_m = _nn_depth_from_subset(bathy_sub, lat_name, lon_name,
                                     elev_name, ref_lat, ref_lon)
    bathy_ds.close()
    return depth_m


def _nn_depth_from_subset(bathy_sub, lat_name, lon_name,
                           elev_name, ref_lat, ref_lon):
    """
    Nearest-neighbour depth extraction from an already-subsetted GEBCO dataset.
    Returns depth_m (positive = below sea surface).
    """
    bathy_lats = bathy_sub[lat_name].values
    bathy_lons = bathy_sub[lon_name].values
    elev_grid  = bathy_sub[elev_name].values  # (lat, lon), negative = below sea level

    bathy_lonlat = np.column_stack([
        np.repeat(bathy_lats, len(bathy_lons)),
        np.tile(bathy_lons,   len(bathy_lats)),
    ])
    tree    = cKDTree(bathy_lonlat)
    ref_xy  = np.column_stack([ref_lat, ref_lon])
    _, idx  = tree.query(ref_xy)

    depth_m = -elev_grid.ravel()[idx]   # elevation (negative) → depth (positive)
    depth_m = np.maximum(depth_m, 0.0)  # clamp land nodes to 0
    return depth_m


# ── 1. Load density profiles ──────────────────────────────────────────────────
ds        = xr.open_dataset(IN_FILE)
years     = np.unique(ds.time.dt.year.values)
pressures = ds["pressure"].values      # (pressure,) in dbar, every 5 dbar
n_press   = len(pressures)
print(f"Opened {IN_FILE}")
print(f"  Profiles : {ds.sizes['time']}   Pressure levels: {n_press}")

# ── 2. Build reference transect ───────────────────────────────────────────────
ref_lat, ref_lon, ref_dist = build_reference_transect(
    LAT_SW, LON_SW, LAT_NE, LON_NE, DX_KM
)
n_dist = len(ref_dist)
print(f"\nReference transect: {ref_dist[-1]:.0f} km total, {n_dist} nodes @ {DX_KM} km")

# ── 3. Extract / load bathymetry along transect ───────────────────────────────
if os.path.exists(BATHY_TRANSECT_FILE):
    print(f"\nLoading existing bathy transect subset: {BATHY_TRANSECT_FILE}")
    bathy_sub = xr.open_dataset(BATHY_TRANSECT_FILE)
    lat_name  = "lat"       if "lat"       in bathy_sub.dims else "latitude"
    lon_name  = "lon"       if "lon"       in bathy_sub.dims else "longitude"
    depth_m   = _nn_depth_from_subset(bathy_sub, lat_name, lon_name,
                                       "elevation", ref_lat, ref_lon)
    bathy_sub.close()
else:
    depth_m = extract_bathy_along_transect(
        BATHY_FILE, ref_lat, ref_lon, BATHY_BUFFER_DEG, BATHY_TRANSECT_FILE
    )

print(f"  Depth range along transect: {depth_m.min():.0f}–{depth_m.max():.0f} m")

# ── 4. Interpolate each year onto the standard grid ───────────────────────────
n_years    = len(years)
out_shape  = (n_years, n_dist, n_press)
out_arrays = {v: np.full(out_shape, np.nan) for v in VARS_TO_INTERP}

print()
for yi, year in enumerate(years):
    year_mask    = ds.time.dt.year.values == year
    ds_yr        = ds.isel(time=year_mask)
    n_casts      = ds_yr.sizes["time"]

    if n_casts < 2:
        print(f"  {year}: only {n_casts} cast(s) – skipping")
        continue

    cast_lats    = ds_yr["latitude"].values
    cast_lons    = ds_yr["longitude"].values
    cast_dist_km = project_casts_to_transect(
        cast_lats, cast_lons, ref_lat, ref_lon, ref_dist
    )

    # Sort by distance along transect
    sort_idx     = np.argsort(cast_dist_km)
    cast_dist_km = cast_dist_km[sort_idx]

    # Remove duplicate distance positions (keep first occurrence)
    _, uniq_idx  = np.unique(cast_dist_km, return_index=True)
    cast_dist_km = cast_dist_km[uniq_idx]
    orig_idx     = sort_idx[uniq_idx]

    d_min, d_max = cast_dist_km[0], cast_dist_km[-1]

    # Reference nodes within this year's outermost cast positions
    in_range = (ref_dist >= d_min) & (ref_dist <= d_max)

    for var in VARS_TO_INTERP:
        data_yr = ds_yr[var].values[orig_idx, :]  # (n_uniq_casts, pressure)

        for pi in range(n_press):
            col   = data_yr[:, pi]
            valid = ~np.isnan(col)
            # Need at least 2 valid casts at this pressure level to interpolate
            if valid.sum() < 2:
                continue
            f = interp1d(
                cast_dist_km[valid], col[valid],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            out_arrays[var][yi, in_range, pi] = f(ref_dist[in_range])

    print(f"  {year}: {n_casts} casts | dist {d_min:.0f}–{d_max:.0f} km | "
          f"{in_range.sum()} grid nodes filled")

# ── 5. Apply bathymetry mask ──────────────────────────────────────────────────
# depth_m[i] is the seafloor depth in metres at reference node i.
# Since 1 dbar ≈ 1 m, mask any grid cell where pressure > seafloor depth.
print("\nApplying bathymetry mask …")

# below_bathy shape: (distance_km, pressure)  →  True = below seafloor
pres_2d     = pressures[np.newaxis, :]    # (1, pressure)
depth_2d    = depth_m[:, np.newaxis]      # (distance_km, 1)
below_bathy = pres_2d > depth_2d         # (distance_km, pressure)

for var in VARS_TO_INTERP:
    # Broadcast mask over year axis
    out_arrays[var][:, below_bathy] = np.nan

print("  Done.")

# ── 6. Build output dataset ───────────────────────────────────────────────────
data_vars = {}
for var in VARS_TO_INTERP:
    attrs = ds[var].attrs if var in ds else {}
    data_vars[var] = xr.Variable(
        dims=["year", "distance_km", "pressure"],
        data=out_arrays[var],
        attrs=attrs,
    )

data_vars["bottom_depth"] = xr.Variable(
    dims=["distance_km"],
    data=depth_m,
    attrs={"long_name": "Seafloor depth from GEBCO 2021 sub-ice",
           "units": "m", "positive": "down"},
)

ds_out = xr.Dataset(
    data_vars,
    coords={
        "year":        ("year",        years),
        "distance_km": ("distance_km", ref_dist,
                        {"long_name": "Distance along AR7W transect from SW end",
                         "units": "km"}),
        "pressure":    ("pressure",    pressures,
                        {"long_name": "Sea pressure", "units": "dbar"}),
        "latitude":    ("distance_km", ref_lat,
                        {"long_name": "Latitude of reference transect node",
                         "units": "degrees_north"}),
        "longitude":   ("distance_km", ref_lon,
                        {"long_name": "Longitude of reference transect node",
                         "units": "degrees_east"}),
    },
    attrs={
        "title":        "AR7W Labrador Sea – Density transects on standard grid",
        "description":  (
            f"Density and hydrographic variables interpolated onto a coast-to-coast "
            f"reference transect ({DX_KM} km spacing, linear). NaN outside each "
            f"year's outermost cast positions and below GEBCO 2021 seafloor."
        ),
        "transect_SW":  f"({LAT_SW}°N, {LON_SW}°E)",
        "transect_NE":  f"({LAT_NE}°N, {LON_NE}°E)",
        "spacing_km":   DX_KM,
        "bathy_source": BATHY_FILE,
        "source":       IN_FILE,
    },
)

# ── 7. Save with compression ──────────────────────────────────────────────────
encoding = {
    v: {"zlib": True, "complevel": 4}
    for v in list(data_vars) + ["latitude", "longitude"]
}
ds_out.to_netcdf(OUT_FILE, encoding=encoding)
print(f"\nSaved transect dataset → {OUT_FILE}")
print(ds_out)
