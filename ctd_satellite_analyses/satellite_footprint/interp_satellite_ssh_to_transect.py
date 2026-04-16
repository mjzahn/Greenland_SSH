import glob
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# ── Config ────────────────────────────────────────────────────────────────────
DX_KM   = 10
LAT_SW, LON_SW = 53.56, -55.66
LAT_NE, LON_NE = 60.62, -48.17

data_path = Path("~/efs-mount-point/mzahn/data/satellite_data/ssh_cmems_yearly/").expanduser()
out_path  = Path("~/efs-mount-point/mzahn/data/satellite_data/ssh_cmems_ar7w/").expanduser()
out_path.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_reference_transect(lat_sw, lon_sw, lat_ne, lon_ne, dx_km):
    total_km = haversine_km(lat_sw, lon_sw, lat_ne, lon_ne)
    n_pts    = int(np.ceil(total_km / dx_km)) + 1
    fracs    = np.linspace(0, 1, n_pts)
    ref_lat  = lat_sw + fracs * (lat_ne - lat_sw)
    ref_lon  = lon_sw + fracs * (lon_ne - lon_sw)
    ref_dist = fracs * total_km
    return ref_lat, ref_lon, ref_dist

# ── Build transect once ───────────────────────────────────────────────────────
tr_lat, tr_lon, tr_dist = build_reference_transect(
    LAT_SW, LON_SW, LAT_NE, LON_NE, DX_KM
)
n_pts = len(tr_dist)
print(f"Transect: {n_pts} points, {tr_dist[-1]:.1f} km total")

# ── Process each yearly file ──────────────────────────────────────────────────
files = sorted(data_path.glob("ssh_cmems_l4_0pt125deg_50N_65N_60W_40W_*.nc"))
print(f"Found {len(files)} files\n")

for fpath in files:
    year = fpath.stem.split("_")[-1]          # extract year from filename
    print(f"Processing {year} ...", end=" ")

    ds = xr.open_dataset(fpath)
    ssh = ds["sla"]                            # (time, latitude, longitude)

    # Subset spatially to transect bounding box + small buffer (speeds up interp)
    buf = 0.5
    lat_min, lat_max = tr_lat.min() - buf, tr_lat.max() + buf
    lon_min, lon_max = tr_lon.min() - buf, tr_lon.max() + buf
    ssh_sub = ssh.sel(
        latitude =slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )

    lats = ssh_sub.latitude.values
    lons = ssh_sub.longitude.values

    # Pre-allocate output array (time, distance)
    n_time = ssh_sub.sizes["time"]
    ssh_transect = np.full((n_time, n_pts), np.nan, dtype=np.float32)

    # Interpolate each time step using RegularGridInterpolator
    # (much faster than xr.interp in a loop because we reuse the grid)
    for t in range(n_time):
        field = ssh_sub.isel(time=t).values          # (lat, lon)

        # Build interpolator — bounds_error=False returns NaN outside grid
        interp = RegularGridInterpolator(
            (lats, lons),
            field,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        ssh_transect[t, :] = interp(
            np.column_stack([tr_lat, tr_lon])
        )

    # ── Build output dataset ──────────────────────────────────────────────────
    ds_out = xr.Dataset(
        {
            "sla": xr.DataArray(
                ssh_transect,
                dims=["time", "distance_km"],
                attrs={
                    "long_name": "Sea level anomaly along AR7W transect",
                    "units"    : "m",
                    "source"   : f"Interpolated from CMEMS L4 SSH, {year}",
                },
            ),
        },
        coords={
            "time"       : ds["time"].values,
            "distance_km": ("distance_km", tr_dist,
                            {"long_name": "Along-transect distance", "units": "km"}),
            "latitude"   : ("distance_km", tr_lat,
                            {"long_name": "Latitude",  "units": "degrees_north"}),
            "longitude"  : ("distance_km", tr_lon,
                            {"long_name": "Longitude", "units": "degrees_east"}),
        },
    )

    ds_out.attrs.update({
        "description" : "CMEMS L4 SSH interpolated onto AR7W transect",
        "transect"    : "AR7W  SW(53.56N,55.66W) → NE(60.62N,48.17W)",
        "dx_km"       : DX_KM,
        "created_with": "scipy RegularGridInterpolator, linear",
    })

    # ── Save ──────────────────────────────────────────────────────────────────
    out_file = out_path / f"ssh_cmems_ar7w_{year}.nc"
    ds_out.to_netcdf(
        out_file,
        encoding={
            "sla": {"dtype": "float32", "zlib": True, "complevel": 4},
        },
    )
    ds.close()
    print(f"saved → {out_file.name}")

print("\nDone.")