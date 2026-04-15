"""
Step 1: Calculate seawater density from AR7W CTD profiles.

Reads the raw AR7W dataset, filters to May–August, computes in-situ
density using the TEOS-10 Gibbs SeaWater (gsw) toolbox, and saves a
new NetCDF file that preserves all spatial/temporal metadata.

"""

import numpy as np
import xarray as xr
import gsw

# ── 1. Open raw dataset ────────────────────────────────────────────────────────
ar7w_ds = xr.open_dataset(
    "~/efs-mount-point/mzahn/data/ar7w/igor_ar7w.nc"
)

# ── 2. Filter to May–August (months 5–8) ──────────────────────────────────────
mask = ar7w_ds["time"].dt.month.isin([5, 6, 7, 8])
ds = ar7w_ds.sel(time=mask)

print(f"Total profiles after May–Aug filter: {ds.sizes['time']}")
print(f"Year range: {ds.time.dt.year.values.min()} – {ds.time.dt.year.values.max()}")

# ── 3. Calculate density ───────────────────────────────────────────────────────
# pressure grid is the same for every profile (coordinate 'pressure', in dbar)
# latitude varies per cast – broadcast to (time, pressure) for gsw

lat  = ds["latitude"].values          # (time,)
lon  = ds["longitude"].values         # (time,)
pres = ds["pressure"].values          # (pressure,)   in dbar

SP   = ds["salinity"].values          # (time, pressure)   practical salinity
T    = ds["temperature"].values       # (time, pressure)   in-situ °C

# Broadcast latitude to (time, pressure)
lat_2d = np.broadcast_to(lat[:, np.newaxis], SP.shape)
lon_2d = np.broadcast_to(lon[:, np.newaxis], SP.shape)
pres_2d = np.broadcast_to(pres[np.newaxis, :], SP.shape)

# Convert practical salinity → absolute salinity (g/kg)
SA = gsw.SA_from_SP(SP, pres_2d, lon_2d, lat_2d)

# Convert in-situ temperature → conservative temperature (°C)
CT = gsw.CT_from_t(SA, T, pres_2d)

# In-situ density (kg/m³)
rho = gsw.rho(SA, CT, pres_2d)

# Potential density referenced to surface (sigma-0, kg/m³)
sigma0 = gsw.sigma0(SA, CT)

# Potential density referenced to 1000 dbar (sigma-1, kg/m³)
sigma1 = gsw.sigma1(SA, CT)

# ── 4. Build output dataset ────────────────────────────────────────────────────
ds_out = xr.Dataset(
    {
        "salinity": (["time", "pressure"], SP,
                     {"long_name": "Practical Salinity", "units": "PSU"}),
        "temperature": (["time", "pressure"], T,
                        {"long_name": "In-situ Temperature", "units": "degC"}),
        "absolute_salinity": (["time", "pressure"], SA,
                              {"long_name": "Absolute Salinity (TEOS-10)", "units": "g/kg"}),
        "conservative_temperature": (["time", "pressure"], CT,
                                     {"long_name": "Conservative Temperature (TEOS-10)", "units": "degC"}),
        "density": (["time", "pressure"], rho,
                    {"long_name": "In-situ Density", "units": "kg/m3"}),
        "sigma0": (["time", "pressure"], sigma0,
                   {"long_name": "Potential Density Anomaly (ref 0 dbar)", "units": "kg/m3"}),
        "sigma1": (["time", "pressure"], sigma1,
                   {"long_name": "Potential Density Anomaly (ref 1000 dbar)", "units": "kg/m3"}),
    },
    coords={
        "time":         ds["time"],
        "pressure":     ds["pressure"],
        "latitude":     ds["latitude"],
        "longitude":    ds["longitude"],
        "station":      ds["station"],
        "distance":     ds["distance"],
        "bottom_depth": ds["bottom_depth"],
    },
    attrs={
        "title":       "AR7W Labrador Sea – CTD density profiles (May–Aug)",
        "description": "Density computed from TEOS-10 gsw. Filtered to May–August only.",
        "source":      "igor_ar7w.nc",
        "gsw_version": gsw.__version__,
    },
)

# ── 5. Save ────────────────────────────────────────────────────────────────────
out_path = "~/efs-mount-point/mzahn/data/ar7w/ar7w_density_profiles.nc"
ds_out.to_netcdf(out_path)
print(f"\nSaved density profiles → {out_path}")
print(ds_out)
