import numpy as np
import xarray as xr
import gsw


# ── Helper: build fine pressure grid ─────────────────────────────────────────
def _build_p_grid(p_v, p_ref, max_dp_i):
    """
    Build a fine pressure grid at <= max_dp_i resolution,
    ensuring p=0 and p_ref are included.
    """
    # Start grid from 0 if the profile doesn't already begin there
    p_start = 0.0 if p_v[0] > 0 else p_v[0]
    all_p   = np.concatenate([[p_start], p_v])

    # Insert p_ref exactly into the breakpoints if it falls within the profile range
    if p_ref not in all_p and p_v.min() <= p_ref <= p_v.max():
        insert_idx = np.searchsorted(all_p, p_ref)
        all_p      = np.insert(all_p, insert_idx, p_ref)

    # Build each segment between consecutive bottle pressures at 1 dbar resolution
    segments = []
    for i in range(len(all_p) - 1):
        p_lo, p_hi = all_p[i], all_p[i + 1]
        n_steps    = int(np.ceil((p_hi - p_lo) / max_dp_i))
        seg        = np.linspace(p_lo, p_hi, n_steps + 1)
        segments.append(seg if i == 0 else seg[1:])  # trim duplicate endpoints

    return np.concatenate(segments)


# ── Dynamic height function ───────────────────────────────────────────────────
def geo_strf_dyn_height_from_rho(rho, p, p_ref):
    """
    Calculate dynamic height anomaly from in situ density profiles.
    Returns only the surface value (p=0), i.e. the full 0 -> p_ref integral.

    Parameters
    ----------
    rho   : np.ndarray - in situ density (kg/m3), shape (depth,)
    p     : np.ndarray - sea pressure (dbar), shape (depth,)
    p_ref : float      - reference pressure (dbar)

    Returns
    -------
    float - dynamic height anomaly at surface (m^2/s^2), or np.nan if profile
            doesn't reach p_ref
    """
    p_ref = float(p_ref)
    if p_ref < 0:
        raise ValueError("p_ref must be positive")

    rho_vals = np.asarray(rho, dtype=float)
    p_vals   = np.asarray(p,   dtype=float)

    db2Pa    = 1e4
    max_dp_i = 1.0

    # Drop NaN levels
    valid = ~np.isnan(rho_vals + p_vals)
    if valid.sum() < 2:
        return np.nan
    rho_v = rho_vals[valid]
    p_v   = p_vals[valid]

    # Profile must reach p_ref to compute a meaningful integral
    if p_v.max() < p_ref:
        return np.nan

    # Build 1 dbar fine grid from 0 to p_ref (anchored at both ends)
    p_fine = _build_p_grid(p_v, p_ref, max_dp_i)

    # Linearly interpolate density onto the fine grid
    rho_i = np.interp(p_fine, p_v, rho_v)

    # Specific volume anomaly: delta = 1/rho - 1/rho_ref(S=35, T=0)
    alpha_i     = 1.0 / rho_i
    s_ref_i     = np.full_like(p_fine, gsw.SR_from_SP(35), dtype=float)
    t_ref_i     = np.zeros_like(p_fine, dtype=float)
    alpha_ref_i = 1.0 / gsw.rho(s_ref_i, t_ref_i, p_fine)
    B_i         = alpha_i - alpha_ref_i

    # Trapezoidal integration downward from surface
    B_i_av       = 0.5 * (B_i[:-1] + B_i[1:])
    D_i          = B_i_av * np.diff(p_fine) * db2Pa
    dyn_h0_i     = np.zeros(len(p_fine))
    dyn_h0_i[1:] = -np.cumsum(D_i)

    # Subtract value at p_ref so the anomaly is referenced to 3500 dbar
    p_ref_idx_i = np.argmin(np.abs(p_fine - p_ref))
    dyn_h_i     = dyn_h0_i - dyn_h0_i[p_ref_idx_i]

    # Return the surface value (index 0 = p=0), which holds the full integral
    return dyn_h_i[0]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    input_path  = "~/efs-mount-point/mzahn/data/ar7w/ar7w_density_transects.nc"
    output_path = "~/efs-mount-point/mzahn/data/ar7w/ar7w_dynamic_height.nc"

    print("Loading dataset...")
    ds = xr.open_dataset(input_path)

    years     = ds.year.values
    distances = ds.distance_km.values
    pressures = ds.pressure.values
    lats      = ds.latitude.values
    lons      = ds.longitude.values
    p_ref     = 3500.0   # reference pressure in dbar

    n_years = len(years)
    n_dist  = len(distances)

    # ── Bottom-fill: extend the deepest valid density value downward ──────────
    # Ensures no trailing NaNs below the seafloor so profiles can reach p_ref
    print("Filling bottom-most valid density values downward...")
    density_filled = ds.density.values.copy()   # (year, distance_km, pressure)

    for iy in range(n_years):
        for id_ in range(n_dist):
            profile   = density_filled[iy, id_, :]
            valid_idx = np.where(~np.isnan(profile))[0]
            if len(valid_idx) == 0:
                continue
            last_valid = valid_idx[-1]
            profile[last_valid + 1:] = profile[last_valid]   # fill downward
            density_filled[iy, id_, :] = profile

    # ── Compute surface dynamic height at every (year, distance) point ────────
    # Output is 2D: (year, distance_km) — one scalar per vertical profile
    print(f"Computing dynamic height for {n_years} years x {n_dist} distances...")
    dyn_height = np.full((n_years, n_dist), np.nan)

    for iy, yr in enumerate(years):
        print(f"  Processing year {yr}...")
        for id_ in range(n_dist):
            rho_profile = density_filled[iy, id_, :]
            valid       = ~np.isnan(rho_profile)
            if valid.sum() < 2:
                continue

            # Skip shallow stations that don't reach the reference level
            if pressures[valid].max() < p_ref:
                continue

            try:
                result = geo_strf_dyn_height_from_rho(
                    rho_profile,
                    pressures.astype(float),
                    p_ref
                )
                # Divide by gravity (m/s^2) to convert m^2/s^2 -> metres
                dyn_height[iy, id_] = result / 9.7963
            except Exception as e:
                print(f"    Warning: year={yr}, dist_idx={id_} failed: {e}")
                continue

    # ── Build output DataArray and save ──────────────────────────────────────
    # Shape is now (year, distance_km) — no pressure dimension
    print("Building output DataArray...")
    da = xr.DataArray(
        dyn_height,
        coords={
            "year":        ("year",        years),
            "distance_km": ("distance_km", distances),
            "latitude":    ("distance_km", lats),
            "longitude":   ("distance_km", lons),
        },
        dims=["year", "distance_km"],
        attrs={
            "units":     "m",
            "long_name": "Dynamic Height Anomaly (0 to p_ref, surface value)",
            "p_ref":     f"{p_ref} dbar",
        }
    )

    ds_out = xr.Dataset({"dynamic_height": da})
    ds_out.to_netcdf(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
