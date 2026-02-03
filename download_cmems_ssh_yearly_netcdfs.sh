#!/usr/bin/env bash
set -euo pipefail

# ==========================
# User settings
# ==========================

DATASET_ID="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"

# Spatial subset (degrees)
MIN_LON=-60
MAX_LON=-40
MIN_LAT=50
MAX_LAT=65

# Time range
START_YEAR=1993
END_YEAR=2025

# Output directory
OUTDIR="/Users/mzahn/data/SSH/cmems_ssh_yearly"

# ==========================
# Setup
# ==========================

mkdir -p "${OUTDIR}"

echo "Starting CMEMS SSH yearly downloads..."
echo "Years: ${START_YEAR} to ${END_YEAR}"
echo "Region: ${MIN_LAT}-${MAX_LAT}N, ${MIN_LON}-${MAX_LON}E"

# ==========================
# Download loop
# ==========================

for YEAR in $(seq ${START_YEAR} ${END_YEAR}); do

  echo "----------------------------------------"
  echo "Downloading year ${YEAR}..."

  copernicusmarine subset \
    --dataset-id "${DATASET_ID}" \
    --minimum-longitude "${MIN_LON}" \
    --maximum-longitude "${MAX_LON}" \
    --minimum-latitude "${MIN_LAT}" \
    --maximum-latitude "${MAX_LAT}" \
    --start-datetime "${YEAR}-01-01T00:00:00" \
    --end-datetime   "${YEAR}-12-31T23:59:59" \
    --output-directory "${OUTDIR}" \
    --output-filename "ssh_cmems_l4_0pt125deg_50N_65N_60W_40W_${YEAR}.nc"

  echo "Finished ${YEAR}"
done

echo "All downloads completed successfully."
