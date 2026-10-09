import rasterio
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output files
# ---------------------------------------------------------

ndvi_file = (
    "data/satellite/"
    "harare_ndvi_20251019.tif"
)

nbr_file = (
    "data/satellite/"
    "harare_nbr_20251019.tif"
)

output_file = Path(
    "data/satellite/"
    "harare_potential_burn_mask_20251019.tif"
)


# ---------------------------------------------------------
# 2. Read NDVI
# ---------------------------------------------------------

ndvi_src = rasterio.open(ndvi_file)

ndvi = ndvi_src.read(1)
profile = ndvi_src.profile.copy()

ndvi_src.close()


# ---------------------------------------------------------
# 3. Read corrected NBR
# ---------------------------------------------------------

nbr_src = rasterio.open(nbr_file)

nbr = nbr_src.read(1)

nbr_src.close()


# ---------------------------------------------------------
# 4. Check that both rasters have the same dimensions
# ---------------------------------------------------------

if ndvi.shape != nbr.shape:
    raise ValueError(
        f"NDVI and NBR dimensions do not match: "
        f"{ndvi.shape} vs {nbr.shape}"
    )


# ---------------------------------------------------------
# 5. Create potential burn/disturbance candidates
#
# Condition 1:
# NBR < 0
#
# Condition 2:
# NDVI < 0.30
#
# Both conditions must be true.
#
# 1 = potential burn/disturbance candidate
# 0 = other areas
# ---------------------------------------------------------

potential_burn = np.where(
    (nbr < 0) & (ndvi < 0.30),
    1,
    0
).astype(np.uint8)


# ---------------------------------------------------------
# 6. Save the mask
# ---------------------------------------------------------

profile.update(
    dtype="uint8",
    count=1,
    nodata=0
)

with rasterio.open(
    output_file,
    "w",
    **profile
) as dst:
    dst.write(potential_burn, 1)


# ---------------------------------------------------------
# 7. Calculate statistics
# ---------------------------------------------------------

candidate_pixels = int(
    potential_burn.sum()
)

total_pixels = potential_burn.size

candidate_percentage = (
    candidate_pixels / total_pixels
) * 100


# Each pixel is 10 m × 10 m
pixel_area_m2 = 100

candidate_area_m2 = (
    candidate_pixels * pixel_area_m2
)

candidate_area_km2 = (
    candidate_area_m2 / 1_000_000
)


# ---------------------------------------------------------
# 8. Print results
# ---------------------------------------------------------

print(
    "Potential burn/disturbance mask saved:",
    output_file
)

print(
    "Potential candidate pixels:",
    candidate_pixels
)

print(
    "Total pixels:",
    total_pixels
)

print(
    "Potential candidate percentage:",
    round(candidate_percentage, 2)
)

print(
    "Potential candidate area:",
    round(candidate_area_m2, 2),
    "m²"
)

print(
    "Potential candidate area:",
    round(candidate_area_km2, 4),
    "km²"
)