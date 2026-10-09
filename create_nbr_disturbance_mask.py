import rasterio
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output files
# ---------------------------------------------------------

input_file = "data/satellite/harare_nbr_20251019.tif"

output_file = Path(
    "data/satellite/harare_nbr_disturbance_mask_20251019.tif"
)


# ---------------------------------------------------------
# 2. Read the corrected NBR raster
# ---------------------------------------------------------

src = rasterio.open(input_file)

nbr = src.read(1)
profile = src.profile.copy()

src.close()


# ---------------------------------------------------------
# 3. Create potential disturbance mask
#
# 1 = Potential low-NBR disturbance
# 0 = Other areas
#
# Threshold:
# NBR < 0
# ---------------------------------------------------------

mask = np.where(
    nbr < 0,
    1,
    0
).astype(np.uint8)


# ---------------------------------------------------------
# 4. Save the mask
# ---------------------------------------------------------

profile.update(
    dtype="uint8",
    count=1,
    nodata=0
)

with rasterio.open(output_file, "w", **profile) as dst:
    dst.write(mask, 1)


# ---------------------------------------------------------
# 5. Calculate statistics
# ---------------------------------------------------------

potential_pixels = int(mask.sum())
total_pixels = mask.size

potential_percentage = (
    potential_pixels / total_pixels
) * 100

# Each pixel is 10 m x 10 m = 100 m²
pixel_area_m2 = 10 * 10

potential_area_m2 = (
    potential_pixels * pixel_area_m2
)

potential_area_km2 = (
    potential_area_m2 / 1_000_000
)


# ---------------------------------------------------------
# 6. Print results
# ---------------------------------------------------------

print(
    "Potential disturbance mask saved:",
    output_file
)

print(
    "Potential disturbance pixels:",
    potential_pixels
)

print(
    "Total pixels:",
    total_pixels
)

print(
    "Potential disturbance percentage:",
    round(potential_percentage, 2)
)

print(
    "Potential disturbance area:",
    round(potential_area_m2, 2),
    "m²"
)

print(
    "Potential disturbance area:",
    round(potential_area_km2, 4),
    "km²"
)