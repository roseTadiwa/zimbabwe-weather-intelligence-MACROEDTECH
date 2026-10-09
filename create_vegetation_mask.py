import rasterio
import numpy as np
from pathlib import Path

input_file = "data/satellite/harare_ndvi_20251019.tif"
output_file = Path("data/satellite/harare_vegetation_mask_20251019.tif")

src = rasterio.open(input_file)

ndvi = src.read(1)
profile = src.profile.copy()

src.close()

# Classify pixels
# 1 = vegetation (NDVI > 0.30)
# 0 = non-vegetation / sparse vegetation
mask = (ndvi > 0.30).astype(np.uint8)

profile.update(
    dtype="uint8",
    count=1,
    nodata=0
)

with rasterio.open(output_file, "w", **profile) as dst:
    dst.write(mask, 1)

print("Vegetation mask saved:", output_file)
print("Vegetation pixels:", int(mask.sum()))
print("Total pixels:", mask.size)
print("Vegetation percentage:", round(mask.mean() * 100, 2))