import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from pathlib import Path
import pystac_client
import planetary_computer


# ---------------------------------------------------------
# 1. Connect to Microsoft Planetary Computer
# ---------------------------------------------------------

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    ids=["S2A_MSIL2A_20251019T075021_R135_T36KTF_20251019T111413"]
)

item = list(search.items())[0]

b08_url = item.assets["B08"].href
b12_url = item.assets["B12"].href


# ---------------------------------------------------------
# 2. Read B08 (NIR) - 10 m resolution
# ---------------------------------------------------------

nir_src = rasterio.open(b08_url)

nir_window = rasterio.windows.Window(
    0,
    0,
    1000,
    1000
)

nir = nir_src.read(
    1,
    window=nir_window
).astype(np.float32)

# Get geographic bounds of the B08 window
left, bottom, right, top = rasterio.windows.bounds(
    nir_window,
    nir_src.transform
)

# Get transform for the output raster
output_transform = nir_src.window_transform(nir_window)

# Copy raster metadata
profile = nir_src.profile.copy()

nir_src.close()


# ---------------------------------------------------------
# 3. Read B12 (SWIR) from the SAME geographic area
# ---------------------------------------------------------

swir_src = rasterio.open(b12_url)

# Find the B12 window corresponding to the B08 area
swir_window = from_bounds(
    left,
    bottom,
    right,
    top,
    transform=swir_src.transform
)

# B12 is 20 m resolution.
# Resample it to the same 1000 x 1000 grid as B08.
swir = swir_src.read(
    1,
    window=swir_window,
    out_shape=(1000, 1000),
    resampling=Resampling.bilinear
).astype(np.float32)

swir_src.close()


# ---------------------------------------------------------
# 4. Calculate NBR
#
# NBR = (NIR - SWIR) / (NIR + SWIR)
# ---------------------------------------------------------

denominator = nir + swir

nbr = np.where(
    denominator != 0,
    (nir - swir) / denominator,
    np.nan
).astype(np.float32)


# ---------------------------------------------------------
# 5. Save corrected NBR raster
# ---------------------------------------------------------

output = Path(
    "data/satellite/harare_nbr_20251019.tif"
)

profile.update(
    height=1000,
    width=1000,
    count=1,
    dtype="float32",
    transform=output_transform,
    nodata=np.nan
)

with rasterio.open(output, "w", **profile) as dst:
    dst.write(nbr, 1)


# ---------------------------------------------------------
# 6. Print NBR statistics
# ---------------------------------------------------------

print("Corrected NBR saved:", output)
print("Shape:", nbr.shape)
print("Minimum:", float(np.nanmin(nbr)))
print("Maximum:", float(np.nanmax(nbr)))
print("Mean:", float(np.nanmean(nbr)))