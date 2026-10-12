
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


# ============================================================
# Zimbabwe Weather Intelligence
# Satellite Index Statistics
# Uses tifffile instead of rasterio
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SATELLITE_DIR = PROJECT_DIR / "data" / "satellite"
OUTPUT_DIR = PROJECT_DIR / "reports" / "satellite"

NDVI_FILE = SATELLITE_DIR / "harare_ndvi_20251019.tif"
NBR_FILE = SATELLITE_DIR / "harare_nbr_20251019.tif"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_AREA_M2 = 100.0  # 10 m x 10 m, confirmed from GeoTIFF tags


def read_geotiff(path):
    """Read a single-band GeoTIFF using tifffile."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        image = page.asarray().astype(np.float64)

        scale_tag = page.tags.get("ModelPixelScaleTag")
        nodata_tag = page.tags.get("GDAL_NODATA")

        pixel_size = None
        if scale_tag is not None:
            scale = scale_tag.value
            pixel_size = (float(scale[0]), float(scale[1]))

        nodata = None
        if nodata_tag is not None:
            try:
                nodata = float(nodata_tag.value)
            except (TypeError, ValueError):
                nodata = None

    valid = np.isfinite(image)

    # Exclude only the explicitly declared NoData value.
    # NaN is already excluded by np.isfinite().
    if nodata is not None and np.isfinite(nodata):
        valid &= image != nodata

    return image, valid, pixel_size, nodata


def summarize_index(name, image, valid):
    values = image[valid]

    if values.size == 0:
        raise ValueError(f"No valid pixels found for {name}.")

    return {
        "index": name,
        "valid_pixels": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


# Read images
ndvi, ndvi_valid, ndvi_pixel_size, ndvi_nodata = read_geotiff(NDVI_FILE)
nbr, nbr_valid, nbr_pixel_size, nbr_nodata = read_geotiff(NBR_FILE)

# Confirm raster alignment before comparing pixels
if ndvi.shape != nbr.shape:
    raise ValueError(
        f"Raster dimensions differ: NDVI {ndvi.shape}, NBR {nbr.shape}"
    )

if ndvi_pixel_size != nbr_pixel_size:
    raise ValueError(
        f"Pixel sizes differ: NDVI {ndvi_pixel_size}, NBR {nbr_pixel_size}"
    )

# Use only pixels valid in BOTH images for combined thresholds
combined_valid = ndvi_valid & nbr_valid

# Index statistics
summaries = [
    summarize_index("NDVI", ndvi, ndvi_valid),
    summarize_index("NBR", nbr, nbr_valid),
]

# Threshold indicators, restricted to valid pixels
vegetation_mask = ndvi_valid & (ndvi > 0.30)
low_nbr_mask = nbr_valid & (nbr < 0)
potential_candidate_mask = (
    combined_valid & (ndvi < 0.30) & (nbr < 0)
)

ndvi_valid_count = int(ndvi_valid.sum())
nbr_valid_count = int(nbr_valid.sum())
combined_valid_count = int(combined_valid.sum())

vegetation_pixels = int(vegetation_mask.sum())
low_nbr_pixels = int(low_nbr_mask.sum())
candidate_pixels = int(potential_candidate_mask.sum())


def area_km2(pixel_count):
    return pixel_count * PIXEL_AREA_M2 / 1_000_000


def percentage(count, total):
    return (count / total * 100) if total else 0.0


print("=" * 60)
print("HARARE SATELLITE INDEX STATISTICS")
print("=" * 60)

print("\nRaster information")
print("NDVI shape:", ndvi.shape)
print("NBR shape:", nbr.shape)
print("Pixel size (m):", ndvi_pixel_size)
print("Pixel area (m²):", PIXEL_AREA_M2)
print("NDVI NoData tag:", ndvi_nodata)
print("NBR NoData tag:", nbr_nodata)

print("\nIndex statistics")
for result in summaries:
    print(f"\n{result['index']}")
    print("Valid pixels:", result["valid_pixels"])
    print("Minimum:", round(result["minimum"], 6))
    print("Maximum:", round(result["maximum"], 6))
    print("Mean:", round(result["mean"], 6))
    print("Median:", round(result["median"], 6))

print("\nThreshold indicators")
print(
    f"NDVI > 0.30: {vegetation_pixels:,} pixels "
    f"({percentage(vegetation_pixels, ndvi_valid_count):.2f}% of valid NDVI pixels)"
)
print(
    f"NBR < 0: {low_nbr_pixels:,} pixels "
    f"({percentage(low_nbr_pixels, nbr_valid_count):.2f}% of valid NBR pixels)"
)
print(
    f"NDVI < 0.30 and NBR < 0: {candidate_pixels:,} pixels "
    f"({percentage(candidate_pixels, combined_valid_count):.2f}% of pixels valid in both images)"
)

print("\nApproximate threshold areas")
print(f"NDVI > 0.30: {area_km2(vegetation_pixels):.4f} km²")
print(f"NBR < 0: {area_km2(low_nbr_pixels):.4f} km²")
print(f"Combined candidate area: {area_km2(candidate_pixels):.4f} km²")

# Save machine-readable results
rows = []

for result in summaries:
    rows.append({
        "date": "2025-10-19",
        **result,
    })

rows.extend([
    {
        "date": "2025-10-19",
        "index": "NDVI > 0.30",
        "valid_pixels": ndvi_valid_count,
        "minimum": np.nan,
        "maximum": np.nan,
        "mean": np.nan,
        "median": np.nan,
        "threshold_pixels": vegetation_pixels,
        "percentage": percentage(vegetation_pixels, ndvi_valid_count),
        "area_km2": area_km2(vegetation_pixels),
    },
    {
        "date": "2025-10-19",
        "index": "NBR < 0",
        "valid_pixels": nbr_valid_count,
        "minimum": np.nan,
        "maximum": np.nan,
        "mean": np.nan,
        "median": np.nan,
        "threshold_pixels": low_nbr_pixels,
        "percentage": percentage(low_nbr_pixels, nbr_valid_count),
        "area_km2": area_km2(low_nbr_pixels),
    },
    {
        "date": "2025-10-19",
        "index": "NDVI < 0.30 and NBR < 0",
        "valid_pixels": combined_valid_count,
        "minimum": np.nan,
        "maximum": np.nan,
        "mean": np.nan,
        "median": np.nan,
        "threshold_pixels": candidate_pixels,
        "percentage": percentage(candidate_pixels, combined_valid_count),
        "area_km2": area_km2(candidate_pixels),
    },
])

output_file = OUTPUT_DIR / "harare_satellite_index_statistics_20251019.csv"
pd.DataFrame(rows).to_csv(output_file, index=False)

print("\nCSV report saved:", output_file)
print("\nStatistics completed successfully.")