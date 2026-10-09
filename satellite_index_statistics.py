import rasterio
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. Input files
# ---------------------------------------------------------

ndvi_file = (
    "data/satellite/"
    "harare_ndvi_20251019.tif"
)

nbr_file = (
    "data/satellite/"
    "harare_nbr_20251019.tif"
)


# ---------------------------------------------------------
# 2. Read NDVI
# ---------------------------------------------------------

ndvi_src = rasterio.open(ndvi_file)

ndvi = ndvi_src.read(1).astype(np.float32)

ndvi_src.close()


# ---------------------------------------------------------
# 3. Read NBR
# ---------------------------------------------------------

nbr_src = rasterio.open(nbr_file)

nbr = nbr_src.read(1).astype(np.float32)

nbr_src.close()


# ---------------------------------------------------------
# 4. Remove invalid values
# ---------------------------------------------------------

valid_ndvi = ndvi[np.isfinite(ndvi)]
valid_nbr = nbr[np.isfinite(nbr)]


# ---------------------------------------------------------
# 5. Calculate NDVI statistics
# ---------------------------------------------------------

ndvi_min = float(np.min(valid_ndvi))
ndvi_max = float(np.max(valid_ndvi))
ndvi_mean = float(np.mean(valid_ndvi))
ndvi_median = float(np.median(valid_ndvi))


# ---------------------------------------------------------
# 6. Calculate NBR statistics
# ---------------------------------------------------------

nbr_min = float(np.min(valid_nbr))
nbr_max = float(np.max(valid_nbr))
nbr_mean = float(np.mean(valid_nbr))
nbr_median = float(np.median(valid_nbr))


# ---------------------------------------------------------
# 7. Classification statistics
# ---------------------------------------------------------

total_pixels = len(valid_ndvi)

# Active/greener vegetation
ndvi_vegetation_pixels = int(
    np.sum(ndvi > 0.30)
)

# Low NBR
nbr_low_pixels = int(
    np.sum(nbr < 0)
)

# Combined candidate condition
potential_candidate_pixels = int(
    np.sum(
        (ndvi < 0.30) &
        (nbr < 0)
    )
)


# ---------------------------------------------------------
# 8. Calculate percentages
# ---------------------------------------------------------

ndvi_vegetation_percentage = (
    ndvi_vegetation_pixels /
    total_pixels
) * 100

nbr_low_percentage = (
    nbr_low_pixels /
    total_pixels
) * 100

potential_candidate_percentage = (
    potential_candidate_pixels /
    total_pixels
) * 100


# ---------------------------------------------------------
# 9. Calculate areas
#
# Each pixel = 10 m × 10 m = 100 m²
# ---------------------------------------------------------

pixel_area_m2 = 100

ndvi_vegetation_area_km2 = (
    ndvi_vegetation_pixels *
    pixel_area_m2 /
    1_000_000
)

nbr_low_area_km2 = (
    nbr_low_pixels *
    pixel_area_m2 /
    1_000_000
)

potential_candidate_area_km2 = (
    potential_candidate_pixels *
    pixel_area_m2 /
    1_000_000
)


# ---------------------------------------------------------
# 10. Create results table
# ---------------------------------------------------------

results = pd.DataFrame({
    "Metric": [
        "NDVI minimum",
        "NDVI maximum",
        "NDVI mean",
        "NDVI median",
        "NBR minimum",
        "NBR maximum",
        "NBR mean",
        "NBR median",
        "NDVI > 0.30 pixels",
        "NDVI > 0.30 percentage",
        "NDVI > 0.30 area (km²)",
        "NBR < 0 pixels",
        "NBR < 0 percentage",
        "NBR < 0 area (km²)",
        "Potential candidate pixels",
        "Potential candidate percentage",
        "Potential candidate area (km²)"
    ],
    "Value": [
        ndvi_min,
        ndvi_max,
        ndvi_mean,
        ndvi_median,
        nbr_min,
        nbr_max,
        nbr_mean,
        nbr_median,
        ndvi_vegetation_pixels,
        ndvi_vegetation_percentage,
        ndvi_vegetation_area_km2,
        nbr_low_pixels,
        nbr_low_percentage,
        nbr_low_area_km2,
        potential_candidate_pixels,
        potential_candidate_percentage,
        potential_candidate_area_km2
    ]
})


# ---------------------------------------------------------
# 11. Save CSV report
# ---------------------------------------------------------

output_file = Path(
    "reports/satellite/"
    "harare_satellite_index_statistics_20251019.csv"
)

results.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 12. Print report
# ---------------------------------------------------------

print("\n==========================================")
print("HARARE SATELLITE INDEX STATISTICS")
print("19 OCTOBER 2025")
print("==========================================\n")

print(
    f"NDVI minimum:              {ndvi_min:.4f}"
)

print(
    f"NDVI maximum:              {ndvi_max:.4f}"
)

print(
    f"NDVI mean:                 {ndvi_mean:.4f}"
)

print(
    f"NDVI median:               {ndvi_median:.4f}"
)

print()

print(
    f"NBR minimum:               {nbr_min:.4f}"
)

print(
    f"NBR maximum:               {nbr_max:.4f}"
)

print(
    f"NBR mean:                  {nbr_mean:.4f}"
)

print(
    f"NBR median:                {nbr_median:.4f}"
)

print()

print(
    f"NDVI > 0.30 pixels:        "
    f"{ndvi_vegetation_pixels:,}"
)

print(
    f"NDVI > 0.30 percentage:    "
    f"{ndvi_vegetation_percentage:.2f}%"
)

print(
    f"NDVI > 0.30 area:          "
    f"{ndvi_vegetation_area_km2:.4f} km²"
)

print()

print(
    f"NBR < 0 pixels:            "
    f"{nbr_low_pixels:,}"
)

print(
    f"NBR < 0 percentage:        "
    f"{nbr_low_percentage:.2f}%"
)

print(
    f"NBR < 0 area:              "
    f"{nbr_low_area_km2:.4f} km²"
)

print()

print(
    f"Potential candidate pixels:"
    f" {potential_candidate_pixels:,}"
)

print(
    f"Potential candidate percentage:"
    f" {potential_candidate_percentage:.2f}%"
)

print(
    f"Potential candidate area:  "
    f"{potential_candidate_area_km2:.4f} km²"
)

print()

print(
    "CSV report saved:",
    output_file
)