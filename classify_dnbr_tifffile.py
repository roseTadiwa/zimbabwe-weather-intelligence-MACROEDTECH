
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

# Input and output files
input_raster = Path(
    "data/satellite/harare_dnbr_20251012_20251019.tif"
)

output_raster = Path(
    "data/satellite/harare_dnbr_classes_20251012_20251019.tif"
)

output_csv = Path(
    "reports/satellite/harare_dnbr_class_statistics_20251012_20251019.csv"
)

output_csv.parent.mkdir(parents=True, exist_ok=True)

# Read the existing dNBR raster
with tifffile.TiffFile(input_raster) as tif:
    page = tif.pages[0]
    dnbr = page.asarray().astype(np.float32)
    tags = page.tags

print("Raster shape:", dnbr.shape)

# Classify only valid, finite pixels
valid = np.isfinite(dnbr)
classes = np.zeros(dnbr.shape, dtype=np.uint8)

classes[valid & (dnbr >= 0.20)] = 1
classes[valid & (dnbr >= 0.10) & (dnbr < 0.20)] = 2
classes[valid & (dnbr >= -0.10) & (dnbr < 0.10)] = 3
classes[valid & (dnbr < -0.10)] = 4

# Calculate class statistics
class_information = {
    1: "High positive change (dNBR >= 0.20)",
    2: "Moderate positive change (0.10 <= dNBR < 0.20)",
    3: "Little or no change (-0.10 <= dNBR < 0.10)",
    4: "Negative change (dNBR < -0.10)",
}

pixel_area_m2 = 100
total_valid = int(valid.sum())
records = []

for class_id, description in class_information.items():
    count = int(np.sum(classes == class_id))
    area_km2 = count * pixel_area_m2 / 1_000_000

    records.append({
        "class_id": class_id,
        "description": description,
        "pixels": count,
        "percentage_of_valid_pixels":
            count / total_valid * 100 if total_valid else 0,
        "area_km2": area_km2,
    })

# Save classification while preserving available TIFF metadata
# tifffile does not automatically preserve georeferencing when
# writing a new TIFF, so copy the original geospatial tags.
extratags = []
for tag_id in (33550, 33922, 34735, 34736, 34737, 42113):
    tag = tags.get(tag_id)
    if tag is not None:
        extratags.append(
            (tag.code, tag.dtype, tag.count, tag.value, False)
        )

tifffile.imwrite(
    output_raster,
    classes,
    photometric="minisblack",
    metadata=None,
    extratags=extratags,
)

results = pd.DataFrame(records)
results.to_csv(output_csv, index=False)

print("\nD N B R CLASSIFICATION")
print("======================")
print("Valid pixels:", f"{total_valid:,}")

for record in records:
    print(f"\n{record['description']}")
    print("Pixels:", f"{record['pixels']:,}")
    print("Percentage:", f"{record['percentage_of_valid_pixels']:.2f}%")
    print("Area (km²):", f"{record['area_km2']:.4f}")

print("\nClassification saved:", output_raster)
print("Statistics saved:", output_csv)
