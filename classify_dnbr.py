import rasterio
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output
# ---------------------------------------------------------

input_file = (
    "data/satellite/"
    "harare_dnbr_20251012_20251019.tif"
)

output_raster = Path(
    "data/satellite/"
    "harare_dnbr_classes_20251012_20251019.tif"
)

output_csv = Path(
    "reports/satellite/"
    "harare_dnbr_class_statistics_20251012_20251019.csv"
)


# ---------------------------------------------------------
# 2. Read dNBR
# ---------------------------------------------------------

with rasterio.open(input_file) as src:

    dnbr = src.read(1)

    profile = src.profile.copy()


# ---------------------------------------------------------
# 3. Create classification
# ---------------------------------------------------------
#
# Class 1 = High positive change
#           dNBR >= 0.20
#
# Class 2 = Moderate positive change
#           0.10 <= dNBR < 0.20
#
# Class 3 = Little / no change
#           -0.10 <= dNBR < 0.10
#
# Class 4 = Negative change
#           dNBR < -0.10
# ---------------------------------------------------------

classes = np.zeros(
    dnbr.shape,
    dtype=np.uint8
)

classes[
    dnbr >= 0.20
] = 1

classes[
    (dnbr >= 0.10) &
    (dnbr < 0.20)
] = 2

classes[
    (dnbr >= -0.10) &
    (dnbr < 0.10)
] = 3

classes[
    dnbr < -0.10
] = 4


# ---------------------------------------------------------
# 4. Save classified raster
# ---------------------------------------------------------

profile.update(
    dtype="uint8",
    count=1,
    nodata=0
)

with rasterio.open(
    output_raster,
    "w",
    **profile
) as dst:

    dst.write(
        classes,
        1
    )


# ---------------------------------------------------------
# 5. Calculate statistics
# ---------------------------------------------------------

pixel_area_m2 = 100

total_pixels = dnbr.size

records = []

class_information = {

    1: "High positive change (dNBR >= 0.20)",

    2: "Moderate positive change (0.10 <= dNBR < 0.20)",

    3: "Little or no change (-0.10 <= dNBR < 0.10)",

    4: "Negative change (dNBR < -0.10)"
}


for class_id, description in class_information.items():

    pixel_count = int(
        np.sum(classes == class_id)
    )

    percentage = (
        pixel_count /
        total_pixels
    ) * 100

    area_m2 = (
        pixel_count *
        pixel_area_m2
    )

    area_km2 = (
        area_m2 /
        1_000_000
    )

    records.append({

        "class_id":
            class_id,

        "description":
            description,

        "pixels":
            pixel_count,

        "percentage":
            percentage,

        "area_km2":
            area_km2
    })


# ---------------------------------------------------------
# 6. Save statistics
# ---------------------------------------------------------

results = pd.DataFrame(
    records
)

results.to_csv(
    output_csv,
    index=False
)


# ---------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------

print()
print("==========================================")
print("dNBR CHANGE CLASSIFICATION")
print("==========================================")

print()

for record in records:

    print(
        record["description"]
    )

    print(
        "Pixels:",
        f'{record["pixels"]:,}'
    )

    print(
        "Percentage:",
        f'{record["percentage"]:.2f}%'
    )

    print(
        "Area:",
        f'{record["area_km2"]:.4f} km²'
    )

    print()


print(
    "Classified raster saved:",
    output_raster
)

print(
    "Statistics saved:",
    output_csv
)