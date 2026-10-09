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

output_mask = Path(
    "data/satellite/"
    "harare_high_dnbr_regions_20251012_20251019.tif"
)

output_csv = Path(
    "reports/satellite/"
    "harare_high_dnbr_regions_20251012_20251019.csv"
)


# ---------------------------------------------------------
# 2. Read dNBR
# ---------------------------------------------------------

with rasterio.open(input_file) as src:

    dnbr = src.read(1)

    profile = src.profile.copy()

    transform = src.transform


# ---------------------------------------------------------
# 3. Create high-change mask
# ---------------------------------------------------------
#
# High positive dNBR:
#
# dNBR >= 0.20
#
# These are candidate areas of substantial surface
# disturbance/change.
# ---------------------------------------------------------

mask = np.where(
    dnbr >= 0.20,
    1,
    0
).astype(np.uint8)


print("High-change pixels:", int(mask.sum()))


# ---------------------------------------------------------
# 4. Connected-component analysis
# ---------------------------------------------------------
#
# NumPy-only implementation.
#
# 8-connectivity means pixels touching horizontally,
# vertically or diagonally are grouped together.
# ---------------------------------------------------------

height, width = mask.shape

visited = np.zeros(
    (height, width),
    dtype=bool
)

labeled_array = np.zeros(
    (height, width),
    dtype=np.int32
)

region_id = 0

region_sizes = []

neighbors = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]


print(
    "Starting high-change region analysis..."
)


# ---------------------------------------------------------
# 5. Flood-fill
# ---------------------------------------------------------

for row in range(height):

    for col in range(width):

        if mask[row, col] == 0:
            continue

        if visited[row, col]:
            continue

        region_id += 1

        stack = [
            (row, col)
        ]

        visited[row, col] = True

        pixel_count = 0

        while stack:

            current_row, current_col = stack.pop()

            labeled_array[
                current_row,
                current_col
            ] = region_id

            pixel_count += 1

            for row_offset, col_offset in neighbors:

                neighbour_row = (
                    current_row +
                    row_offset
                )

                neighbour_col = (
                    current_col +
                    col_offset
                )

                if (
                    neighbour_row < 0
                    or neighbour_row >= height
                    or neighbour_col < 0
                    or neighbour_col >= width
                ):
                    continue

                if mask[
                    neighbour_row,
                    neighbour_col
                ] == 0:
                    continue

                if visited[
                    neighbour_row,
                    neighbour_col
                ]:
                    continue

                visited[
                    neighbour_row,
                    neighbour_col
                ] = True

                stack.append(
                    (
                        neighbour_row,
                        neighbour_col
                    )
                )

        region_sizes.append(
            pixel_count
        )


print(
    "Initial high-change regions:",
    region_id
)


# ---------------------------------------------------------
# 6. Minimum mapping unit
# ---------------------------------------------------------
#
# Keep regions >= 10 pixels.
#
# 10 pixels × 100 m² = 1,000 m²
# = 0.001 km²
# ---------------------------------------------------------

minimum_pixels = 10

region_sizes = np.array(
    region_sizes,
    dtype=np.int64
)

valid_regions = np.where(
    region_sizes >= minimum_pixels
)[0] + 1


print(
    "Regions >= 10 pixels:",
    len(valid_regions)
)


# ---------------------------------------------------------
# 7. Create filtered mask
# ---------------------------------------------------------

filtered_mask = np.isin(
    labeled_array,
    valid_regions
).astype(np.uint8)


# ---------------------------------------------------------
# 8. Save filtered raster
# ---------------------------------------------------------

profile.update(
    dtype="uint8",
    count=1,
    nodata=0
)

with rasterio.open(
    output_mask,
    "w",
    **profile
) as dst:

    dst.write(
        filtered_mask,
        1
    )


# ---------------------------------------------------------
# 9. Calculate region statistics
# ---------------------------------------------------------

records = []


for current_region_id in valid_regions:

    pixel_count = int(
        region_sizes[
            current_region_id - 1
        ]
    )

    area_m2 = (
        pixel_count *
        100
    )

    area_km2 = (
        area_m2 /
        1_000_000
    )

    rows, cols = np.where(
        labeled_array ==
        current_region_id
    )

    centroid_row = float(
        rows.mean()
    )

    centroid_col = float(
        cols.mean()
    )

    x, y = rasterio.transform.xy(
        transform,
        centroid_row,
        centroid_col
    )

    records.append({

        "region_id":
            int(current_region_id),

        "pixels":
            pixel_count,

        "area_m2":
            area_m2,

        "area_km2":
            area_km2,

        "centroid_x":
            float(x),

        "centroid_y":
            float(y)
    })


# ---------------------------------------------------------
# 10. DataFrame
# ---------------------------------------------------------

regions_df = pd.DataFrame(
    records
)


if not regions_df.empty:

    regions_df = regions_df.sort_values(
        "area_km2",
        ascending=False
    )

    regions_df["rank"] = range(
        1,
        len(regions_df) + 1
    )


# ---------------------------------------------------------
# 11. Save CSV
# ---------------------------------------------------------

regions_df.to_csv(
    output_csv,
    index=False
)


# ---------------------------------------------------------
# 12. Summary
# ---------------------------------------------------------

total_pixels = int(
    filtered_mask.sum()
)

total_area_km2 = (
    total_pixels *
    100 /
    1_000_000
)


print()
print("==========================================")
print("HIGH dNBR REGION ANALYSIS")
print("==========================================")

print(
    "Filtered regions:",
    len(valid_regions)
)

print(
    "Pixels in filtered regions:",
    total_pixels
)

print(
    "Total high-change area:",
    round(
        total_area_km2,
        4
    ),
    "km²"
)


if not regions_df.empty:

    print(
        "Largest region:",
        round(
            regions_df.iloc[0]["area_km2"],
            4
        ),
        "km²"
    )

    print()
    print("Top 10 regions:")

    print(
        regions_df[
            [
                "rank",
                "pixels",
                "area_km2",
                "centroid_x",
                "centroid_y"
            ]
        ].head(10).to_string(
            index=False
        )
    )


print()
print(
    "Region raster saved:",
    output_mask
)

print(
    "Region statistics saved:",
    output_csv
)