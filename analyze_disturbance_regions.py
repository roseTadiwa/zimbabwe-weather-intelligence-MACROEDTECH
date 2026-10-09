import rasterio
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output files
# ---------------------------------------------------------

input_file = (
    "data/satellite/"
    "harare_potential_burn_mask_20251019.tif"
)

output_mask = Path(
    "data/satellite/"
    "harare_disturbance_regions_20251019.tif"
)

output_csv = Path(
    "reports/satellite/"
    "harare_disturbance_regions_20251019.csv"
)


# ---------------------------------------------------------
# 2. Read the potential disturbance mask
# ---------------------------------------------------------

with rasterio.open(input_file) as src:

    mask = src.read(1)

    profile = src.profile.copy()

    transform = src.transform


# Convert to binary
mask = (mask > 0).astype(np.uint8)


print("Input mask loaded.")
print("Image shape:", mask.shape)
print("Candidate pixels:", int(mask.sum()))


# ---------------------------------------------------------
# 3. NumPy connected-component analysis
# ---------------------------------------------------------
#
# 8-connectivity:
# Pixels connected horizontally, vertically, or
# diagonally are treated as belonging to the same region.
#
# This replaces scipy.ndimage.label().
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


# 8 neighbouring directions
neighbors = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]


print()
print("Starting connected-component analysis...")


# ---------------------------------------------------------
# 4. Flood-fill each connected region
# ---------------------------------------------------------

for row in range(height):

    for col in range(width):

        # Skip background
        if mask[row, col] == 0:
            continue

        # Skip already processed pixels
        if visited[row, col]:
            continue

        region_id += 1

        stack = [(row, col)]

        visited[row, col] = True

        pixel_count = 0

        while stack:

            current_row, current_col = stack.pop()

            labeled_array[
                current_row,
                current_col
            ] = region_id

            pixel_count += 1

            # Check all 8 neighbours
            for row_offset, col_offset in neighbors:

                neighbour_row = current_row + row_offset
                neighbour_col = current_col + col_offset

                # Check image boundaries
                if (
                    neighbour_row < 0
                    or neighbour_row >= height
                    or neighbour_col < 0
                    or neighbour_col >= width
                ):
                    continue

                # Skip background
                if mask[
                    neighbour_row,
                    neighbour_col
                ] == 0:
                    continue

                # Skip visited pixels
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

        region_sizes.append(pixel_count)


print(
    "Initial connected regions:",
    region_id
)


# ---------------------------------------------------------
# 5. Convert region sizes to NumPy array
# ---------------------------------------------------------

region_sizes = np.array(
    region_sizes,
    dtype=np.int64
)


# ---------------------------------------------------------
# 6. Filter very small regions
# ---------------------------------------------------------
#
# Minimum mapping unit:
#
# 100 pixels × 10 m × 10 m
# = 10,000 m²
# = 0.01 km²
# ---------------------------------------------------------

minimum_pixels = 100

valid_regions = np.where(
    region_sizes >= minimum_pixels
)[0] + 1


print(
    "Regions >= 100 pixels:",
    len(valid_regions)
)


# ---------------------------------------------------------
# 7. Create filtered region mask
# ---------------------------------------------------------

filtered_mask = np.isin(
    labeled_array,
    valid_regions
).astype(np.uint8)


# ---------------------------------------------------------
# 8. Save filtered mask
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
# 9. Calculate statistics for each region
# ---------------------------------------------------------

records = []


for current_region_id in valid_regions:

    pixel_count = int(
        region_sizes[
            current_region_id - 1
        ]
    )

    area_m2 = pixel_count * 100

    area_km2 = area_m2 / 1_000_000

    rows, cols = np.where(
        labeled_array == current_region_id
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
# 10. Create DataFrame
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
# 12. Summary statistics
# ---------------------------------------------------------

total_region_pixels = int(
    filtered_mask.sum()
)

total_region_area_km2 = (
    total_region_pixels
    * 100
    / 1_000_000
)


print()
print("==========================================")
print("DISTURBANCE REGION ANALYSIS")
print("==========================================")

print(
    "Filtered regions:",
    len(valid_regions)
)

print(
    "Pixels in filtered regions:",
    total_region_pixels
)

print(
    "Total filtered area:",
    round(
        total_region_area_km2,
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