
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import tifffile

# --------------------------------------------------
# 1. Input and output paths
# --------------------------------------------------

input_file = Path(
    "data/satellite/harare_dnbr_20251012_20251019.tif"
)

output_mask = Path(
    "data/satellite/harare_high_dnbr_regions_20251012_20251019.tif"
)

output_csv = Path(
    "reports/satellite/harare_high_dnbr_regions_20251012_20251019.csv"
)

output_csv.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Read raster and preserve geospatial tag values
# --------------------------------------------------

with tifffile.TiffFile(input_file) as tif:
    page = tif.pages[0]
    dnbr = page.asarray().astype(np.float32)

    geo_tags = []
    for tag_id in (33550, 33922, 34735, 34736, 34737, 42113):
        tag = page.tags.get(tag_id)
        if tag is not None:
            geo_tags.append(
                (tag.code, tag.dtype, tag.count, tag.value, False)
            )

valid = np.isfinite(dnbr)
high_mask = valid & (dnbr >= 0.20)

height, width = dnbr.shape
visited = np.zeros((height, width), dtype=bool)
labels = np.zeros((height, width), dtype=np.int32)

# Eight-connected neighbours, including diagonals
neighbors = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1)
]

minimum_pixels = 10
pixel_area_m2 = 100
region_records = []
region_id = 0

print("High-change pixels:", int(high_mask.sum()))
print("Starting NumPy connected-region analysis...")

# --------------------------------------------------
# 3. Identify connected regions using flood-fill
# --------------------------------------------------

for start_row in range(height):
    for start_col in range(width):

        if not high_mask[start_row, start_col]:
            continue

        if visited[start_row, start_col]:
            continue

        region_id += 1
        queue = deque([(start_row, start_col)])
        visited[start_row, start_col] = True

        pixels = []

        while queue:
            row, col = queue.pop()
            pixels.append((row, col))

            for dr, dc in neighbors:
                nr = row + dr
                nc = col + dc

                if not (0 <= nr < height and 0 <= nc < width):
                    continue

                if not high_mask[nr, nc] or visited[nr, nc]:
                    continue

                visited[nr, nc] = True
                queue.append((nr, nc))

        pixel_count = len(pixels)

        if pixel_count < minimum_pixels:
            continue

        rows = np.fromiter(
            (p[0] for p in pixels), dtype=np.int32
        )
        cols = np.fromiter(
            (p[1] for p in pixels), dtype=np.int32
        )

        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())

        region_records.append({
            "region_id": region_id,
            "pixels": pixel_count,
            "area_m2": pixel_count * pixel_area_m2,
            "area_km2": pixel_count * pixel_area_m2 / 1_000_000,
            "centroid_row": centroid_row,
            "centroid_col": centroid_col,
        })

# --------------------------------------------------
# 4. Create filtered binary mask
# --------------------------------------------------

filtered_mask = np.zeros((height, width), dtype=np.uint8)

# Rebuild the mask from retained region labels using the
# high-change mask and the recorded retained region IDs.
retained_ids = {r["region_id"] for r in region_records}

# The labels array was not populated during flood-fill above,
# so use a second flood-fill to mark only retained regions.
visited[:] = False
current_id = 0

for start_row in range(height):
    for start_col in range(width):

        if not high_mask[start_row, start_col]:
            continue
        if visited[start_row, start_col]:
            continue

        current_id += 1
        queue = deque([(start_row, start_col)])
        visited[start_row, start_col] = True
        pixels = []

        while queue:
            row, col = queue.pop()
            pixels.append((row, col))

            for dr, dc in neighbors:
                nr = row + dr
                nc = col + dc

                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if not high_mask[nr, nc] or visited[nr, nc]:
                    continue

                visited[nr, nc] = True
                queue.append((nr, nc))

        if len(pixels) >= minimum_pixels:
            for row, col in pixels:
                filtered_mask[row, col] = 1

# --------------------------------------------------
# 5. Convert centroids to approximate map coordinates
# --------------------------------------------------

with tifffile.TiffFile(input_file) as tif:
    tags = tif.pages[0].tags
    scale_tag = tags.get(33550)
    tie_tag = tags.get(33922)

    scale = scale_tag.value if scale_tag is not None else None
    tie = tie_tag.value if tie_tag is not None else None

for record in region_records:
    if scale is not None and tie is not None:
        raster_i, raster_j, _, map_x, map_y, _ = tie[:6]
        record["centroid_x"] = (
            map_x + (record["centroid_col"] - raster_i) * scale[0]
        )
        record["centroid_y"] = (
            map_y - (record["centroid_row"] - raster_j) * scale[1]
        )
    else:
        record["centroid_x"] = np.nan
        record["centroid_y"] = np.nan

# --------------------------------------------------
# 6. Save outputs
# --------------------------------------------------

tifffile.imwrite(
    output_mask,
    filtered_mask,
    photometric="minisblack",
    metadata=None,
    extratags=geo_tags,
)

df = pd.DataFrame(region_records)

if not df.empty:
    df = df.sort_values("area_km2", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))

df.to_csv(output_csv, index=False)

# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

retained_pixels = int(filtered_mask.sum())

print("\nHIGH dNBR REGION ANALYSIS")
print("=========================")
print("Initial high-change regions:", region_id)
print("Retained regions:", len(region_records))
print("Pixels in retained regions:", retained_pixels)
print(
    "Retained area (km²):",
    round(retained_pixels * pixel_area_m2 / 1_000_000, 4)
)

if not df.empty:
    print("\nTop 10 regions:")
    print(
        df[[
            "rank", "pixels", "area_km2",
            "centroid_x", "centroid_y"
        ]].head(10).to_string(index=False)
    )

print("\nRegion raster saved:", output_mask)
print("Region statistics saved:", output_csv)
