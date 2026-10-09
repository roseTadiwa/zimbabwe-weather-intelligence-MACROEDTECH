import rasterio
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output
# ---------------------------------------------------------

input_file = (
    "data/satellite/"
    "harare_high_dnbr_regions_20251012_20251019.tif"
)

output_file = Path(
    "reports/satellite/"
    "harare_high_dnbr_regions_map_20251012_20251019.png"
)


# ---------------------------------------------------------
# 2. Read region mask
# ---------------------------------------------------------

with rasterio.open(input_file) as src:

    regions = src.read(1)


# ---------------------------------------------------------
# 3. Visualize
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    regions,
    interpolation="nearest"
)

plt.colorbar(
    label="High dNBR Region"
)

plt.title(
    "High dNBR Change Regions\n"
    "Harare Sentinel-2: "
    "12 October – 19 October 2025"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "High dNBR region map saved:",
    output_file
)