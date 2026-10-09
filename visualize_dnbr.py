import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output
# ---------------------------------------------------------

input_file = (
    "data/satellite/"
    "harare_dnbr_20251012_20251019.tif"
)

output_file = Path(
    "reports/satellite/"
    "harare_dnbr_map_20251012_20251019.png"
)


# ---------------------------------------------------------
# 2. Read dNBR
# ---------------------------------------------------------

with rasterio.open(input_file) as src:

    dnbr = src.read(1)


# ---------------------------------------------------------
# 3. Create visualization
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 8)
)

image = plt.imshow(
    dnbr,
    vmin=-0.5,
    vmax=0.5
)

plt.colorbar(
    image,
    label="dNBR"
)

plt.title(
    "Harare Sentinel-2 dNBR Change Map\n"
    "12 October 2025 – 19 October 2025"
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
    "dNBR map saved:",
    output_file
)