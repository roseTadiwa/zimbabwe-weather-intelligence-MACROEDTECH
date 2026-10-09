import rasterio
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output files
# ---------------------------------------------------------

input_file = (
    "data/satellite/"
    "harare_nbr_disturbance_mask_20251019.tif"
)

output_file = Path(
    "reports/satellite/"
    "harare_nbr_disturbance_mask_20251019.png"
)


# ---------------------------------------------------------
# 2. Read the disturbance mask
# ---------------------------------------------------------

src = rasterio.open(input_file)

mask = src.read(1)

src.close()


# ---------------------------------------------------------
# 3. Create the visualization
# ---------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(mask, vmin=0, vmax=1)

plt.colorbar(
    label="Potential Disturbance (0 = Other, 1 = Low NBR)"
)

plt.title(
    "Harare Sentinel-2 Potential "
    "Burn/Disturbance Mask - 19 October 2025"
)

plt.axis("off")

plt.tight_layout()


# ---------------------------------------------------------
# 4. Save the figure
# ---------------------------------------------------------

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 5. Confirm output
# ---------------------------------------------------------

print(
    "Disturbance mask map saved:",
    output_file
)