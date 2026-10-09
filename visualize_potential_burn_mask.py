import rasterio
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output files
# ---------------------------------------------------------

input_file = (
    "data/satellite/"
    "harare_potential_burn_mask_20251019.tif"
)

output_file = Path(
    "reports/satellite/"
    "harare_potential_burn_mask_20251019.png"
)


# ---------------------------------------------------------
# 2. Read the mask
# ---------------------------------------------------------

src = rasterio.open(input_file)

mask = src.read(1)

src.close()


# ---------------------------------------------------------
# 3. Create visualization
# ---------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(
    mask,
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Potential Burn/Disturbance "
          "(0 = Other, 1 = Candidate)"
)

plt.title(
    "Harare Sentinel-2 Potential "
    "Burn/Disturbance Candidates - "
    "19 October 2025"
)

plt.axis("off")

plt.tight_layout()


# ---------------------------------------------------------
# 4. Save image
# ---------------------------------------------------------

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Potential burn/disturbance map saved:",
    output_file
)