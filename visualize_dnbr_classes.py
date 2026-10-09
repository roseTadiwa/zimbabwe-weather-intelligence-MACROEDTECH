import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path


# ---------------------------------------------------------
# 1. Input and output
# ---------------------------------------------------------

input_file = (
    "data/satellite/"
    "harare_dnbr_classes_20251012_20251019.tif"
)

output_file = Path(
    "reports/satellite/"
    "harare_dnbr_classified_map_20251012_20251019.png"
)


# ---------------------------------------------------------
# 2. Read classified raster
# ---------------------------------------------------------

with rasterio.open(input_file) as src:

    classes = src.read(1)


# ---------------------------------------------------------
# 3. Define display colours
# ---------------------------------------------------------

cmap = ListedColormap([
    "black",
    "red",
    "orange",
    "lightgray",
    "blue"
])


# ---------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 8)
)

image = plt.imshow(
    classes,
    cmap=cmap,
    vmin=0,
    vmax=4
)

cbar = plt.colorbar(
    image,
    ticks=[1, 2, 3, 4]
)

cbar.ax.set_yticklabels([
    "High positive change",
    "Moderate positive change",
    "Little / no change",
    "Negative change"
])

plt.title(
    "Harare Sentinel-2 dNBR Change Classification\n"
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
    "Classified dNBR map saved:",
    output_file
)