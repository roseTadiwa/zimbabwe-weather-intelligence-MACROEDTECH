import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from pathlib import Path
import pystac_client
import planetary_computer


# ---------------------------------------------------------
# 1. Connect to Microsoft Planetary Computer
# ---------------------------------------------------------

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)


# ---------------------------------------------------------
# 2. Search for Sentinel-2 scenes
# ---------------------------------------------------------
#
# We use the same tile and approximate Harare area.
# The dates are:
#
# 12 October 2025 = pre-event/reference
# 19 October 2025 = post-event/reference
#
# The best scene available for each date is selected
# using cloud cover.
# ---------------------------------------------------------

def find_scene(date_string):

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[
            31.00,
            -17.90,
            31.10,
            -17.75
        ],
        datetime=f"{date_string}T00:00:00Z/{date_string}T23:59:59Z",
        query={
            "eo:cloud_cover": {
                "lt": 20
            }
        }
    )

    scenes = list(search.items())

    if not scenes:
        raise RuntimeError(
            f"No Sentinel-2 scene found for {date_string}"
        )

    scenes.sort(
        key=lambda item:
        item.properties.get(
            "eo:cloud_cover",
            100
        )
    )

    scene = scenes[0]

    print()
    print("Selected scene:")
    print("Date:", date_string)
    print("ID:", scene.id)
    print(
        "Cloud cover:",
        scene.properties.get(
            "eo:cloud_cover"
        )
    )

    return scene


# ---------------------------------------------------------
# 3. Find the two scenes
# ---------------------------------------------------------

pre_item = find_scene(
    "2025-10-12"
)

post_item = find_scene(
    "2025-10-19"
)


# ---------------------------------------------------------
# 4. Calculate NBR for a scene
# ---------------------------------------------------------

def calculate_nbr(
    item,
    reference_window=None
):

    b08_url = item.assets["B08"].href
    b12_url = item.assets["B12"].href


    # ---------------------------------------------
    # Read B08 - NIR
    # ---------------------------------------------

    nir_src = rasterio.open(
        b08_url
    )

    if reference_window is None:

        nir_window = Window(
            0,
            0,
            1000,
            1000
        )

    else:

        nir_window = from_bounds(
            *reference_window,
            transform=nir_src.transform
        )


    nir = nir_src.read(
        1,
        window=nir_window
    ).astype(
        np.float32
    )


    left, bottom, right, top = (
        rasterio.windows.bounds(
            nir_window,
            nir_src.transform
        )
    )


    output_transform = (
        nir_src.window_transform(
            nir_window
        )
    )


    profile = nir_src.profile.copy()

    nir_src.close()


    # ---------------------------------------------
    # Read B12 - SWIR
    # ---------------------------------------------

    swir_src = rasterio.open(
        b12_url
    )


    swir_window = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=swir_src.transform
    )


    swir = swir_src.read(
        1,
        window=swir_window,
        out_shape=(
            1000,
            1000
        ),
        resampling=Resampling.bilinear
    ).astype(
        np.float32
    )


    swir_src.close()


    # ---------------------------------------------
    # Calculate NBR
    # ---------------------------------------------

    denominator = nir + swir

    nbr = np.where(
        denominator != 0,
        (nir - swir) / denominator,
        np.nan
    ).astype(
        np.float32
    )


    return (
        nbr,
        output_transform,
        profile,
        (
            left,
            bottom,
            right,
            top
        )
    )


# ---------------------------------------------------------
# 5. Calculate pre-event NBR
# ---------------------------------------------------------

print()
print("Calculating 12 October NBR...")

(
    pre_nbr,
    pre_transform,
    pre_profile,
    reference_bounds
) = calculate_nbr(
    pre_item
)


# ---------------------------------------------------------
# 6. Calculate post-event NBR
# ---------------------------------------------------------

print()
print("Calculating 19 October NBR...")

(
    post_nbr,
    post_transform,
    post_profile,
    _
) = calculate_nbr(
    post_item,
    reference_window=reference_bounds
)


# ---------------------------------------------------------
# 7. Check dimensions
# ---------------------------------------------------------

if pre_nbr.shape != post_nbr.shape:

    raise ValueError(
        f"NBR dimensions do not match: "
        f"{pre_nbr.shape} vs "
        f"{post_nbr.shape}"
    )


# ---------------------------------------------------------
# 8. Calculate dNBR
#
# dNBR = Pre-event NBR - Post-event NBR
# ---------------------------------------------------------

dnbr = (
    pre_nbr - post_nbr
).astype(
    np.float32
)


# ---------------------------------------------------------
# 9. Save pre-event NBR
# ---------------------------------------------------------

pre_output = Path(
    "data/satellite/"
    "harare_nbr_20251012.tif"
)

pre_profile.update(
    height=1000,
    width=1000,
    count=1,
    dtype="float32",
    transform=pre_transform,
    nodata=np.nan
)

with rasterio.open(
    pre_output,
    "w",
    **pre_profile
) as dst:

    dst.write(
        pre_nbr,
        1
    )


# ---------------------------------------------------------
# 10. Save post-event NBR
# ---------------------------------------------------------

post_output = Path(
    "data/satellite/"
    "harare_nbr_20251019_reference.tif"
)

post_profile.update(
    height=1000,
    width=1000,
    count=1,
    dtype="float32",
    transform=post_transform,
    nodata=np.nan
)

with rasterio.open(
    post_output,
    "w",
    **post_profile
) as dst:

    dst.write(
        post_nbr,
        1
    )


# ---------------------------------------------------------
# 11. Save dNBR
# ---------------------------------------------------------

dnbr_output = Path(
    "data/satellite/"
    "harare_dnbr_20251012_20251019.tif"
)

dnbr_profile = pre_profile.copy()

dnbr_profile.update(
    height=1000,
    width=1000,
    count=1,
    dtype="float32",
    transform=pre_transform,
    nodata=np.nan
)

with rasterio.open(
    dnbr_output,
    "w",
    **dnbr_profile
) as dst:

    dst.write(
        dnbr,
        1
    )


# ---------------------------------------------------------
# 12. Statistics
# ---------------------------------------------------------

valid = dnbr[
    np.isfinite(dnbr)
]


print()
print("==========================================")
print("dNBR ANALYSIS")
print("==========================================")

print(
    "Shape:",
    dnbr.shape
)

print(
    "Minimum:",
    float(np.min(valid))
)

print(
    "Maximum:",
    float(np.max(valid))
)

print(
    "Mean:",
    float(np.mean(valid))
)

print(
    "Median:",
    float(np.median(valid))
)


print()
print(
    "Pre-event NBR saved:",
    pre_output
)

print(
    "Post-event NBR saved:",
    post_output
)

print(
    "dNBR saved:",
    dnbr_output
)