"""
SatQuery - Sentinel-2 Metadata Inspector

Reads a Sentinel-2 .SAFE product and extracts
basic remote-sensing metadata.
"""

from pathlib import Path
import re


def inspect_sentinel_product(product_path):
    """
    Inspect a Sentinel-2 .SAFE product from its
    directory name and internal structure.
    """

    product = Path(product_path)

    print("=" * 60)
    print("SATQUERY - SENTINEL-2 METADATA INSPECTOR")
    print("=" * 60)

    # -----------------------------------------------------
    # Check product
    # -----------------------------------------------------

    if not product.exists():
        print("\n[!] Product does not exist:")
        print(product)
        return None

    if product.suffix.upper() != ".SAFE":
        print("\n[!] This does not appear to be a .SAFE product.")
        return None

    print(f"\nProduct: {product.name}")

    # -----------------------------------------------------
    # Identify Sentinel-2 platform
    # -----------------------------------------------------

    platform = "Unknown"

    if product.name.startswith("S2A_"):
        platform = "Sentinel-2A"

    elif product.name.startswith("S2B_"):
        platform = "Sentinel-2B"

    # -----------------------------------------------------
    # Processing level
    # -----------------------------------------------------

    processing_level = "Unknown"

    if "_MSIL2A_" in product.name:
        processing_level = "Level-2A"

    elif "_MSIL1C_" in product.name:
        processing_level = "Level-1C"

    # -----------------------------------------------------
    # Extract acquisition date
    # -----------------------------------------------------

    date_match = re.search(
        r"_(\d{8})T\d{6}_",
        product.name
    )

    acquisition_date = "Unknown"

    if date_match:
        raw_date = date_match.group(1)

        acquisition_date = (
            f"{raw_date[:4]}-"
            f"{raw_date[4:6]}-"
            f"{raw_date[6:8]}"
        )

    # -----------------------------------------------------
    # Extract tile ID
    # -----------------------------------------------------

    tile_match = re.search(
        r"_T([0-9A-Z]{5,6})_",
        product.name
    )

    tile = "Unknown"

    if tile_match:
        tile = tile_match.group(1)

    # -----------------------------------------------------
    # Find image bands
    # -----------------------------------------------------

    bands = []

    for band in [
        "B01", "B02", "B03", "B04",
        "B05", "B06", "B07", "B08",
        "B8A", "B09", "B10", "B11", "B12"
    ]:

        if any(product.rglob(f"*_{band}_*.jp2")):
            bands.append(band)

    # -----------------------------------------------------
    # Detect resolutions
    # -----------------------------------------------------

    resolutions = []

    for resolution in ["R10m", "R20m", "R60m"]:

        if any(product.rglob(f"{resolution}")):
            resolutions.append(resolution.replace("R", ""))

    # -----------------------------------------------------
    # Determine sensor type
    # -----------------------------------------------------

    sensor_type = "Optical / Multispectral"

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("PRODUCT INFORMATION")
    print("-" * 60)

    print(f"Platform          : {platform}")
    print(f"Processing level   : {processing_level}")
    print(f"Acquisition date   : {acquisition_date}")
    print(f"Tile               : {tile}")
    print(f"Sensor type        : {sensor_type}")

    print("\nAvailable bands:")

    if bands:
        print(" ".join(bands))
    else:
        print("None detected")

    print("\nAvailable resolutions:")

    if resolutions:
        for resolution in resolutions:
            print(f"  {resolution} resolution")
    else:
        print("None detected")

    # -----------------------------------------------------
    # Determine available workflows
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("POSSIBLE SATQUERY WORKFLOWS")
    print("-" * 60)

    print("[+] Single-image analysis")
    print("[+] Visual question answering")
    print("[+] Captioning / scene description")
    print("[+] Spectral analysis")

    print("\n[!] Change analysis requires a second compatible image.")
    print("[!] Optical-SAR analysis requires a SAR image pair.")

    print("\n" + "=" * 60)
    print("METADATA INSPECTION COMPLETE")
    print("=" * 60)

    return {
        "platform": platform,
        "processing_level": processing_level,
        "acquisition_date": acquisition_date,
        "tile": tile,
        "sensor_type": sensor_type,
        "bands": bands,
        "resolutions": resolutions,
    }


# ---------------------------------------------------------
# Run test
# ---------------------------------------------------------

if __name__ == "__main__":

    PRODUCT = (
        "data/"
        "S2B_MSIL2A_20230207T101109_N0510_R022_"
        "T33TUL_20240813T033135.SAFE"
    )

    inspect_sentinel_product(PRODUCT)