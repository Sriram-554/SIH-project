"""
SatQuery - Input Validator

Checks remote-sensing image inputs before they are sent
to the SatQuery analysis pipeline.
"""

from pathlib import Path


# ---------------------------------------------------------
# Supported formats
# ---------------------------------------------------------

SUPPORTED_GEOSPATIAL = {".tif", ".tiff", ".jp2", ".safe"}
SUPPORTED_BENCHMARK = {".png", ".jpg", ".jpeg"}

SUPPORTED_FORMATS = SUPPORTED_GEOSPATIAL | SUPPORTED_BENCHMARK


# ---------------------------------------------------------
# Basic file inspection
# ---------------------------------------------------------

def inspect_file(file_path):
    """
    Inspect one input file and return basic information.
    """

    path = Path(file_path)

    result = {
        "file": path.name,
        "path": str(path),
        "exists": path.exists(),
        "extension": path.suffix.lower(),
        "valid_format": False,
        "type": "unknown",
    }

    # Check whether file exists
    if not path.exists():
        return result

    # Check extension
    if path.suffix.lower() not in SUPPORTED_FORMATS and not (path.is_dir() and path.suffix.lower() == ".safe"):
        return result

    result["valid_format"] = True

    # Determine input type
    if path.suffix.lower() == ".safe" or (path.is_dir() and path.suffix.lower() == ".safe"):
        result["type"] = "sentinel2_safe_product"
    elif path.suffix.lower() in {".tif", ".tiff", ".jp2"}:
        result["type"] = "geospatial_raster"
    elif path.suffix.lower() in SUPPORTED_BENCHMARK:
        result["type"] = "benchmark_image"

    return result


# ---------------------------------------------------------
# Analyze a collection of inputs
# ---------------------------------------------------------

def validate_inputs(file_paths):
    """
    Validate one or more remote-sensing image inputs.
    """

    print("=" * 60)
    print("SATQUERY - INPUT VALIDATOR")
    print("=" * 60)

    print(f"\nNumber of inputs: {len(file_paths)}")

    results = []

    for file_path in file_paths:

        result = inspect_file(file_path)
        results.append(result)

        print("\n----------------------------------------")
        print(f"File       : {result['file']}")
        print(f"Exists     : {result['exists']}")
        print(f"Extension  : {result['extension']}")
        print(f"Format OK  : {result['valid_format']}")
        print(f"Type       : {result['type']}")

    # -----------------------------------------------------
    # Determine workflow
    # -----------------------------------------------------

    valid_files = [
        r for r in results
        if r["exists"] and r["valid_format"]
    ]

    workflow = "invalid"

    if len(valid_files) == 0:
        workflow = "invalid"

    elif len(valid_files) == 1:
        workflow = "single_image"

    elif len(valid_files) == 2:
        workflow = "multi_image"

    else:
        workflow = "unsupported_number_of_inputs"

    print("\n========================================")
    print("VALIDATION RESULT")
    print("========================================")

    print(f"Valid inputs : {len(valid_files)}")
    print(f"Workflow     : {workflow}")

    return {
        "valid": len(valid_files) == len(file_paths),
        "files": results,
        "workflow": workflow,
    }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    # Test with the Sentinel-2 RGB output
    test_file = "outputs/sentinel_rgb.png"

    result = validate_inputs([test_file])

    print("\n========================================")
    print("INPUT VALIDATION COMPLETE")
    print("========================================")