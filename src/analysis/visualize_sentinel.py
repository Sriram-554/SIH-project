"""
SatQuery - Sentinel-2 Visualization Module

Generates True-Color RGB, False-Color NIR, and NDVI maps
from raw Sentinel-2 .SAFE products.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from src.analysis.spectral_indices import SpectralEngine


def generate_sentinel_visualizations(
    safe_path: Optional[str] = None,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """Generates RGB, NDVI, and False-Color maps from Sentinel-2 product."""
    data_dir = Path("data")
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if safe_path:
        target = Path(safe_path)
    else:
        safe_folders = list(data_dir.glob("*.SAFE"))
        if not safe_folders:
            raise FileNotFoundError("No Sentinel-2 .SAFE product found in data/")
        target = safe_folders[0]

    print("=" * 60)
    print("SATQUERY - SENTINEL-2 VISUALIZATION")
    print("=" * 60)
    print(f"\nProcessing product: {target.name}")

    engine = SpectralEngine()
    results = engine.process_safe_product(target, output_dir=out_dir)

    print(f"\n[OK] RGB Image: {results['rgb_path']}")
    if "ndvi_path" in results:
        print(f"[OK] NDVI Map : {results['ndvi_path']}")
    if "ndwi_path" in results:
        print(f"[OK] NDWI Map : {results['ndwi_path']}")
    if "statistics" in results:
        stats = results["statistics"]
        print("\nNDVI Statistics:")
        print(f"  Mean NDVI   : {stats['ndvi_mean']:.3f}")
        print(f"  Dense Veg   : {stats['dense_vegetation_percentage']}%")
        print(f"  Water Body  : {stats['water_percentage']}%")
        print(f"  Barren/Urban: {stats['barren_builtup_percentage']}%")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    generate_sentinel_visualizations()