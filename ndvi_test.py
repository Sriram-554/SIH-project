"""
SatQuery - Quick NDVI Test Utility
"""

from pathlib import Path
from src.analysis.spectral_indices import SpectralEngine

if __name__ == "__main__":
    sample_safe = Path("data/S2B_MSIL2A_20230207T101109_N0510_R022_T33TUL_20240813T033135.SAFE")
    if sample_safe.exists():
        print("Running SatQuery Spectral & NDVI extraction...")
        engine = SpectralEngine()
        res = engine.process_safe_product(sample_safe)
        print("RGB Output :", res["rgb_path"])
        print("NDVI Output:", res.get("ndvi_path"))
        print("NDWI Output:", res.get("ndwi_path"))
        print("Zonal Stats:", res.get("statistics"))
    else:
        print("No sample Sentinel-2 product found in data/.")
