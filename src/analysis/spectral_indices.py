"""
SatQuery - Spectral Indices & Remote Sensing Engine

Extracts multi-spectral bands from Sentinel-2 (.SAFE or GeoTIFF)
and calculates remote-sensing indices (NDVI, NDWI, NBR, False-Color, etc.)
along with zonal land cover statistics.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt


class SpectralEngine:
    """Computes spectral indices and composites from satellite bands."""

    def __init__(self, safe_path_or_dir: Optional[str] = None):
        self.safe_path = Path(safe_path_or_dir) if safe_path_or_dir else None

    @staticmethod
    def normalize_band(arr: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
        """Stretch band reflectance values to [0, 1] for visual display."""
        low = np.percentile(arr, lower_pct)
        high = np.percentile(arr, upper_pct)
        if high - low < 1e-6:
            return np.clip(arr, 0.0, 1.0)
        norm = (arr - low) / (high - low + 1e-6)
        return np.clip(norm, 0.0, 1.0)

    @staticmethod
    def read_raster_band(band_path: Path, max_dimension: int = 1200) -> np.ndarray:
        """Reads a raster band with optional bilinear downsampling for speed."""
        with rasterio.open(band_path) as src:
            scale = min(1.0, max_dimension / max(src.width, src.height))
            out_w = max(1, int(src.width * scale))
            out_h = max(1, int(src.height * scale))
            return src.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.bilinear
            ).astype(np.float32)

    def find_sentinel_band(self, band_name: str, product_path: Optional[Path] = None) -> Optional[Path]:
        """Locates a specific Sentinel-2 band in a .SAFE folder."""
        root = product_path or self.safe_path
        if not root or not root.exists():
            return None

        candidates = list(root.rglob(f"*_{band_name}_10m.jp2"))
        if not candidates:
            candidates = list(root.rglob(f"*_{band_name}_20m.jp2"))
        if not candidates:
            candidates = list(root.rglob(f"*_{band_name}_60m.jp2"))
        if not candidates:
            candidates = list(root.rglob(f"*_{band_name}*.jp2"))
        if not candidates:
            candidates = list(root.rglob(f"*_{band_name}*.tif"))
        return candidates[0] if candidates else None

    def extract_bands_from_safe(self, safe_path: Path, max_dim: int = 1200) -> Dict[str, np.ndarray]:
        """Extracts key optical and infrared bands from Sentinel-2 product."""
        band_names = {
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "nir": "B08",
            "swir1": "B11",
            "swir2": "B12"
        }
        loaded_bands = {}
        for name, code in band_names.items():
            b_path = self.find_sentinel_band(code, safe_path)
            if b_path and b_path.exists():
                loaded_bands[name] = self.read_raster_band(b_path, max_dimension=max_dim)
        return loaded_bands

    @staticmethod
    def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        denom = nir + red + 1e-6
        return np.clip((nir - red) / denom, -1.0, 1.0)

    @staticmethod
    def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        denom = green + nir + 1e-6
        return np.clip((green - nir) / denom, -1.0, 1.0)

    @staticmethod
    def calculate_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        denom = nir + swir2 + 1e-6
        return np.clip((nir - swir2) / denom, -1.0, 1.0)

    def create_true_color_rgb(self, red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
        return np.dstack([
            self.normalize_band(red),
            self.normalize_band(green),
            self.normalize_band(blue)
        ])

    def create_false_color_infrared(self, nir: np.ndarray, red: np.ndarray, green: np.ndarray) -> np.ndarray:
        return np.dstack([
            self.normalize_band(nir),
            self.normalize_band(red),
            self.normalize_band(green)
        ])

    def create_swir_composite(self, swir1: np.ndarray, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        return np.dstack([
            self.normalize_band(swir1),
            self.normalize_band(nir),
            self.normalize_band(red)
        ])

    def compute_zonal_statistics(
        self,
        ndvi: np.ndarray,
        ndwi: Optional[np.ndarray] = None,
        nbr: Optional[np.ndarray] = None,
        brightness: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates statistics directly from the computed scene rasters."""
        total_pixels = ndvi.size
        if total_pixels == 0:
            raise ValueError("Cannot compute spectral statistics from an empty raster.")

        if ndwi is not None:
            water_mask = ndwi > 0.0
        else:
            water_mask = ndvi < 0.0

        barren_mask = (ndvi >= 0.0) & (ndvi < 0.2) & (~water_mask)
        mod_veg_mask = (ndvi >= 0.2) & (ndvi < 0.5)
        dense_veg_mask = ndvi >= 0.5

        water_pct = float(np.mean(water_mask) * 100.0)
        barren_pct = float(np.mean(barren_mask) * 100.0)
        mod_veg_pct = float(np.mean(mod_veg_mask) * 100.0)
        dense_veg_pct = float(np.mean(dense_veg_mask) * 100.0)

        stats = {
            "ndvi_mean": float(np.mean(ndvi)),
            "ndvi_min": float(np.min(ndvi)),
            "ndvi_max": float(np.max(ndvi)),
            "ndvi_std": float(np.std(ndvi)),
            "ndwi_mean": float(np.mean(ndwi)) if ndwi is not None else None,
            "nbr_mean": float(np.mean(nbr)) if nbr is not None else None,
            "water_percentage": round(water_pct, 2),
            "barren_builtup_percentage": round(barren_pct, 2),
            "moderate_vegetation_percentage": round(mod_veg_pct, 2),
            "dense_vegetation_percentage": round(dense_veg_pct, 2),
            "brightness": float(brightness) if brightness is not None else None,
            "dominant_land_cover": (
                "Dense Vegetation / Forest" if dense_veg_pct > max(barren_pct, mod_veg_pct, water_pct)
                else "Agricultural / Moderate Vegetation" if mod_veg_pct > max(barren_pct, water_pct)
                else "Barren / Urban / Built-up" if barren_pct > water_pct
                else "Water Body"
            )
        }
        return stats

    def process_safe_product(self, safe_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Runs complete spectral extraction, index calculation, and exports provenance-safe statistics."""
        safe_path = Path(safe_path)
        out_dir = Path(output_dir) if output_dir else Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        bands = self.extract_bands_from_safe(safe_path)
        if "red" not in bands or "green" not in bands or "blue" not in bands:
            raise ValueError(f"Essential RGB bands missing in {safe_path.name}")

        rgb = self.create_true_color_rgb(bands["red"], bands["green"], bands["blue"])
        rgb_path = out_dir / "sentinel_rgb.png"
        plt.imsave(rgb_path, rgb)

        results: Dict[str, Any] = {
            "rgb_path": str(rgb_path),
            "rgb_array": rgb,
            "bands_found": list(bands.keys()),
            "source_product": str(safe_path)
        }

        if "nir" in bands:
            ndvi = self.calculate_ndvi(bands["nir"], bands["red"])
            ndvi_path = out_dir / "sentinel_ndvi.png"
            plt.figure(figsize=(8, 6))
            plt.imshow(ndvi, vmin=-1.0, vmax=1.0, cmap="RdYlGn")
            plt.colorbar(label="NDVI")
            plt.axis("off")
            plt.title("Sentinel-2 NDVI Vegetation Index", fontsize=12)
            plt.tight_layout()
            plt.savefig(ndvi_path, dpi=150, bbox_inches="tight")
            plt.close()
            results["ndvi"] = ndvi
            results["ndvi_path"] = str(ndvi_path)

            ndwi = self.calculate_ndwi(bands["green"], bands["nir"])
            ndwi_path = out_dir / "sentinel_ndwi.png"
            plt.figure(figsize=(8, 6))
            plt.imshow(ndwi, vmin=-1.0, vmax=1.0, cmap="Blues")
            plt.colorbar(label="NDWI")
            plt.axis("off")
            plt.title("Sentinel-2 NDWI Water Index", fontsize=12)
            plt.tight_layout()
            plt.savefig(ndwi_path, dpi=150, bbox_inches="tight")
            plt.close()
            results["ndwi"] = ndwi
            results["ndwi_path"] = str(ndwi_path)

            nbr = None
            if "swir2" in bands:
                nbr = self.calculate_nbr(bands["nir"], bands["swir2"])
                nbr_path = out_dir / "sentinel_nbr.png"
                plt.figure(figsize=(8, 6))
                plt.imshow(nbr, vmin=-1.0, vmax=1.0, cmap="RdYlGn")
                plt.colorbar(label="NBR")
                plt.axis("off")
                plt.title("Sentinel-2 Normalized Burn Ratio", fontsize=12)
                plt.tight_layout()
                plt.savefig(nbr_path, dpi=150, bbox_inches="tight")
                plt.close()
                results["nbr"] = nbr
                results["nbr_path"] = str(nbr_path)

            fc_nir = self.create_false_color_infrared(bands["nir"], bands["red"], bands["green"])
            fc_path = out_dir / "sentinel_false_color.png"
            plt.imsave(fc_path, fc_nir)
            results["false_color_path"] = str(fc_path)
            results["false_color_array"] = fc_nir

            # Brightness is a measured scene statistic, not a placeholder.
            brightness = float(np.mean((bands["red"] + bands["green"] + bands["blue"]) / 3.0))
            stats = self.compute_zonal_statistics(ndvi, ndwi, nbr, brightness)
            results["statistics"] = stats

            # Persist the exact scene-derived statistics so downstream modules can consume
            # the same values without fabricating or duplicating measurements.
            stats_payload = {
                "source_product": str(safe_path),
                "bands_found": list(bands.keys()),
                "statistics": stats
            }
            stats_path = out_dir / "spectral_stats.json"
            stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
            results["statistics_path"] = str(stats_path)

        return results


if __name__ == "__main__":
    sample_safe = Path("data/S2B_MSIL2A_20230207T101109_N0510_R022_T33TUL_20240813T033135.SAFE")
    if sample_safe.exists():
        engine = SpectralEngine()
        res = engine.process_safe_product(sample_safe)
        print("Spectral Processing Complete!")
        print("RGB:", res["rgb_path"])
        print("NDVI:", res.get("ndvi_path"))
        print("Stats:", res.get("statistics"))
