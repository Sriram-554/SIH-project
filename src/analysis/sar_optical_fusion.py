"""
SatQuery - Cross-Modal Optical + SAR Fusion Engine

Processes co-registered optical/multispectral and SAR image pairs to extract
complementary spectral and structural information.

Prototype note: when no real SAR raster is supplied, a synthetic SAR-like
image may be generated for demonstration. Synthetic SAR is explicitly marked
in the returned result and must not be interpreted as sensor measurements.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio


class OpticalSARFusionEngine:
    """Prototype optical + SAR fusion and heuristic cross-modal analysis."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_raster_or_image(img_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load an optical or SAR raster/image and normalize it to [0, 1]."""
        if isinstance(img_input, np.ndarray):
            arr = img_input.astype(np.float32)
            if arr.size and arr.max() > 1.0:
                arr = arr / 255.0
            return np.clip(arr, 0.0, 1.0)

        path = Path(img_input)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        if path.suffix.lower() in [".tif", ".tiff", ".jp2"]:
            with rasterio.open(path) as src:
                count = min(src.count, 3)
                data = src.read(list(range(1, count + 1))).astype(np.float32)
                if data.ndim == 3:
                    data = np.transpose(data, (1, 2, 0))
                low, high = np.percentile(data, 2), np.percentile(data, 98)
                if high - low > 1e-6:
                    data = np.clip((data - low) / (high - low), 0.0, 1.0)
                else:
                    data = np.clip(data, 0.0, 1.0)
                return data

        return np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32) / 255.0

    def generate_synthetic_sar_from_optical(self, optical_rgb: np.ndarray) -> np.ndarray:
        """Generate a SAR-like visualization for prototype/demo fallback only."""
        gray = np.mean(optical_rgb, axis=-1)
        gy, gx = np.gradient(gray)
        roughness = np.sqrt(gx**2 + gy**2)
        sar_sim = np.clip(
            0.4 * gray + 0.6 * (roughness / (roughness.max() + 1e-6)),
            0.0,
            1.0,
        )
        noise = np.random.default_rng(42).gamma(shape=4.0, scale=0.25, size=sar_sim.shape)
        return np.clip(sar_sim * noise, 0.0, 1.0).astype(np.float32)

    def fuse_optical_and_sar(
        self,
        optical_input: Union[str, Path, np.ndarray],
        sar_input: Optional[Union[str, Path, np.ndarray]] = None,
        query: str = "Use optical and SAR images together to identify built-up and water-covered regions."
    ) -> Dict[str, Any]:
        """Execute prototype cross-modal analysis.

        The current classification is heuristic. It is intended for a hackathon
        demonstration, not calibrated geophysical retrieval or validated land-cover mapping.
        """
        opt_arr = self.load_raster_or_image(optical_input)
        if opt_arr.ndim == 2:
            opt_arr = np.dstack([opt_arr, opt_arr, opt_arr])
        if opt_arr.shape[-1] == 1:
            opt_arr = np.repeat(opt_arr, 3, axis=-1)

        sar_source = "real_sar_input"
        if sar_input is not None and Path(str(sar_input)).exists() if isinstance(sar_input, (str, Path)) else sar_input is not None:
            sar_arr = self.load_raster_or_image(sar_input)
            if sar_arr.ndim == 3:
                sar_arr = np.mean(sar_arr, axis=-1)
        else:
            sar_arr = self.generate_synthetic_sar_from_optical(opt_arr)
            sar_source = "synthetic_sar_fallback"

        h = min(opt_arr.shape[0], sar_arr.shape[0])
        w = min(opt_arr.shape[1], sar_arr.shape[1])
        opt_arr = opt_arr[:h, :w]
        sar_arr = sar_arr[:h, :w]

        total_pixels = max(h * w, 1)
        # These are normalized-intensity heuristics, not calibrated dB measurements.
        water_mask = (sar_arr < 0.18) & (opt_arr[:, :, 2] >= opt_arr[:, :, 0] * 0.8)
        urban_mask = (sar_arr > 0.45) & (~water_mask)
        vegetation_mask = (opt_arr[:, :, 1] > opt_arr[:, :, 0] * 1.1) & (~urban_mask) & (~water_mask)
        bare_soil_mask = (~water_mask) & (~urban_mask) & (~vegetation_mask)

        water_pct = float(np.sum(water_mask) / total_pixels * 100.0)
        urban_pct = float(np.sum(urban_mask) / total_pixels * 100.0)
        veg_pct = float(np.sum(vegetation_mask) / total_pixels * 100.0)
        soil_pct = float(np.sum(bare_soil_mask) / total_pixels * 100.0)

        cross_modal_composite = np.dstack([opt_arr[:, :, 0], opt_arr[:, :, 1], sar_arr])

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(opt_arr)
        axes[0].set_title("Optical Input", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(sar_arr, cmap="gray")
        axes[1].set_title("SAR / SAR-like Intensity", fontsize=11)
        axes[1].axis("off")
        axes[2].imshow(cross_modal_composite)
        axes[2].set_title("Optical + SAR Fusion", fontsize=11)
        axes[2].axis("off")
        plt.tight_layout()

        fusion_img_path = self.output_dir / "optical_sar_fusion_analysis.png"
        plt.savefig(fusion_img_path, dpi=150, bbox_inches="tight")
        plt.close()

        if sar_source == "real_sar_input":
            source_note = "Analysis used the supplied SAR raster. The reported percentages are heuristic intensity-based estimates, not a validated classifier."
        else:
            source_note = "No usable SAR raster was supplied; a deterministic synthetic SAR-like representation was generated for demonstration."

        answer = (
            "**Cross-Modal Optical-SAR Prototype Analysis**:\n"
            f"- **Built-up / high-backscatter candidate region**: **{urban_pct:.1f}%**\n"
            f"- **Water / low-backscatter candidate region**: **{water_pct:.1f}%**\n"
            f"- **Vegetation candidate region**: **{veg_pct:.1f}%**\n"
            f"- **Bare soil / transition candidate region**: **{soil_pct:.1f}%**\n\n"
            f"**Data source:** {sar_source.replace('_', ' ')}.\n"
            f"**Interpretation:** {source_note}"
        )

        return {
            "task": "optical_sar_analysis",
            "query": query,
            "fusion_image_path": str(fusion_img_path),
            "urban_percentage": round(urban_pct, 2),
            "water_percentage": round(water_pct, 2),
            "vegetation_percentage": round(veg_pct, 2),
            "soil_percentage": round(soil_pct, 2),
            "sar_source": sar_source,
            "analysis_type": "heuristic_cross_modal_prototype",
            "answer": answer,
            "confidence": 0.75 if sar_source == "real_sar_input" else 0.60,
            "status": "success"
        }


if __name__ == "__main__":
    test_opt = "outputs/sentinel_rgb.png"
    if Path(test_opt).exists():
        engine = OpticalSARFusionEngine()
        res = engine.fuse_optical_and_sar(test_opt)
        print("Optical-SAR Fusion Results:")
        print(res["answer"])
        print("Composite Path:", res["fusion_image_path"])
