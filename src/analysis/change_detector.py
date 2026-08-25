"""
SatQuery - Change Detection Engine

Performs bi-temporal change analysis on multi-date satellite observations.
Computes delta NDVI, structural pixel differences, and categorizes land-use transitions
(vegetation loss/deforestation, vegetation regrowth, water boundary changes).
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ChangeDetector:
    """Detects and quantifies environmental & structural changes between two time steps."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_image_as_array(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Loads an image path or array into normalized [0, 1] float32 numpy array."""
        if isinstance(image_input, np.ndarray):
            arr = image_input.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr

        img = Image.open(str(image_input)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def detect_change_rgb(
        self,
        img_t1_path: Union[str, Path, np.ndarray],
        img_t2_path: Union[str, Path, np.ndarray],
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """Calculates structural RGB difference between two dates."""
        t1 = self.load_image_as_array(img_t1_path)
        t2 = self.load_image_as_array(img_t2_path)

        # Resize if dimensions differ
        if t1.shape[:2] != t2.shape[:2]:
            h, w = min(t1.shape[0], t2.shape[0]), min(t1.shape[1], t2.shape[1])
            t1 = t1[:h, :w]
            t2 = t2[:h, :w]

        # Euclidean difference in RGB space
        diff = np.sqrt(np.sum((t2 - t1) ** 2, axis=-1)) / np.sqrt(3.0)
        changed_mask = diff > threshold

        total_pixels = diff.size
        change_pct = float(np.sum(changed_mask) / total_pixels * 100.0)

        # Save change visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(t1)
        axes[0].set_title("Time 1 (Initial)", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(t2)
        axes[1].set_title("Time 2 (Recent)", fontsize=11)
        axes[1].axis("off")

        im = axes[2].imshow(diff, cmap="coolwarm", vmin=0, vmax=1)
        axes[2].set_title(f"Change Heatmap ({change_pct:.1f}% modified)", fontsize=11)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_path = self.output_dir / "change_analysis_rgb.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {
            "task": "change_detection_rgb",
            "difference_map_path": str(out_path),
            "mean_difference": float(np.mean(diff)),
            "max_difference": float(np.max(diff)),
            "change_percentage": round(change_pct, 2),
            "summary": (
                f"Detected {change_pct:.1f}% surface modification between the two observations. "
                f"Mean structural shift is {np.mean(diff):.3f}."
            )
        }

    def detect_change_ndvi(
        self,
        ndvi_t1: np.ndarray,
        ndvi_t2: np.ndarray,
        t1_label: str = "T1",
        t2_label: str = "T2"
    ) -> Dict[str, Any]:
        """Calculates Delta NDVI (NDVI_T2 - NDVI_T1) and categorizes vegetation dynamics."""
        # Align shapes if needed
        if ndvi_t1.shape != ndvi_t2.shape:
            h = min(ndvi_t1.shape[0], ndvi_t2.shape[0])
            w = min(ndvi_t1.shape[1], ndvi_t2.shape[1])
            ndvi_t1 = ndvi_t1[:h, :w]
            ndvi_t2 = ndvi_t2[:h, :w]

        delta_ndvi = ndvi_t2 - ndvi_t1
        total_pixels = delta_ndvi.size

        # Classification thresholds
        # Significant Loss: delta < -0.15
        # Significant Gain: delta > +0.15
        # Stable: -0.15 <= delta <= 0.15
        loss_mask = delta_ndvi < -0.15
        gain_mask = delta_ndvi > 0.15
        stable_mask = (~loss_mask) & (~gain_mask)

        loss_pct = float(np.sum(loss_mask) / total_pixels * 100.0)
        gain_pct = float(np.sum(gain_mask) / total_pixels * 100.0)
        stable_pct = float(np.sum(stable_mask) / total_pixels * 100.0)

        # Plot Delta NDVI
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        im0 = axes[0].imshow(ndvi_t1, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[0].set_title(f"NDVI {t1_label}", fontsize=11)
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(ndvi_t2, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[1].set_title(f"NDVI {t2_label}", fontsize=11)
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(delta_ndvi, cmap="bwr", vmin=-0.5, vmax=0.5)
        axes[2].set_title(f"Delta NDVI ({gain_pct:.1f}% Gain, {loss_pct:.1f}% Loss)", fontsize=11)
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Δ NDVI")

        plt.tight_layout()
        out_path = self.output_dir / "change_analysis_ndvi.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {
            "task": "change_detection_ndvi",
            "delta_ndvi_path": str(out_path),
            "delta_mean": float(np.mean(delta_ndvi)),
            "vegetation_gain_percentage": round(gain_pct, 2),
            "vegetation_loss_percentage": round(loss_pct, 2),
            "stable_percentage": round(stable_pct, 2),
            "summary": (
                f"Vegetation Change Analysis: {gain_pct:.1f}% area experienced biomass growth/greening, "
                f"{loss_pct:.1f}% experienced vegetation loss/clearing, and {stable_pct:.1f}% remained stable."
            )
        }
