"""
SatQuery - Visual Grounding & Spatial Highlighting Module

Locates and highlights geographic features (water bodies, dense vegetation,
urban centers, anomalies) on satellite imagery, generating bounding boxes,
centroid coordinates, and overlay masks.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class VisualGrounder:
    """Performs spatial grounding and feature localization on remote sensing data."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _find_bounding_boxes(mask: np.ndarray, min_area: int = 50, max_boxes: int = 10) -> List[Dict[str, Any]]:
        """Simple connected region bounding box finder using standard numpy/scipy-free logic."""
        # Simple raster labeling / threshold bounding box finder
        boxes = []
        h, w = mask.shape
        if not np.any(mask):
            return boxes

        # Grid subdivision method for fast robust spatial clustering without heavy cv2 dependency
        grid_rows, grid_cols = 8, 8
        gh, gw = h // grid_rows, w // grid_cols

        for r in range(grid_rows):
            for c in range(grid_cols):
                sub_mask = mask[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
                pixel_count = np.sum(sub_mask)
                if pixel_count >= min_area:
                    # Find exact extent in this sub-grid
                    coords = np.argwhere(sub_mask)
                    y0 = int(r * gh + coords[:, 0].min())
                    y1 = int(r * gh + coords[:, 0].max())
                    x0 = int(c * gw + coords[:, 1].min())
                    x1 = int(c * gw + coords[:, 1].max())

                    area_pct = float(pixel_count / (h * w) * 100.0)
                    boxes.append({
                        "bbox": [y0, x0, y1, x1],
                        "centroid": [int((y0 + y1) / 2), int((x0 + x1) / 2)],
                        "pixel_count": int(pixel_count),
                        "area_percentage": round(area_pct, 2)
                    })

        # Sort by pixel count descending
        boxes.sort(key=lambda b: b["pixel_count"], reverse=True)
        return boxes[:max_boxes]

    def ground_feature(
        self,
        rgb_image: Union[str, Path, np.ndarray],
        target_feature: str = "water",
        ndvi: Optional[np.ndarray] = None,
        ndwi: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Highlights and grounds a specific feature type on the image."""
        if isinstance(rgb_image, (str, Path)):
            img = Image.open(str(rgb_image)).convert("RGB")
            img_arr = np.array(img, dtype=np.float32) / 255.0
        else:
            img_arr = rgb_image.copy()
            if img_arr.max() > 1.0:
                img_arr = img_arr / 255.0

        h, w = img_arr.shape[:2]
        target_lower = target_feature.lower()

        # Build feature mask based on spectral indices or RGB thresholding
        if "water" in target_lower or "lake" in target_lower or "river" in target_lower:
            label = "Water Body"
            color = "cyan"
            if ndwi is not None:
                mask = ndwi > 0.05
            elif ndvi is not None:
                mask = ndvi < -0.05
            else:
                # RGB heuristic: blue dominant & dark
                mask = (img_arr[:, :, 2] > img_arr[:, :, 0]) & (img_arr[:, :, 1] < 0.4)
        elif "dense" in target_lower or "forest" in target_lower:
            label = "Dense Forest / Vegetation"
            color = "lime"
            if ndvi is not None:
                mask = ndvi >= 0.5
            else:
                mask = (img_arr[:, :, 1] > img_arr[:, :, 0] * 1.2) & (img_arr[:, :, 1] > img_arr[:, :, 2])
        elif "vegetation" in target_lower or "agriculture" in target_lower or "crop" in target_lower:
            label = "Vegetation / Cropland"
            color = "chartreuse"
            if ndvi is not None:
                mask = (ndvi >= 0.25)
            else:
                mask = img_arr[:, :, 1] > img_arr[:, :, 0]
        elif "urban" in target_lower or "built" in target_lower or "barren" in target_lower:
            label = "Built-up / Barren Area"
            color = "orange"
            if ndvi is not None:
                mask = (ndvi >= 0.0) & (ndvi < 0.2)
            else:
                brightness = np.mean(img_arr, axis=-1)
                mask = brightness > 0.45
        else:
            label = f"Region of Interest ({target_feature})"
            color = "magenta"
            brightness = np.mean(img_arr, axis=-1)
            mask = brightness > np.percentile(brightness, 80)

        # Detect bounding clusters
        boxes = self._find_bounding_boxes(mask, min_area=30, max_boxes=6)

        # Generate annotated plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_arr)

        # Create semi-transparent overlay mask
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        if color == "cyan":
            overlay[mask] = [0.0, 0.8, 1.0, 0.4]
        elif color == "lime":
            overlay[mask] = [0.0, 1.0, 0.2, 0.4]
        elif color == "chartreuse":
            overlay[mask] = [0.5, 1.0, 0.0, 0.4]
        elif color == "orange":
            overlay[mask] = [1.0, 0.5, 0.0, 0.4]
        else:
            overlay[mask] = [1.0, 0.0, 1.0, 0.4]

        ax.imshow(overlay)

        # Draw bounding boxes
        for idx, b in enumerate(boxes, 1):
            y0, x0, y1, x1 = b["bbox"]
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2.5,
                edgecolor=color,
                facecolor="none",
                linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(
                x0 + 4, y0 + 16,
                f"#{idx} {label} ({b['area_percentage']}%)",
                color="white",
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7)
            )

        ax.axis("off")
        ax.set_title(f"SatQuery Visual Grounding: {label} Localization", fontsize=13, weight="bold")
        plt.tight_layout()

        safe_name = re.sub(r"[^\w\-_.]", "_", target_feature)
        out_path = self.output_dir / f"grounding_{safe_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        total_coverage = round(float(np.sum(mask) / (h * w) * 100.0), 2)

        return {
            "target": target_feature,
            "label": label,
            "grounded_image_path": str(out_path),
            "total_coverage_percentage": total_coverage,
            "detected_clusters_count": len(boxes),
            "bounding_boxes": boxes,
            "summary": (
                f"Located {len(boxes)} major clusters of '{label}' covering {total_coverage}% "
                f"of the observed satellite scene."
            )
        }
