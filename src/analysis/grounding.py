"""
SatQuery - Visual Grounding & Spatial Highlighting Module

Locates and highlights geographic features (water bodies, dense vegetation,
urban centers, anomalies) on satellite imagery, generating bounding boxes,
centroid coordinates, and overlay masks.

The grounding implementation is intentionally lightweight and dependency-safe:
- Uses NDWI/NDVI when available.
- Falls back to RGB heuristics for ordinary images.
- Removes no-data/black borders before localization.
- Uses connected-component style flood-fill on a downsampled mask instead of
  the previous fixed 8x8 grid, which could create thin/incorrect boxes.
- Reports region area relative to the valid observed scene.
"""

import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class VisualGrounder:
    """Performs spatial grounding and feature localization on remote sensing data."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _prepare_valid_mask(img_arr: np.ndarray) -> np.ndarray:
        """Identify pixels that belong to the observed image rather than no-data borders."""
        if img_arr.ndim != 3 or img_arr.shape[2] < 3:
            raise ValueError("RGB image must have shape (height, width, 3).")

        brightness = np.max(img_arr[:, :, :3], axis=2)
        # Black/no-data pixels are normally exactly zero. A small threshold also
        # removes near-black padding without removing normal dark water.
        return brightness > 0.015

    @staticmethod
    def _connected_components(
        mask: np.ndarray,
        min_pixels: int,
        max_components: int = 6,
        max_working_size: int = 700,
    ) -> List[Dict[str, Any]]:
        """
        Find spatially connected regions without requiring scipy/opencv.

        The mask is downsampled only for the component search. Bounding boxes
        are then mapped back to the original image and the original-resolution
        mask is used for area calculations. This keeps the implementation fast
        on large Sentinel-2 renders while avoiding the old fixed grid artifacts.
        """
        if mask.ndim != 2 or not np.any(mask):
            return []

        h, w = mask.shape
        stride = max(1, int(np.ceil(max(h, w) / max_working_size)))

        # Sampling preserves the spatial layout while keeping the flood-fill
        # problem bounded for very large satellite products.
        small = mask[::stride, ::stride].astype(bool, copy=False)
        sh, sw = small.shape

        # Scale the minimum component threshold to the downsampled mask.
        min_small_pixels = max(2, int(np.ceil(min_pixels / (stride * stride))))
        visited = np.zeros((sh, sw), dtype=bool)
        components: List[Tuple[int, int, int, int, int]] = []

        for sy in range(sh):
            for sx in range(sw):
                if not small[sy, sx] or visited[sy, sx]:
                    continue

                queue = deque([(sy, sx)])
                visited[sy, sx] = True
                count = 0
                min_y = max_y = sy
                min_x = max_x = sx

                while queue:
                    y, x = queue.popleft()
                    count += 1
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)

                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

                if count >= min_small_pixels:
                    components.append((count, min_y, min_x, max_y, max_x))

        components.sort(key=lambda item: item[0], reverse=True)

        boxes: List[Dict[str, Any]] = []
        for _, sy0, sx0, sy1, sx1 in components[:max_components]:
            # Expand sampled coordinates to original-image coordinates.
            y0 = max(0, sy0 * stride)
            x0 = max(0, sx0 * stride)
            y1 = min(h, (sy1 + 1) * stride)
            x1 = min(w, (sx1 + 1) * stride)

            region = mask[y0:y1, x0:x1]
            pixel_count = int(np.sum(region))
            if pixel_count <= 0:
                continue

            coords = np.argwhere(region)
            exact_y0 = int(y0 + coords[:, 0].min())
            exact_y1 = int(y0 + coords[:, 0].max())
            exact_x0 = int(x0 + coords[:, 1].min())
            exact_x1 = int(x0 + coords[:, 1].max())

            boxes.append({
                "bbox": [exact_y0, exact_x0, exact_y1, exact_x1],
                "centroid": [
                    int((exact_y0 + exact_y1) / 2),
                    int((exact_x0 + exact_x1) / 2),
                ],
                "pixel_count": pixel_count,
            })

        return boxes

    def ground_feature(
        self,
        rgb_image: Union[str, Path, np.ndarray],
        target_feature: str = "water",
        ndvi: Optional[np.ndarray] = None,
        ndwi: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Highlights and grounds a specific feature type on the image."""
        if isinstance(rgb_image, (str, Path)):
            img = Image.open(str(rgb_image)).convert("RGB")
            img_arr = np.array(img, dtype=np.float32) / 255.0
        else:
            img_arr = np.asarray(rgb_image).copy()
            if img_arr.ndim != 3 or img_arr.shape[2] < 3:
                raise ValueError("rgb_image must be an RGB image array.")
            if img_arr.max() > 1.0:
                img_arr = img_arr / 255.0

        img_arr = np.clip(img_arr[:, :, :3], 0.0, 1.0)
        h, w = img_arr.shape[:2]
        target_lower = target_feature.lower()
        valid_mask = self._prepare_valid_mask(img_arr)

        # Ensure supplied spectral arrays are shape-compatible before using them.
        ndvi_ok = ndvi is not None and np.shape(ndvi) == (h, w)
        ndwi_ok = ndwi is not None and np.shape(ndwi) == (h, w)

        if "water" in target_lower or "lake" in target_lower or "river" in target_lower:
            label = "Water Body"
            color = "cyan"
            if ndwi_ok:
                mask = np.isfinite(ndwi) & (ndwi > 0.05)
            elif ndvi_ok:
                mask = np.isfinite(ndvi) & (ndvi < -0.05)
            else:
                # RGB fallback: blue/cyan pixels that are not bright land.
                red = img_arr[:, :, 0]
                green = img_arr[:, :, 1]
                blue = img_arr[:, :, 2]
                mask = (blue > red * 1.08) & (blue >= green * 0.95) & (green < 0.65)

        elif "dense" in target_lower or "forest" in target_lower:
            label = "Dense Forest / Vegetation"
            color = "lime"
            if ndvi_ok:
                mask = np.isfinite(ndvi) & (ndvi >= 0.5)
            else:
                mask = (img_arr[:, :, 1] > img_arr[:, :, 0] * 1.2) & (
                    img_arr[:, :, 1] > img_arr[:, :, 2]
                )

        elif "vegetation" in target_lower or "agriculture" in target_lower or "crop" in target_lower:
            label = "Vegetation / Cropland"
            color = "chartreuse"
            if ndvi_ok:
                mask = np.isfinite(ndvi) & (ndvi >= 0.25)
            else:
                mask = img_arr[:, :, 1] > img_arr[:, :, 0] * 1.05

        elif "urban" in target_lower or "built" in target_lower or "barren" in target_lower:
            label = "Built-up / Barren Area"
            color = "orange"
            if ndvi_ok:
                mask = np.isfinite(ndvi) & (ndvi >= 0.0) & (ndvi < 0.2)
            else:
                brightness = np.mean(img_arr, axis=-1)
                mask = brightness > 0.45

        else:
            label = f"Region of Interest ({target_feature})"
            color = "magenta"
            brightness = np.mean(img_arr, axis=-1)
            mask = brightness > np.percentile(brightness[valid_mask], 80) if np.any(valid_mask) else brightness > 0.8

        # Never localize pixels outside the observed satellite scene.
        mask = np.asarray(mask, dtype=bool) & valid_mask

        # Remove isolated noise. The component detector performs the main
        # cleanup, while this threshold prevents tiny detections from appearing.
        min_area = max(30, int(h * w * 0.00005))
        boxes = self._connected_components(mask, min_pixels=min_area, max_components=6)

        valid_pixels = int(np.sum(valid_mask))
        total_feature_pixels = int(np.sum(mask))

        # Calculate each cluster's area against the valid observed scene,
        # not against its grid cell (the source of the previous 0.01% bug).
        for box in boxes:
            box["area_percentage"] = round(float(box["pixel_count"] / max(valid_pixels, 1) * 100.0), 2)

        # Generate a clean annotated plot. The axis is turned off without
        # changing the image extent, so boxes remain aligned with the source.
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_arr, interpolation="nearest")

        overlay = np.zeros((h, w, 4), dtype=np.float32)
        if color == "cyan":
            overlay[mask] = [0.0, 0.8, 1.0, 0.35]
        elif color == "lime":
            overlay[mask] = [0.0, 1.0, 0.2, 0.35]
        elif color == "chartreuse":
            overlay[mask] = [0.5, 1.0, 0.0, 0.35]
        elif color == "orange":
            overlay[mask] = [1.0, 0.5, 0.0, 0.35]
        else:
            overlay[mask] = [1.0, 0.0, 1.0, 0.35]
        ax.imshow(overlay, interpolation="nearest")

        for idx, box in enumerate(boxes, 1):
            y0, x0, y1, x1 = box["bbox"]
            rect = patches.Rectangle(
                (x0, y0),
                max(1, x1 - x0),
                max(1, y1 - y0),
                linewidth=2.5,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

            # Keep labels inside the image where possible.
            text_x = min(max(x0 + 4, 4), max(4, w - 180))
            text_y = min(max(y0 + 16, 16), max(16, h - 6))
            ax.text(
                text_x,
                text_y,
                f"#{idx} {label} ({box['area_percentage']:.2f}%)",
                color="white",
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
            )

        ax.set_xlim(0, w - 1)
        ax.set_ylim(h - 1, 0)
        ax.axis("off")
        ax.set_title(f"SatQuery Visual Grounding: {label} Localization", fontsize=13, weight="bold")
        plt.tight_layout(pad=0.2)

        safe_name = re.sub(r"[^\w\-_.]", "_", target_feature)
        out_path = self.output_dir / f"grounding_{safe_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        total_coverage = round(float(total_feature_pixels / max(valid_pixels, 1) * 100.0), 2)

        return {
            "target": target_feature,
            "label": label,
            "grounded_image_path": str(out_path),
            "total_coverage_percentage": total_coverage,
            "detected_clusters_count": len(boxes),
            "bounding_boxes": boxes,
            "summary": (
                f"Located {len(boxes)} major clusters of '{label}' covering "
                f"{total_coverage}% of the valid observed satellite scene."
            ),
        }
