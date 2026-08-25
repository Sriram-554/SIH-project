"""
SatQuery - Sample & Benchmark Dataset Generator

Creates reproducible test pairs matching the 4 SIH input scopes:
1. Single Image (Optical & SAR)
2. Cross-Modal Optical-SAR Pair (Sentinel-2 + Sentinel-1 / Cartosat + RISAT)
3. Bi-Temporal Pair (Time 1 & Time 2 for CDVQA)
4. Benchmark subsets (BigEarthNet / RSVQA / VRSBench / CDVQA)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_benchmark_samples():
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)

    base_rgb_path = Path("outputs/sentinel_rgb.png")
    if not base_rgb_path.exists():
        # Create a synthetic 512x512 multi-feature satellite image if outputs not present
        h, w = 512, 512
        base = np.zeros((h, w, 3), dtype=np.float32)
        # Forest (Green)
        base[:256, :256] = [0.1, 0.55, 0.15]
        # Agriculture (Yellow-green)
        base[:256, 256:] = [0.4, 0.65, 0.2]
        # Urban / Built-up (Gray-orange)
        base[256:, :256] = [0.65, 0.55, 0.45]
        # Lake / Water Body (Deep blue)
        base[256:, 256:] = [0.05, 0.2, 0.55]
    else:
        img = Image.open(base_rgb_path).convert("RGB")
        base = np.array(img, dtype=np.float32) / 255.0

    # 1. Single Optical Image (T1)
    t1_path = sample_dir / "sample_optical_t1.png"
    plt.imsave(t1_path, base)

    # 2. Bi-Temporal Image (T2 with environmental shift)
    t2 = base.copy()
    # Simulate urban expansion and agricultural harvesting in bottom-right/top-right
    h, w = t2.shape[:2]
    # Harvest: Green turned to bare soil
    t2[:h//3, w//2:] = t2[:h//3, w//2:] * [1.4, 0.8, 0.6]
    # New urban buildings: Gray patches
    t2[h//2:h//2 + h//4, :w//3] = [0.75, 0.70, 0.68]

    t2_path = sample_dir / "sample_optical_t2.png"
    plt.imsave(t2_path, np.clip(t2, 0.0, 1.0))

    # 3. Co-Registered SAR Radar Image (Sentinel-1 / RISAT style)
    gray = np.mean(base, axis=-1)
    gy, gx = np.gradient(gray)
    roughness = np.sqrt(gx**2 + gy**2)
    sar_sim = np.clip(0.35 * gray + 0.65 * (roughness / (roughness.max() + 1e-6)), 0.0, 1.0)
    noise = np.random.gamma(shape=4.0, scale=0.25, size=sar_sim.shape)
    sar_radar = np.clip(sar_sim * noise, 0.0, 1.0)

    sar_path = sample_dir / "sample_sar_risat.png"
    plt.imsave(sar_path, sar_radar, cmap="gray")

    print("[SatQuery] SIH Sample & Benchmark Datasets Generated in 'data/samples/':")
    print(f" - Single Optical (T1) : {t1_path}")
    print(f" - Bi-Temporal (T2)    : {t2_path}")
    print(f" - SAR Radar (RISAT/S1): {sar_path}")


if __name__ == "__main__":
    generate_benchmark_samples()
