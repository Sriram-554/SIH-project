import matplotlib
matplotlib.use("Agg")
"""
SatQuery - Comprehensive Test Suite

Tests spectral index formulas, query classification, change detection,
visual grounding, fallback reasoning, and end-to-end pipeline execution.
"""

import unittest
from pathlib import Path
import numpy as np

from src.analysis.spectral_indices import SpectralEngine
from src.analysis.query_router import classify_query
from src.analysis.input_validator import inspect_file, validate_inputs
from src.analysis.change_detector import ChangeDetector
from src.analysis.grounding import VisualGrounder
from src.models.spectral_vqa_fallback import SpectralFallbackEngine
from src.pipeline import SatQueryPipeline


class TestSatQuery(unittest.TestCase):

    def setUp(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def test_spectral_indices_formulas(self):
        """Test NDVI, NDWI, and NBR calculations on known arrays."""
        nir = np.array([[0.8, 0.6], [0.1, 0.4]], dtype=np.float32)
        red = np.array([[0.2, 0.2], [0.1, 0.4]], dtype=np.float32)
        green = np.array([[0.3, 0.3], [0.5, 0.2]], dtype=np.float32)
        swir2 = np.array([[0.1, 0.2], [0.3, 0.1]], dtype=np.float32)

        # NDVI = (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
        ndvi = SpectralEngine.calculate_ndvi(nir, red)
        self.assertAlmostEqual(ndvi[0, 0], 0.6, places=4)

        # NDWI = (Green - NIR) / (Green + NIR) = (0.3 - 0.8) / (0.3 + 0.8) = -0.5 / 1.1 ~= -0.4545
        ndwi = SpectralEngine.calculate_ndwi(green, nir)
        self.assertAlmostEqual(ndwi[0, 0], -0.4545, places=3)

        # NBR = (NIR - SWIR2) / (NIR + SWIR2) = (0.8 - 0.1) / (0.8 + 0.1) = 0.7 / 0.9 ~= 0.7778
        nbr = SpectralEngine.calculate_nbr(nir, swir2)
        self.assertAlmostEqual(nbr[0, 0], 0.7778, places=3)

    def test_zonal_statistics(self):
        """Test zonal statistics and land cover categorization."""
        engine = SpectralEngine()
        # Create synthetic NDVI with dense veg (0.7), mod veg (0.3), barren (0.1), water (-0.3)
        ndvi = np.array([[0.7, 0.3], [0.1, -0.3]], dtype=np.float32)
        stats = engine.compute_zonal_statistics(ndvi)

        self.assertEqual(stats["dense_vegetation_percentage"], 25.0)
        self.assertEqual(stats["moderate_vegetation_percentage"], 25.0)
        self.assertEqual(stats["barren_builtup_percentage"], 25.0)
        self.assertEqual(stats["water_percentage"], 25.0)

    def test_query_router(self):
        """Test NLP query classification for various tasks."""
        self.assertEqual(classify_query("Describe the land cover and terrain.", 1), "captioning")
        self.assertEqual(classify_query("What type of vegetation is visible?", 1), "vqa")
        self.assertEqual(classify_query("Highlight the water body location", 1), "region_grounding")
        self.assertEqual(classify_query("What changed between these dates?", 2), "change_analysis")
        self.assertEqual(classify_query("Combine optical and SAR radar data", 2), "optical_sar_analysis")

    def test_input_validator(self):
        """Test file inspection and workflow classification."""
        res = inspect_file("outputs/sentinel_rgb.png")
        if Path("outputs/sentinel_rgb.png").exists():
            self.assertTrue(res["exists"])
            self.assertEqual(res["type"], "benchmark_image")

    def test_change_detection_rgb(self):
        """Test change detection on synthetic images."""
        detector = ChangeDetector()
        img1 = np.zeros((100, 100, 3), dtype=np.float32)
        img2 = np.zeros((100, 100, 3), dtype=np.float32)
        # Add change in 20% area
        img2[:20, :100, :] = 1.0

        res = detector.detect_change_rgb(img1, img2, threshold=0.1)
        self.assertIn("change_percentage", res)
        self.assertGreater(res["change_percentage"], 15.0)
        self.assertTrue(Path(res["difference_map_path"]).exists())

    def test_visual_grounding(self):
        """Test visual grounding module on synthetic test image."""
        grounder = VisualGrounder()
        rgb = np.ones((120, 120, 3), dtype=np.float32) * 0.5
        ndvi = np.ones((120, 120), dtype=np.float32) * 0.1
        # Set a water patch in top-left
        ndvi[:30, :30] = -0.4

        res = grounder.ground_feature(rgb, target_feature="water", ndvi=ndvi)
        self.assertIn("grounded_image_path", res)
        self.assertTrue(Path(res["grounded_image_path"]).exists())
        self.assertGreater(res["total_coverage_percentage"], 0.0)

    def test_spectral_fallback_engine(self):
        """Test rule-based deterministic spectral answer generation."""
        engine = SpectralFallbackEngine()
        stats = {
            "ndvi_mean": 0.55,
            "water_percentage": 12.0,
            "dense_vegetation_percentage": 50.0,
            "moderate_vegetation_percentage": 30.0,
            "barren_builtup_percentage": 8.0,
            "dominant_land_cover": "Dense Forest & Canopy"
        }

        res = engine.answer_query("Is there water present?", stats=stats)
        self.assertIn("water", res["answer"].lower())
        self.assertEqual(res["status"], "success_spectral_fallback")

    def test_end_to_end_pipeline(self):
        """Test full pipeline on Sentinel-2 dataset or RGB output."""
        pipeline = SatQueryPipeline()
        safe_path = Path("data/S2B_MSIL2A_20230207T101109_N0510_R022_T33TUL_20240813T033135.SAFE")

        input_target = safe_path if safe_path.exists() else "outputs/sentinel_rgb.png"
        if Path(input_target).exists():
            out = pipeline.process(
                query="Describe the dominant land cover and vegetation vigor.",
                primary_input=input_target
            )
            self.assertIn("answer", out)
            self.assertIn("task", out)
            self.assertIn("statistics", out)


if __name__ == "__main__":
    unittest.main()
