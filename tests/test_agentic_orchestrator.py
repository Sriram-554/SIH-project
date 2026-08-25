"""
SatQuery - Comprehensive Agentic & SIH Compliance Test Suite

Tests:
1. ToolRegistry and all 6 specialist remote sensing tools.
2. Optical-SAR cross-modal fusion engine.
3. CDVQA bi-temporal change reasoning engine.
4. BigEarthNet-19 domain taxonomy classification.
5. Observable, auditable execution trace generation.
6. End-to-end agentic orchestration pipeline.
"""

import unittest
from pathlib import Path
import numpy as np

from src.agent.tool_registry import ToolRegistry
from src.agent.orchestrator import SatQueryAgenticController
from src.analysis.bigearthnet_taxonomy import BigEarthNetTaxonomy, BIGEARTHNET_19_CLASSES
from src.analysis.sar_optical_fusion import OpticalSARFusionEngine
from src.analysis.cdvqa_engine import CDVQAEngine


class TestAgenticSatQuery(unittest.TestCase):

    def setUp(self):
        self.controller = SatQueryAgenticController()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def test_tool_registry_initialization(self):
        """Verify all 6 mandatory remote sensing specialist tools are registered."""
        tools = self.controller.registry.list_tools()
        self.assertEqual(len(tools), 6)
        tool_names = [t["name"] for t in tools]
        self.assertIn("SingleImageVQATool", tool_names)
        self.assertIn("RemoteSensingCaptionerTool", tool_names)
        self.assertIn("RegionGroundingTool", tool_names)
        self.assertIn("BiTemporalChangeVQATool", tool_names)
        self.assertIn("OpticalSARFusionTool", tool_names)
        self.assertIn("SpectralIndicesTool", tool_names)

    def test_bigearthnet_19_taxonomy(self):
        """Verify BigEarthNet-19 Corine Land Cover ontology classifications."""
        taxonomy = BigEarthNetTaxonomy()
        self.assertEqual(len(BIGEARTHNET_19_CLASSES), 19)

        preds = taxonomy.classify_from_spectral_metrics(
            ndvi_mean=0.55,
            water_pct=15.0,
            dense_veg_pct=45.0,
            mod_veg_pct=25.0,
            barren_pct=15.0,
            sar_backscatter_db=-7.5
        )
        self.assertGreater(len(preds), 0)
        class_names = [p["class_name"] for p in preds]
        # Should detect forest, water, and continuous urban fabric
        self.assertTrue(any("forest" in c.lower() for c in class_names))
        self.assertTrue(any("water" in c.lower() for c in class_names))
        self.assertTrue(any("urban" in c.lower() for c in class_names))

    def test_optical_sar_fusion(self):
        """Test cross-modal optical + SAR complementary feature extraction."""
        engine = OpticalSARFusionEngine()
        opt = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        sar = np.ones((100, 100), dtype=np.float32) * 0.6  # High double-bounce

        res = engine.fuse_optical_and_sar(opt, sar)
        self.assertEqual(res["task"], "optical_sar_analysis")
        self.assertIn("urban_percentage", res)
        self.assertIn("water_percentage", res)
        self.assertTrue(Path(res["fusion_image_path"]).exists())
        self.assertIn("Cross-Modal", res["answer"])

    def test_cdvqa_engine(self):
        """Test change-based visual question answering on bi-temporal pairs."""
        cdvqa = CDVQAEngine()
        t1 = np.zeros((100, 100, 3), dtype=np.float32)
        t2 = np.zeros((100, 100, 3), dtype=np.float32)
        # Add urban expansion in Northwest
        t2[:40, :40] = 0.9

        res = cdvqa.answer_change_query(
            t1, t2,
            "Has the built-up area increased, decreased, or remained unchanged?"
        )
        self.assertEqual(res["task"], "change_analysis")
        self.assertIn("Increased", res["answer"])
        self.assertIn("Northwest", res["primary_location"])
        self.assertTrue(Path(res["difference_map_path"]).exists())

    def test_auditable_execution_trace(self):
        """Verify the agent emits an observable, auditable execution trace with all SIH required fields."""
        test_img = "data/samples/sample_optical_t1.png"
        res = self.controller.execute_query(
            query="Describe the land-cover and major objects visible in this image.",
            primary_input=test_img
        )

        self.assertIn("execution_trace", res)
        trace = res["execution_trace"]
        self.assertIn("trace_id", trace)
        self.assertIn("selected_task", trace)
        self.assertIn("selected_tool", trace)
        self.assertIn("input_summary", trace)
        self.assertIn("parameters_configured", trace)
        self.assertIn("execution_time_ms", trace)
        self.assertIn("confidence", trace)
        self.assertEqual(trace["input_summary"]["compatibility_check"], "PASSED")


if __name__ == "__main__":
    unittest.main()
