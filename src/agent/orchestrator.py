"""SatQuery agentic orchestration controller."""

import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.agent.tool_registry import ToolRegistry, SpecialistTool
from src.analysis.query_router import classify_query
from src.analysis.spectral_indices import SpectralEngine
from src.analysis.grounding import VisualGrounder
from src.analysis.change_detector import ChangeDetector
from src.analysis.cdvqa_engine import CDVQAEngine
from src.analysis.sar_optical_fusion import OpticalSARFusionEngine
from src.analysis.bigearthnet_taxonomy import BigEarthNetTaxonomy
from src.models.vqa_model import RemoteVQAModel


class SatQueryAgenticController:
    """Central controller for query routing, specialist execution and trace generation."""

    def __init__(self, output_dir: str = "outputs", hf_token: Optional[str] = None,
                 vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spectral_engine = SpectralEngine()
        self.grounder = VisualGrounder(output_dir=str(self.output_dir))
        self.change_detector = ChangeDetector(output_dir=str(self.output_dir))
        self.cdvqa_engine = CDVQAEngine(output_dir=str(self.output_dir))
        self.fusion_engine = OpticalSARFusionEngine(output_dir=str(self.output_dir))
        self.taxonomy = BigEarthNetTaxonomy()
        self.remote_vqa = RemoteVQAModel(token=hf_token, model_name=vlm_model)
        self.registry = ToolRegistry()
        self._register_specialist_tools()

    def _register_specialist_tools(self):
        common_formats = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".safe", ".jp2"]
        self.registry.register_tool(SpecialistTool(
            name="SingleImageVQATool", task_type="vqa",
            description="Performs visual question answering on satellite imagery.",
            required_image_count=1, supported_modalities=["optical", "sar", "multispectral"],
            supported_formats=common_formats, executor=self._exec_vqa))
        self.registry.register_tool(SpecialistTool(
            name="RemoteSensingCaptionerTool", task_type="captioning",
            description="Generates remote-sensing scene descriptions with domain taxonomy context.",
            required_image_count=1, supported_modalities=["optical", "sar", "multispectral"],
            supported_formats=common_formats, executor=self._exec_captioning))
        self.registry.register_tool(SpecialistTool(
            name="RegionGroundingTool", task_type="region_grounding",
            description="Creates spectral/RGB masks and approximate spatial regions.",
            required_image_count=1, supported_modalities=["optical", "multispectral"],
            supported_formats=common_formats, executor=self._exec_grounding))
        self.registry.register_tool(SpecialistTool(
            name="BiTemporalChangeVQATool", task_type="change_analysis",
            description="Analyzes two observations with image-difference and spectral heuristics.",
            required_image_count=2, supported_modalities=["optical", "sar", "multispectral"],
            supported_formats=common_formats, executor=self._exec_change))
        self.registry.register_tool(SpecialistTool(
            name="OpticalSARFusionTool", task_type="optical_sar_analysis",
            description="Combines optical imagery with supplied or synthetic SAR-like structural information.",
            required_image_count=1, supported_modalities=["optical", "sar", "cross_modal"],
            supported_formats=common_formats, executor=self._exec_fusion))
        self.registry.register_tool(SpecialistTool(
            name="SpectralIndicesTool", task_type="spectral_analysis",
            description="Computes NDVI, NDWI, NBR and zonal environmental statistics.",
            required_image_count=1, supported_modalities=["optical", "multispectral"],
            supported_formats=[".safe", ".tif", ".tiff", ".jp2", ".png", ".jpg"],
            executor=self._exec_spectral))

    def _taxonomy_predictions(self, stats: Dict[str, Any]) -> list:
        return self.taxonomy.classify_from_spectral_metrics(
            ndvi_mean=stats.get("ndvi_mean", 0.0),
            water_pct=stats.get("water_percentage", 0.0),
            dense_veg_pct=stats.get("dense_vegetation_percentage", 0.0),
            mod_veg_pct=stats.get("moderate_vegetation_percentage", 0.0),
            barren_pct=stats.get("barren_builtup_percentage", 0.0))

    def _exec_vqa(self, query, primary_img, stats, **kwargs):
        preds = self._taxonomy_predictions(stats)
        domain_prompt = self.taxonomy.get_domain_prompt_context(preds)
        result = self.remote_vqa.answer(primary_img, f"{query}\n\n{domain_prompt}", spectral_context=stats)
        result["bigearthnet_predictions"] = preds
        return result

    def _exec_captioning(self, query, primary_img, stats, **kwargs):
        preds = self._taxonomy_predictions(stats)
        domain_prompt = self.taxonomy.get_domain_prompt_context(preds)
        result = self.remote_vqa.answer(primary_img, f"{query}\n\n{domain_prompt}", spectral_context=stats)
        result["bigearthnet_predictions"] = preds
        return result

    def _exec_grounding(self, query, primary_img, ndvi_arr, ndwi_arr, **kwargs):
        q = query.lower()
        target = "water" if any(x in q for x in ["water", "lake", "river", "pond", "wetland"]) else "vegetation" if any(x in q for x in ["vegetation", "forest", "crop", "canopy"]) else "urban"
        return self.grounder.ground_feature(primary_img, target_feature=target, ndvi=ndvi_arr, ndwi=ndwi_arr)

    def _exec_change(self, query, primary_img, secondary_img, ndvi_t1, ndvi_t2, **kwargs):
        return self.cdvqa_engine.answer_change_query(primary_img, secondary_img, query, ndvi_t1=ndvi_t1, ndvi_t2=ndvi_t2)

    def _exec_fusion(self, query, primary_img, secondary_img=None, **kwargs):
        return self.fusion_engine.fuse_optical_and_sar(primary_img, secondary_img, query=query)

    def _exec_spectral(self, stats, **kwargs):
        return {"answer": (
            "**Multispectral Remote Sensing Analysis**:\n"
            f"- **Mean NDVI**: {stats.get('ndvi_mean', 0.0):.3f}\n"
            f"- **Dominant Land Cover**: {stats.get('dominant_land_cover', 'Unavailable')}\n"
            f"- **Dense Canopy**: {stats.get('dense_vegetation_percentage', 0.0):.1f}%\n"
            f"- **Moderate Vegetation**: {stats.get('moderate_vegetation_percentage', 0.0):.1f}%\n"
            f"- **Water**: {stats.get('water_percentage', 0.0):.1f}%\n"
            f"- **Barren / Urban**: {stats.get('barren_builtup_percentage', 0.0):.1f}%"),
            "confidence": 0.98, "status": "success"}

    def _query_requests_visual(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ["show", "display", "visual", "image", "picture", "map", "plot", "highlight", "overlay", "bbox", "bounding box", "locate", "ground", "where is", "generate image", "generate map"])

    def _generate_visual_evidence_for_query(self, query, primary_img, secondary_img, ndvi_arr, ndwi_arr):
        q = query.lower()
        if secondary_img and any(k in q for k in ["change", "changed", "before and after", "difference", "increased", "decreased", "growth", "loss"]):
            out = self.cdvqa_engine.answer_change_query(primary_img, secondary_img, query, ndvi_t1=ndvi_arr, ndvi_t2=None)
            return {"difference_map_path": out.get("difference_map_path")}
        if secondary_img and any(k in q for k in ["sar", "radar", "fusion", "joint", "cross-modal"]):
            out = self.fusion_engine.fuse_optical_and_sar(primary_img, secondary_img, query=query)
            return {"fusion_image_path": out.get("fusion_image_path")}
        target = "water" if any(k in q for k in ["water", "lake", "river", "pond", "wetland"]) else "vegetation" if any(k in q for k in ["vegetation", "forest", "crop", "canopy"]) else "urban"
        out = self.grounder.ground_feature(primary_img, target_feature=target, ndvi=ndvi_arr, ndwi=ndwi_arr)
        return {"grounded_image_path": out.get("grounded_image_path")}

    @staticmethod
    def _basic_image_stats(path: Path) -> Dict[str, Any]:
        """Return image-independent-safe metadata for ordinary RGB images.

        RGB PNG/JPG files do not contain the spectral bands required to compute
        NDVI/NDWI/NBR. Do not fabricate spectral statistics for them.
        """
        return {
            "spectral_data_available": False,
            "note": "RGB image supplied; NDVI/NDWI/NBR require compatible multispectral bands.",
        }

    def execute_query(self, query: str, primary_input: Union[str, Path], secondary_input=None,
                      modalities: Optional[List[str]] = None, forced_task: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        trace_id = f"SATQ-{str(uuid.uuid4())[:8].upper()}"
        p_path = Path(primary_input)
        s_path = Path(secondary_input) if secondary_input else None
        input_files = [p_path] + ([s_path] if s_path else [])
        image_count = len(input_files)
        detected_modalities = modalities or (["optical"] * image_count)
        detected_formats = [f.suffix.lower() if f.suffix else ".safe" for f in input_files]

        stats: Dict[str, Any] = {}
        ndvi_arr = ndwi_arr = None
        rgb_image_path = str(p_path)
        if p_path.is_dir() and p_path.suffix.upper() == ".SAFE":
            spec_data = self.spectral_engine.process_safe_product(p_path, self.output_dir)
            rgb_image_path = spec_data.get("rgb_path", str(p_path))
            ndvi_arr, ndwi_arr = spec_data.get("ndvi"), spec_data.get("ndwi")
            stats = spec_data.get("statistics", {})
        elif p_path.exists() and p_path.is_file():
            stats = self._basic_image_stats(p_path)
        elif Path("outputs/sentinel_rgb.png").exists():
            rgb_image_path = "outputs/sentinel_rgb.png"
            stats = self._basic_image_stats(Path(rgb_image_path))

        selected_task = forced_task or classify_query(query, number_of_images=image_count, modalities=detected_modalities)
        tool = self.registry.get_tool(selected_task) or self.registry.get_tool("vqa")
        if tool is None:
            raise RuntimeError("SatQuery tool registry has no VQA fallback tool")
        selected_task = tool.task_type
        valid_input, validation_msg = tool.validate_inputs(image_count=image_count, modalities=detected_modalities, file_formats=detected_formats)

        params = {"query": query, "primary_img": rgb_image_path, "secondary_img": str(s_path) if s_path else None,
                  "stats": stats, "ndvi_arr": ndvi_arr, "ndwi_arr": ndwi_arr,
                  "ndvi_t1": ndvi_arr, "ndvi_t2": None}
        tool_output = tool.execute(**params)

        if self._query_requests_visual(query):
            for key, value in self._generate_visual_evidence_for_query(query, rgb_image_path, str(s_path) if s_path else None, ndvi_arr, ndwi_arr).items():
                if value:
                    tool_output[key] = value

        execution_trace = {
            "trace_id": trace_id, "query": query, "selected_task": selected_task,
            "selected_tool": tool.name, "tool_description": tool.description,
            "input_summary": {"image_count": image_count, "modalities": detected_modalities, "formats": detected_formats,
                              "compatibility_check": "PASSED" if valid_input else "WARNING", "validation_message": validation_msg},
            "parameters_configured": {"spectral_telemetry_injected": bool(stats.get("spectral_data_available", True)) if stats else False,
                                       "domain_adaptation_applied": "BigEarthNet-19 context" if selected_task in ["vqa", "captioning"] else "Not applicable"},
            "execution_time_ms": round((time.time() - start_time) * 1000, 1),
            "confidence": tool_output.get("confidence", 0.94), "status": tool_output.get("status", "completed_success")}

        return {"query": query, "answer": tool_output.get("answer", tool_output.get("summary", "Analysis completed.")),
                "task": selected_task, "tool_name": tool.name, "confidence": tool_output.get("confidence", 0.94),
                "visual_evidence": {"rgb": rgb_image_path, "difference_map": tool_output.get("difference_map_path"),
                                    "fusion_map": tool_output.get("fusion_image_path"), "grounding_map": tool_output.get("grounded_image_path")},
                "bigearthnet_classes": tool_output.get("bigearthnet_predictions", []), "statistics": stats,
                "execution_trace": execution_trace}


if __name__ == "__main__":
    controller = SatQueryAgenticController()
    test_img = "outputs/sentinel_rgb.png"
    if Path(test_img).exists():
        res = controller.execute_query("Describe the land-cover and major objects visible in this image.", test_img)
        print(res["answer"])
        print(res["execution_trace"])
