"""
SatQuery - Master Orchestration Pipeline

Integrates Sentinel-2 metadata extraction, multi-spectral index generation,
query intent classification, change detection, visual grounding, optical-SAR fusion,
and agentic tool orchestration into a unified system.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.orchestrator import SatQueryAgenticController
from src.analysis.sentinel_metadata import inspect_sentinel_product


class SatQueryPipeline:
    """Master pipeline coordinating all SatQuery remote sensing analysis tasks."""

    def __init__(
        self,
        output_dir: str = "outputs",
        hf_token: Optional[str] = None,
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the central Agentic Controller
        self.controller = SatQueryAgenticController(
            output_dir=str(self.output_dir),
            hf_token=hf_token,
            vlm_model=vlm_model
        )

    def process(
        self,
        query: str,
        primary_input: Union[str, Path],
        secondary_input: Optional[Union[str, Path]] = None,
        modalities: Optional[List[str]] = None,
        forced_task: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes end-to-end agentic analysis on the given satellite input(s) and query."""
        primary_path = Path(primary_input)

        # Inspect metadata if .SAFE product
        metadata = {}
        if primary_path.is_dir() and primary_path.suffix.upper() == ".SAFE":
            metadata = inspect_sentinel_product(primary_path) or {}

        # Delegate execution to agentic controller
        agent_result = self.controller.execute_query(
            query=query,
            primary_input=primary_input,
            secondary_input=secondary_input,
            modalities=modalities,
            forced_task=forced_task
        )

        # Attach product metadata and unified artifacts structure
        agent_result["metadata"] = metadata
        agent_result["artifacts"] = {
            "rgb_path": agent_result["visual_evidence"].get("rgb"),
            "ndvi_path": str(self.output_dir / "sentinel_ndvi.png") if (self.output_dir / "sentinel_ndvi.png").exists() else None,
            "ndwi_path": str(self.output_dir / "sentinel_ndwi.png") if (self.output_dir / "sentinel_ndwi.png").exists() else None,
            "false_color_path": str(self.output_dir / "sentinel_false_color.png") if (self.output_dir / "sentinel_false_color.png").exists() else None,
            "change_map": agent_result["visual_evidence"].get("difference_map"),
            "fusion_map": agent_result["visual_evidence"].get("fusion_map"),
            "grounding_map": agent_result["visual_evidence"].get("grounding_map")
        }

        return agent_result


if __name__ == "__main__":
    sample_safe = Path("data/S2B_MSIL2A_20230207T101109_N0510_R022_T33TUL_20240813T033135.SAFE")
    test_target = sample_safe if sample_safe.exists() else "outputs/sentinel_rgb.png"

    pipeline = SatQueryPipeline()
    output = pipeline.process(
        query="Describe the land-cover and major objects visible in this image.",
        primary_input=test_target
    )

    print("\n" + "=" * 60)
    print("PIPELINE AGENTIC EXECUTION OUTPUT")
    print("=" * 60)
    print("Task         :", output["task"])
    print("Selected Tool:", output["tool_name"])
    print("Confidence   :", output["confidence"])
    print("\nAnswer:\n", output["answer"])
    print("\nAuditable Execution Trace:")
    print(output["execution_trace"])
    print("=" * 60)
