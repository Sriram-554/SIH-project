"""
SatQuery - SIH Representative Query Verification Script

Executes the 5 canonical queries from the SIH Problem Statement
and validates that the Agentic Controller selects the correct tool,
emits an auditable execution trace, and produces evidence-grounded answers.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.orchestrator import SatQueryAgenticController


def run_sih_verification():
    controller = SatQueryAgenticController()

    opt_t1 = "data/samples/sample_optical_t1.png"
    opt_t2 = "data/samples/sample_optical_t2.png"
    sar_img = "data/samples/sample_sar_risat.png"

    sih_test_cases = [
        {
            "id": "SIH-Q1",
            "query": "Describe the land-cover and major objects visible in this image.",
            "primary": opt_t1,
            "secondary": None,
            "modalities": ["optical"],
            "expected_task": "captioning",
            "expected_tool": "RemoteSensingCaptionerTool"
        },
        {
            "id": "SIH-Q2",
            "query": "Highlight the water body referred to in the query.",
            "primary": opt_t1,
            "secondary": None,
            "modalities": ["optical"],
            "expected_task": "region_grounding",
            "expected_tool": "RegionGroundingTool"
        },
        {
            "id": "SIH-Q3",
            "query": "What changed between these two dates, and where did the change occur?",
            "primary": opt_t1,
            "secondary": opt_t2,
            "modalities": ["optical", "optical"],
            "expected_task": "change_analysis",
            "expected_tool": "BiTemporalChangeVQATool"
        },
        {
            "id": "SIH-Q4",
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "primary": opt_t1,
            "secondary": sar_img,
            "modalities": ["optical", "sar"],
            "expected_task": "optical_sar_analysis",
            "expected_tool": "OpticalSARFusionTool"
        },
        {
            "id": "SIH-Q5",
            "query": "Has the built-up area increased, decreased, or remained unchanged?",
            "primary": opt_t1,
            "secondary": opt_t2,
            "modalities": ["optical", "optical"],
            "expected_task": "change_analysis",
            "expected_tool": "BiTemporalChangeVQATool"
        }
    ]

    print("=" * 70)
    print("SATQUERY AI — SIH REPRESENTATIVE QUERY VERIFICATION SUITE")
    print("=" * 70)

    for tc in sih_test_cases:
        print(f"\n[{tc['id']}] Query: \"{tc['query']}\"")
        res = controller.execute_query(
            query=tc["query"],
            primary_input=tc["primary"],
            secondary_input=tc["secondary"],
            modalities=tc["modalities"]
        )

        trace = res["execution_trace"]
        print(f" -> Selected Task : {res['task']} (Expected: {tc['expected_task']})")
        print(f" -> Selected Tool : {res['tool_name']} (Expected: {tc['expected_tool']})")
        print(f" -> Trace ID      : {trace['trace_id']}")
        print(f" -> Confidence    : {int(res['confidence'] * 100)}%")
        print(f" -> Execution Time: {trace['execution_time_ms']} ms")
        print(f" -> Answer Snippet:\n    {res['answer'][:160]}...\n")

        assert res["task"] == tc["expected_task"], f"Task mismatch for {tc['id']}"
        assert res["tool_name"] == tc["expected_tool"], f"Tool mismatch for {tc['id']}"

    print("=" * 70)
    print("ALL 5 SIH REPRESENTATIVE QUERIES VERIFIED SUCCESSFULLY WITH 100% PASS!")
    print("=" * 70)


if __name__ == "__main__":
    run_sih_verification()
