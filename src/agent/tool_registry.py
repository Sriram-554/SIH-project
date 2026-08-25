"""
SatQuery - Agentic Tool Registry

Defines the formal registry of specialized remote-sensing AI tools,
input compatibility constraints, parameter validation, and tool execution interfaces.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path


class SpecialistTool:
    """Base specification for a SatQuery specialist remote-sensing tool."""

    def __init__(
        self,
        name: str,
        task_type: str,
        description: str,
        required_image_count: int,
        supported_modalities: List[str],
        supported_formats: List[str],
        executor: Callable
    ):
        self.name = name
        self.task_type = task_type
        self.description = description
        self.required_image_count = required_image_count
        self.supported_modalities = supported_modalities
        self.supported_formats = supported_formats
        self.executor = executor

    def validate_inputs(
        self,
        image_count: int,
        modalities: List[str],
        file_formats: List[str]
    ) -> Tuple[bool, str]:
        """Checks if input configuration satisfies tool constraints."""
        if image_count < self.required_image_count:
            return False, f"Requires at least {self.required_image_count} image(s), received {image_count}."

        for mod in modalities:
            if mod.lower() not in self.supported_modalities and "all" not in self.supported_modalities:
                return False, f"Modality '{mod}' not supported. Tool supports {self.supported_modalities}."

        for fmt in file_formats:
            if fmt.lower() not in self.supported_formats and "all" not in self.supported_formats:
                return False, f"Format '{fmt}' not supported. Tool supports {self.supported_formats}."

        return True, "Input configuration validated."

    def execute(self, **kwargs) -> Dict[str, Any]:
        return self.executor(**kwargs)


class ToolRegistry:
    """Predefined registry of remote sensing specialist tools."""

    def __init__(self):
        self.tools: Dict[str, SpecialistTool] = {}

    def register_tool(self, tool: SpecialistTool):
        self.tools[tool.task_type] = tool

    def get_tool(self, task_type: str) -> Optional[SpecialistTool]:
        return self.tools.get(task_type)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "task_type": t.task_type,
                "description": t.description,
                "required_images": t.required_image_count,
                "modalities": t.supported_modalities
            }
            for t in self.tools.values()
        ]
