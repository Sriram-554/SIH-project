"""
SatQuery - VQA Model Interface

Provides unified abstractions and implementations for local and remote
remote-sensing VQA backends.
"""

from typing import Dict, Any, Optional
from src.models.hf_vqa import HuggingFaceVQAModel
from src.models.spectral_vqa_fallback import SpectralFallbackEngine


class VQAModel:
    """Base interface for SatQuery VQA models."""

    def __init__(self, model_name: str = "base_vqa"):
        self.model_name = model_name

    def answer(
        self,
        image_path: str,
        question: str,
        spectral_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError("VQA backend has not been implemented.")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "task": "remote-sensing visual question answering",
            "status": "ready"
        }


class RemoteVQAModel(VQAModel):
    """Hosted cloud Vision-Language model with automatic fallback."""

    def __init__(self, token: Optional[str] = None, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__(model_name)
        self.backend = HuggingFaceVQAModel(token=token, preferred_model=model_name)

    def answer(
        self,
        image_path: str,
        question: str,
        spectral_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.backend.answer(image_path, question, spectral_context)


class LocalVQAModel(VQAModel):
    """Deterministic local spectral reasoning engine."""

    def __init__(self, model_name: str = "local_spectral_engine"):
        super().__init__(model_name)
        self.backend = SpectralFallbackEngine()

    def answer(
        self,
        image_path: str,
        question: str,
        spectral_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.backend.answer_query(question, stats=spectral_context)


if __name__ == "__main__":
    local_model = LocalVQAModel()
    res = local_model.answer("", "What is the vegetation state?", {"ndvi_mean": 0.52, "dense_vegetation_percentage": 60})
    print("Local Model Result:", res)