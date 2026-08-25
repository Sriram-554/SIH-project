"""
SatQuery - Multimodal Vision-Language Inference Engine

Communicates with Hugging Face's OpenAI-compatible Inference Router
with multi-model fallback cascade, context-enriched remote-sensing prompts,
and automatic failover to the local deterministic SpectralFallbackEngine.
"""

import sys
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from src.models.spectral_vqa_fallback import SpectralFallbackEngine


# Candidate models supported by HF Serverless Inference Providers
CANDIDATE_VLM_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]


class HuggingFaceVQAModel:
    """Multimodal Vision-Language Model interface for SatQuery."""

    def __init__(
        self,
        token: Optional[str] = None,
        preferred_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        model_name: Optional[str] = None
    ):
        # Support model_name alias
        chosen_model = model_name or preferred_model

        # Check token from arg, env, or .env file
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            env_file = PROJECT_ROOT / ".env"
            if env_file.exists():
                with open(env_file, "r") as f:
                    for line in f:
                        if line.startswith("HF_TOKEN="):
                            self.token = line.strip().split("=", 1)[1].strip("\"' ")

        self.preferred_model = chosen_model
        self.fallback_engine = SpectralFallbackEngine()

        if self.token:
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.token,
                timeout=15.0
            )
        else:
            self.client = None

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encodes an image file to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def answer(
        self,
        image_path: str,
        question: str,
        spectral_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Answers a visual question using cloud VLM or local spectral fallback."""
        if not Path(image_path).exists():
            return {
                "answer": f"Error: Specified image file was not found: {image_path}",
                "confidence": 0.0,
                "evidence": None,
                "model": "none",
                "status": "error_file_not_found"
            }

        # If no client/token available, use deterministic spectral reasoning
        if not self.client or not self.token:
            print("[SatQuery] No HF_TOKEN detected. Using deterministic Spectral Reasoner.")
            return self.fallback_engine.answer_query(question, stats=spectral_context)

        encoded_image = self._encode_image(image_path)
        ext = Path(image_path).suffix.lower()
        _mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                     ".tif": "image/tiff", ".tiff": "image/tiff", ".jp2": "image/jp2"}
        mime_type = _mime_map.get(ext, "image/png")

        # Build context-enriched prompt for remote sensing
        context_str = ""
        if spectral_context:
            context_str = (
                f"\n[Ground Truth Sensor Telemetry]:\n"
                f"- Landscape Mean NDVI: {spectral_context.get('ndvi_mean', 0.0):.3f}\n"
                f"- Dense Canopy Coverage: {spectral_context.get('dense_vegetation_percentage', 0.0)}%\n"
                f"- Open Water Coverage: {spectral_context.get('water_percentage', 0.0)}%\n"
                f"- Barren/Built-up Area: {spectral_context.get('barren_builtup_percentage', 0.0)}%\n"
            )

        system_instruction = (
            "You are SatQuery AI, an expert Earth Observation and Remote Sensing intelligence system. "
            "Analyze the provided satellite image with high precision. Incorporate any physical sensor metrics "
            "provided and deliver concise, scientifically grounded answers."
        )

        prompt_text = f"{context_str}\nQuestion: {question}"

        # Try models in order: preferred model first, then up to 2 fallback candidates
        candidate_fallbacks = [m for m in CANDIDATE_VLM_MODELS if m != self.preferred_model][:2]
        models_to_try = [self.preferred_model] + candidate_fallbacks

        last_error = None
        for candidate_model in models_to_try:
            try:
                print(f"[SatQuery] Querying VLM model: {candidate_model}...")
                response = self.client.chat.completions.create(
                    model=candidate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_instruction
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{encoded_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=350,
                    temperature=0.2
                )

                answer_text = response.choices[0].message.content
                print(f"[SatQuery] Received answer successfully from {candidate_model}.")

                return {
                    "answer": answer_text,
                    "confidence": 0.92,
                    "evidence": "Sentinel-2 Multi-Spectral Composite + VLM Analysis",
                    "model": candidate_model,
                    "status": "success"
                }

            except Exception as e:
                print(f"[SatQuery] Model {candidate_model} failed: {e}")
                last_error = str(e)
                continue

        # If all cloud models fail, fallback gracefully to spectral engine
        print("[SatQuery] Cloud VLM models unavailable. Failing over to Spectral Engine.")
        fallback_res = self.fallback_engine.answer_query(question, stats=spectral_context)
        fallback_res["note"] = f"Cloud VLM unavailable ({last_error}). Answered via local spectral analysis."
        return fallback_res


if __name__ == "__main__":
    test_img = "outputs/sentinel_rgb.png"
    q = "What type of land cover is visible in this satellite image?"

    print("=" * 60)
    print("SATQUERY - VQA INFERENCE TEST")
    print("=" * 60)
    print("Image:", test_img)
    print("Question:", q)

    model = HuggingFaceVQAModel()
    res = model.answer(test_img, q, spectral_context={"ndvi_mean": 0.42, "water_percentage": 3.1})
    print("\nModel:", res["model"])
    print("Answer:\n", res["answer"])
    print("Status:", res["status"])
    print("=" * 60)