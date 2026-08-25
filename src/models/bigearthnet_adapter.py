"""
SatQuery - BigEarthNet Domain Adaptation Model
A lightweight trainable classifier adapted to BigEarthNet-19 ontology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import json

BIGEARTHNET_19_CLASSES = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Arable land (annual crops)",
    "Permanent crops (vineyards, orchards, olive groves)",
    "Pastures and other agricultural areas",
    "Complex cultivation patterns",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland-shrub",
    "Beaches, dunes, sands",
    "Bare rocks and sparsely vegetated areas",
    "Inland wetlands (marshes, peatbogs)",
    "Coastal wetlands",
    "Inland waters (rivers, lakes, reservoirs)",
    "Marine waters"
]


class BigEarthNetAdapter(nn.Module):
    """Lightweight MLP classifier adapted for BigEarthNet-19 multi-label classification."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_classes: int = 19):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.classes = BIGEARTHNET_19_CLASSES

    def forward(self, x):
        return self.network(x)

    def predict(self, features: np.ndarray, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Predict multi-label BigEarthNet classes from spectral feature vector."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze().numpy()

        results = []
        for i, prob in enumerate(probs):
            if prob >= threshold:
                results.append({
                    "class_name": self.classes[i],
                    "confidence": round(float(prob), 3)
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)

        if not results:
            # fallback
            top_idx = int(np.argmax(probs))
            results.append({
                "class_name": self.classes[top_idx],
                "confidence": round(float(probs[top_idx]), 3)
            })

        return results


def extract_spectral_features(stats: Dict[str, Any]) -> np.ndarray:
    """Convert spectral statistics into a fixed feature vector for the adapter."""
    return np.array([
        stats.get("ndvi_mean", 0.3),
        stats.get("ndwi_mean", 0.0),
        stats.get("water_percentage", 5.0) / 100.0,
        stats.get("dense_vegetation_percentage", 30.0) / 100.0,
        stats.get("moderate_vegetation_percentage", 30.0) / 100.0,
        stats.get("barren_builtup_percentage", 20.0) / 100.0,
        stats.get("nbr_mean", 0.2),
        stats.get("brightness", 0.4)
    ], dtype=np.float32)


class BigEarthNetDomainAdapter:
    """High-level interface used by the agentic system."""

    def __init__(self, model_path: str = "models/bigearthnet_adapter.pt"):
        self.model = BigEarthNetAdapter()
        self.model_path = Path(model_path)
        self.device = torch.device("cpu")
        self.model.to(self.device)

        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.is_trained = True
        else:
            self.is_trained = False
            print("[BigEarthNetAdapter] No trained weights found. Using randomly initialized model.")

    def classify(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        features = extract_spectral_features(stats)
        return self.model.predict(features)

    def get_domain_prompt(self, predictions: List[Dict[str, Any]]) -> str:
        if not predictions:
            return "[BigEarthNet-19 Domain Adaptation]: No strong land-cover signals detected."

        top = ", ".join([f"{p['class_name']} ({p['confidence']*100:.0f}%)" for p in predictions[:4]])
        return (
            f"[BigEarthNet-19 Domain-Adapted Classification]:\n"
            f"Predicted CLC classes: {top}\n"
            f"Use standard ESA / ISRO remote sensing land-cover terminology."
        )

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[BigEarthNetAdapter] Model saved to {self.model_path}")