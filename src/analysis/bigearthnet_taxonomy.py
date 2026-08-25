"""
SatQuery - BigEarthNet-19 Domain Adaptation & Taxonomy Engine

Implements the official BigEarthNet-19 Corine Land Cover (CLC) remote-sensing
ontology for domain-adapted visual question answering, multi-label scene
classification, and sensor feature mapping.
"""

from typing import Dict, Any, List, Optional
import numpy as np


# Official 19 BigEarthNet Corine Land Cover (CLC) Classes
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


class BigEarthNetTaxonomy:
    """Adapts remote sensing multi-spectral and SAR features to the BigEarthNet-19 ontology."""

    def __init__(self):
        self.classes = BIGEARTHNET_19_CLASSES

    def classify_from_spectral_metrics(
        self,
        ndvi_mean: float,
        water_pct: float,
        dense_veg_pct: float,
        mod_veg_pct: float,
        barren_pct: float,
        sar_backscatter_db: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Predicts multi-label BigEarthNet-19 classes based on multi-spectral and SAR physical metrics."""
        scores = {}

        # 1. Water bodies
        if water_pct > 2.0:
            scores["Inland waters (rivers, lakes, reservoirs)"] = min(1.0, water_pct / 40.0)
        if water_pct > 30.0:
            scores["Marine waters"] = min(1.0, water_pct / 60.0)

        # 2. Forests & Canopy
        if dense_veg_pct > 15.0:
            scores["Broad-leaved forest"] = min(1.0, dense_veg_pct / 50.0)
            scores["Mixed forest"] = min(1.0, dense_veg_pct / 60.0)
        if dense_veg_pct > 30.0:
            scores["Coniferous forest"] = min(1.0, (dense_veg_pct - 20.0) / 40.0)

        # 3. Agriculture & Cultivation
        if mod_veg_pct > 10.0:
            scores["Arable land (annual crops)"] = min(1.0, mod_veg_pct / 40.0)
            scores["Complex cultivation patterns"] = min(1.0, mod_veg_pct / 50.0)
            scores["Pastures and other agricultural areas"] = min(1.0, mod_veg_pct / 60.0)

        # 4. Urban & Built-up
        if barren_pct > 10.0:
            if sar_backscatter_db is not None and sar_backscatter_db > -10.0:
                # High SAR double-bounce indicates dense urban structures
                scores["Continuous urban fabric"] = min(1.0, barren_pct / 30.0)
                scores["Industrial or commercial units"] = min(1.0, barren_pct / 35.0)
            else:
                scores["Discontinuous urban fabric"] = min(1.0, barren_pct / 40.0)
                scores["Bare rocks and sparsely vegetated areas"] = min(1.0, barren_pct / 50.0)

        # 5. Natural grassland & transitional
        if 0.2 <= ndvi_mean < 0.4:
            scores["Natural grassland and sparsely vegetated areas"] = 0.75
            scores["Transitional woodland-shrub"] = 0.65

        # Format sorted predictions
        results = [
            {"class_name": cls, "confidence": round(float(conf), 2)}
            for cls, conf in scores.items() if conf >= 0.25
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)

        if not results:
            results.append({"class_name": "Complex cultivation patterns", "confidence": 0.50})

        return results

    def get_domain_prompt_context(self, predictions: List[Dict[str, Any]]) -> str:
        """Constructs BigEarthNet-adapted ontology context for VLM system prompts."""
        class_list_str = ", ".join([f"{p['class_name']} ({int(p['confidence']*100)}%)" for p in predictions[:4]])
        return (
            f"[BigEarthNet-19 Remote-Sensing Domain Grounding]:\n"
            f"Detected CLC Classes: {class_list_str}\n"
            f"Please utilize standard European Space Agency (ESA) and ISRO remote-sensing taxonomy."
        )


if __name__ == "__main__":
    taxonomy = BigEarthNetTaxonomy()
    preds = taxonomy.classify_from_spectral_metrics(
        ndvi_mean=0.45,
        water_pct=5.0,
        dense_veg_pct=42.0,
        mod_veg_pct=35.0,
        barren_pct=18.0,
        sar_backscatter_db=-8.5
    )
    print("BigEarthNet-19 Multi-Label Predictions:")
    for p in preds:
        print(f" - {p['class_name']}: {p['confidence']}")
