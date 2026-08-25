"""
SatQuery - BigEarthNet Domain Adaptation Layer
Uses the trained BigEarthNetAdapter model.
"""

from typing import Dict, Any, List, Optional
from src.models.bigearthnet_adapter import BigEarthNetDomainAdapter

class BigEarthNetTaxonomy:
    def __init__(self):
        self.adapter = BigEarthNetDomainAdapter()

    def classify_from_spectral_metrics(
        self,
        ndvi_mean: float = 0.35,
        water_pct: float = 5.0,
        dense_veg_pct: float = 30.0,
        mod_veg_pct: float = 30.0,
        barren_pct: float = 20.0,
        sar_backscatter_db: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        stats = {
            "ndvi_mean": ndvi_mean,
            "water_percentage": water_pct,
            "dense_vegetation_percentage": dense_veg_pct,
            "moderate_vegetation_percentage": mod_veg_pct,
            "barren_builtup_percentage": barren_pct,
            "ndwi_mean": kwargs.get("ndwi_mean", 0.0),
            "nbr_mean": kwargs.get("nbr_mean", 0.2),
            "brightness": kwargs.get("brightness", 0.4)
        }
        return self.adapter.classify(stats)

    def get_domain_prompt_context(self, predictions: List[Dict[str, Any]]) -> str:
        return self.adapter.get_domain_prompt(predictions)