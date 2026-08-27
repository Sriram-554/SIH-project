"""
SatQuery - BigEarthNet Domain Adaptation Layer
Uses the trained BigEarthNetAdapter model and the latest real spectral statistics.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from src.models.bigearthnet_adapter import BigEarthNetDomainAdapter


class BigEarthNetTaxonomy:
    def __init__(self, stats_path: str = "outputs/spectral_stats.json"):
        self.adapter = BigEarthNetDomainAdapter()
        self.stats_path = Path(stats_path)

    def _load_latest_spectral_stats(self) -> Optional[Dict[str, Any]]:
        """Load statistics produced by SpectralEngine for the latest SAFE product."""
        if not self.stats_path.exists():
            return None
        try:
            payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
            stats = payload.get("statistics")
            if isinstance(stats, dict) and stats.get("ndvi_mean") is not None:
                return stats
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return None

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
        """
        Classify using the latest scene-derived statistics when available.

        The previous implementation accepted caller-supplied placeholder values.
        For the Streamlit benchmark workflow, the SpectralEngine now persists the
        measured statistics to outputs/spectral_stats.json. Those measurements are
        preferred so the BigEarthNet prediction is tied to the actual Sentinel-2 scene.
        """
        latest = self._load_latest_spectral_stats()

        if latest is not None:
            stats = dict(latest)
            # Preserve optional caller telemetry only when the latest scene does not
            # provide that field. Do not overwrite measured spectral statistics.
            if stats.get("nbr_mean") is None and kwargs.get("nbr_mean") is not None:
                stats["nbr_mean"] = kwargs["nbr_mean"]
            if stats.get("brightness") is None and kwargs.get("brightness") is not None:
                stats["brightness"] = kwargs["brightness"]
        else:
            # Backward-compatible path for callers that already provide real metrics.
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

    def get_latest_spectral_stats(self) -> Optional[Dict[str, Any]]:
        """Expose the latest measured statistics for UI/reporting layers."""
        return self._load_latest_spectral_stats()

    def get_domain_prompt_context(self, predictions: List[Dict[str, Any]]) -> str:
        return self.adapter.get_domain_prompt(predictions)
