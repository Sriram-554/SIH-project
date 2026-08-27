"""Deterministic spectral VQA fallback for SatQuery."""

from typing import Dict, Any, Optional


class SpectralFallbackEngine:
    """Answers supported questions from supplied remote-sensing statistics.

    This engine deliberately does not invent spectral measurements when an RGB
    image has no multispectral bands.
    """

    def answer_query(self, question: str, stats: Optional[Dict[str, Any]] = None,
                     bands_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        q_lower = question.lower()
        stats = stats or {}

        if stats.get("spectral_data_available") is False:
            answer = (
                "The local spectral reasoner cannot compute NDVI/NDWI/NBR from this input because "
                "it contains an ordinary RGB image rather than the required multispectral bands. "
                "Provide a compatible Sentinel-2/multispectral product or use the cloud VLM for "
                "general visual questions."
            )
            return {
                "answer": answer,
                "confidence": 0.99,
                "evidence": stats.get("note", "Required multispectral bands are unavailable."),
                "model": "SatQuery-Deterministic-Spectral-Engine",
                "status": "limited_input_spectral_fallback",
            }

        if not stats:
            return {
                "answer": "No spectral measurements were supplied, so the local spectral reasoner cannot produce a grounded answer.",
                "confidence": 0.99,
                "evidence": "No spectral statistics available.",
                "model": "SatQuery-Deterministic-Spectral-Engine",
                "status": "insufficient_spectral_data",
            }

        ndvi_m = float(stats.get("ndvi_mean", 0.0))
        water_pct = float(stats.get("water_percentage", 0.0))
        dense_pct = float(stats.get("dense_vegetation_percentage", 0.0))
        mod_pct = float(stats.get("moderate_vegetation_percentage", 0.0))
        barren_pct = float(stats.get("barren_builtup_percentage", 0.0))
        dominant = stats.get("dominant_land_cover", "Mixed Landscape")

        if any(w in q_lower for w in ["land cover", "what type", "describe", "see", "what is visible", "scene"]):
            answer = (
                f"Based on multispectral analysis, the dominant land cover is **{dominant}**.\n"
                f"- **Dense Vegetation / Canopy**: {dense_pct:.1f}%\n"
                f"- **Moderate / Agricultural Vegetation**: {mod_pct:.1f}%\n"
                f"- **Barren / Urban / Built-up**: {barren_pct:.1f}%\n"
                f"- **Water Bodies**: {water_pct:.1f}%\n"
                f"- **Mean NDVI**: {ndvi_m:.3f}"
            )
        elif any(w in q_lower for w in ["vegetation", "forest", "crop", "agriculture", "green", "ndvi"]):
            health = "vigorous and dense" if ndvi_m > 0.45 else "moderately active" if ndvi_m > 0.25 else "sparse/dormant"
            answer = (f"Vegetation vigor is **{health}** with mean NDVI **{ndvi_m:.3f}**. "
                      f"Dense vegetation covers {dense_pct:.1f}% and moderate vegetation {mod_pct:.1f}%.")
        elif any(w in q_lower for w in ["water", "river", "lake", "ocean", "wetland", "ndwi"]):
            answer = (f"Surface-water analysis indicates approximately **{water_pct:.1f}%** water coverage."
                      if water_pct > 1.0 else f"Water detection indicates minimal open surface water (< {water_pct:.1f}%).")
        elif any(w in q_lower for w in ["urban", "city", "building", "built-up", "settlement", "road", "barren"]):
            answer = f"Barren/built-up candidate surfaces account for approximately **{barren_pct:.1f}%** of the tile area."
        elif any(w in q_lower for w in ["change", "difference", "delta"]):
            answer = f"Bi-temporal comparison is required for change analysis; the current observation has mean NDVI {ndvi_m:.3f}."
        else:
            answer = f"The multispectral scene is classified as **{dominant}** with mean NDVI **{ndvi_m:.3f}**."

        return {
            "answer": answer,
            "confidence": 0.95,
            "evidence": f"Supplied spectral indices (NDVI={ndvi_m:.3f}, Water={water_pct:.1f}%)",
            "model": "SatQuery-Deterministic-Spectral-Engine",
            "status": "success_spectral_fallback",
        }
