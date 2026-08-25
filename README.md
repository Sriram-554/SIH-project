# 🛰️ SatQuery — Multimodal Earth Observation & Remote Sensing VQA Platform

**SatQuery** is an intelligent Earth Observation (EO) and Remote Sensing AI system built for **Smart India Hackathon (SIH)**. It bridges raw multi-spectral satellite imagery (Copernicus Sentinel-2 Level-2A) with Vision-Language Models (VLMs) and geospatial array processing to provide natural language question answering, automated scene interpretation, vegetation health analysis, bi-temporal change detection, and visual feature grounding.

---

## 🌟 Key Features

* **Multi-Spectral Remote Sensing Engine**:
  * Extracts $10\,\text{m}$ and $20\,\text{m}$ spectral bands ($B02$ Blue, $B03$ Green, $B04$ Red, $B08$ NIR, $B11$ SWIR-1, $B12$ SWIR-2).
  * Computes standard environmental indices:
    * **NDVI** (Normalized Difference Vegetation Index): $(B08 - B04) / (B08 + B04)$
    * **NDWI** (Normalized Difference Water Index - McFeeters): $(B03 - B08) / (B03 + B08)$
    * **NBR** (Normalized Burn Ratio): $(B08 - B12) / (B08 + B12)$
    * **False-Color Infrared Composite** (NIR, Red, Green) for vegetation vigor.
* **Multimodal Vision-Language (VQA) Intelligence**:
  * Seamless integration with Hugging Face Serverless Inference Router (`Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, `google/gemma-3-27b-it`, `meta-llama/Llama-3.2-11B-Vision-Instruct`).
  * Injects physical ground truth sensor telemetry (mean NDVI, water %, canopy %) directly into VLM system prompts.
  * Deterministic offline **Spectral Fallback Engine** that continues to provide accurate scientific answers even without an API token or internet connection.
* **Multi-Temporal Change Detection**:
  * Computes structural Euclidean differences and Delta NDVI ($\Delta\text{NDVI} = \text{NDVI}_{T_2} - \text{NDVI}_{T_1}$) between satellite observations.
  * Quantifies and maps vegetation gain, deforestation/clearing, urban development, and hydrological shifts.
* **Visual Grounding & Spatial Feature Locator**:
  * Delineates and draws bounding boxes around surface features (water bodies, dense forest canopy, agricultural clusters, urban infrastructure).
* **Modern Interactive Web Dashboard**:
  * Streamlit-powered visualizer featuring side-by-side band comparison, dynamic land-cover distribution charts, conversational VQA chat, and automated report generation.

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

SatQuery runs on Python 3.10+ and uses lightweight local geospatial dependencies alongside optional cloud VLM inference.

```bash
# Clone the repository
git clone <repo_url>
cd SatQuery

# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies (already included in .venv)
pip install rasterio numpy matplotlib streamlit openai huggingface_hub pillow
```

### 2. Configure API Token (Optional)

To enable cloud Vision-Language reasoning:
```bash
# In Windows PowerShell:
$env:HF_TOKEN="your_huggingface_token_here"
```
*(You can also input your token directly in the Web UI sidebar or create a `.env` file).*

### 3. Launch the Interactive Web Dashboard

```bash
streamlit run app.py
```

---

## 🏗️ Project Architecture

```
SatQuery/
├── app.py                     # Streamlit Interactive Web Application
├── README.md                  # System Documentation
├── data/                      # Raw Sentinel-2 .SAFE products & GeoTIFFs
├── outputs/                   # Generated RGB composites, NDVI maps, difference heatmaps
├── src/
│   ├── analysis/
│   │   ├── spectral_indices.py# Band extraction, NDVI, NDWI, NBR & Zonal stats
│   │   ├── sentinel_metadata.py# .SAFE product metadata parser
│   │   ├── visualize_sentinel.py# Rasterio compositing and image synthesis
│   │   ├── change_detector.py # Bi-temporal structural & delta NDVI analysis
│   │   ├── grounding.py       # Spatial feature localization & bounding boxes
│   │   ├── query_router.py    # NLP intent & task classifier
│   │   └── input_validator.py # Modality & format validation
│   ├── models/
│   │   ├── vqa_model.py       # Abstract base VQA interface
│   │   ├── hf_vqa.py          # Multimodal VLM inference client with fallback
│   │   └── spectral_vqa_fallback.py # Offline deterministic spectral reasoner
│   └── pipeline.py            # Master orchestration controller
└── tests/
    └── test_pipeline.py       # Comprehensive unit & integration tests
```

---

## 🧪 Running Automated Tests

Run the complete test suite:

```bash
.\.venv\Scripts\python.exe -m unittest tests/test_pipeline.py
```

---

## 💻 Hardware Requirements Assessment

* **Local Geospatial Computation**: Handled with fast vectorized NumPy and Rasterio operations on standard CPUs (e.g. Intel Core i5/i7/i9, AMD Ryzen) with 8–16 GB RAM.
* **VLM Deep Learning Inference**: Uses cloud serverless APIs (or lightweight local ONNX backends), eliminating the need for expensive dedicated multi-GPU workstations or high VRAM consumption.
