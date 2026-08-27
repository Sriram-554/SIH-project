# 🛰️ SatQuery AI

### An Interactive Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis Through Text Queries

SatQuery AI is a **Smart India Hackathon (SIH) prototype** for interactive Earth-observation analysis. Users ask natural-language questions about satellite imagery, and SatQuery routes the request to an appropriate remote-sensing analysis workflow.

The system combines **remote-sensing computation, deterministic spectral reasoning, optional cloud VLM inference, bi-temporal analysis, spatial grounding, and optical–SAR analysis** in one Streamlit application.

> **Prototype status:** SatQuery is a research/demo prototype. Some modules currently use deterministic heuristics or synthetic fallback data when a trained model or raw modality is unavailable. Results from these paths should be interpreted accordingly.

## 🎯 SIH Alignment

SatQuery is developed for the ISRO/SAC problem statement **SIH26167 — “SatQuery AI - An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries”**, listed under the Space Technology theme.

## ✨ Core Capabilities

### 1. 🧠 Agentic Query Orchestration

A central controller interprets the query, validates available inputs, selects a specialist tool, executes it, and records an execution trace.

Current specialist tools include:

- Single-image VQA
- Remote-sensing captioning
- Region/feature grounding
- Bi-temporal change analysis
- Optical + SAR analysis
- Spectral-index analysis

The current query router is primarily **deterministic intent/keyword based**, which keeps the prototype lightweight and predictable.

### 2. 🛰️ Multispectral Remote-Sensing Analysis

For compatible Sentinel-2 products, SatQuery can process:

- B02 — Blue
- B03 — Green
- B04 — Red
- B08 — NIR
- B11 — SWIR-1
- B12 — SWIR-2

It computes:

- **NDVI:** `(NIR - Red) / (NIR + Red)`
- **NDWI (McFeeters):** `(Green - NIR) / (Green + NIR)`
- **NBR:** `(NIR - SWIR2) / (NIR + SWIR2)`
- RGB and false-colour composites
- Basic zonal statistics

### 3. 💬 Vision-Language Reasoning

SatQuery supports an optional Hugging Face VLM backend. Candidate models include:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-72B-Instruct`
- `google/gemma-3-27b-it`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`

When remote VLM inference is unavailable, SatQuery can fall back to a deterministic **Spectral Fallback Engine** for supported spectral questions.

### 4. 📈 Bi-Temporal Change Analysis

Given two observations, the prototype can calculate:

- RGB structural difference
- Change percentage
- ΔNDVI when NDVI arrays are available
- Vegetation gain/loss/stability
- Spatial difference maps

The current implementation uses image-difference and spectral thresholds. It should therefore be treated as **prototype change analysis**, not a validated operational change-detection model.

### 5. 📍 Visual / Spectral Feature Grounding

The grounding module creates masks, approximate spatial clusters, bounding boxes, centroids, and annotated overlays for features such as:

- Water
- Vegetation / crops
- Dense vegetation
- Built-up / barren regions

The current implementation uses spectral/RGB thresholds and raster clustering rather than a trained segmentation model.

### 6. 🛰️ Optical + SAR Analysis

SatQuery supports an optical + SAR analysis workflow.

When an actual SAR raster is supplied, it can be processed alongside optical imagery. When raw SAR is unavailable, the prototype can generate a **synthetic SAR-like representation from optical imagery** so the workflow can still be demonstrated.

> **Important:** synthetic SAR output is not real RISAT/Sentinel-1 sensor data and must not be presented as such.

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │ Query + Imagery      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ SatQuery Pipeline    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Agentic Controller   │
                    │ Routing + Validation │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       Spectral Engine     VLM + Fallback   Specialist
                                             Tools
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Answer + Evidence    │
                    │ + Execution Trace    │
                    └──────────────────────┘
```

## 📁 Project Structure

```text
SIH-project/
├── app.py
├── README.md
├── requirements.txt
├── run_satquery.bat
├── train_bigearthnet_adapter.py
├── ndvi_test.py
├── test_vqa.py
├── data/
├── models/
├── outputs/
├── tests/
│   └── test_pipeline.py
└── src/
    ├── agent/
    │   ├── orchestrator.py
    │   └── tool_registry.py
    ├── analysis/
    │   ├── spectral_indices.py
    │   ├── sentinel_metadata.py
    │   ├── visualize_sentinel.py
    │   ├── change_detector.py
    │   ├── grounding.py
    │   ├── sar_optical_fusion.py
    │   ├── cdvqa_engine.py
    │   ├── query_router.py
    │   └── input_validator.py
    ├── models/
    │   ├── vqa_model.py
    │   ├── hf_vqa.py
    │   └── spectral_vqa_fallback.py
    └── pipeline.py
```

## 💻 Requirements

- Python 3.10+
- 8 GB RAM minimum; 16 GB recommended for comfortable local geospatial processing
- Internet access only when using the optional cloud VLM backend
- No dedicated GPU is required for the local spectral/image-processing components

## 🚀 Quick Start

### Windows

```powershell
git clone https://github.com/Sriram-554/SIH-project.git
cd SIH-project

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m streamlit run app.py
```

If PowerShell blocks activation, run Streamlit directly through the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also use the included Windows launcher:

```text
run_satquery.bat
```

## 🔐 Optional Hugging Face Configuration

The VLM backend is optional. Without a token, supported spectral queries can use the local deterministic fallback.

PowerShell:

```powershell
$env:HF_TOKEN="your_huggingface_token_here"
python -m streamlit run app.py
```

A token can also be entered through the application UI if enabled.

**Never commit API tokens, passwords, or other secrets to GitHub.**

## 🧪 Testing

Run the repository test suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_pipeline.py
```

The tests cover:

- NDVI / NDWI / NBR calculations
- Zonal statistics
- Query routing
- Input validation
- RGB change detection
- Visual grounding
- Spectral fallback reasoning
- Basic end-to-end pipeline execution

The tests validate deterministic prototype components; they do not guarantee availability of an external VLM provider.

## 🧪 Recommended SIH Demo Workflow

For a stable presentation, demonstrate a small number of reliable workflows:

1. **Single-image question** — ask about land cover or vegetation.
2. **Feature grounding** — ask to highlight a water body or vegetation region.
3. **Bi-temporal analysis** — ask what changed between two observations.
4. **Optical + SAR** — demonstrate fusion and clearly distinguish real SAR input from synthetic fallback.
5. **Execution trace** — show how the query was routed and which specialist tool executed it.

## ⚠️ Prototype Limitations

SatQuery is intentionally a hackathon prototype. Important limitations are:

- The query router currently uses deterministic intent rules rather than a fully learned planner.
- Grounding uses spectral/RGB thresholding and raster clustering rather than a trained segmentation model.
- Change detection uses image-difference and spectral thresholds and is not a validated operational change-detection model.
- Optical/SAR classification uses heuristic thresholds; synthetic SAR may be generated when real SAR is absent.
- The spectral fallback is deterministic and should not be described as a VLM.
- Cloud VLM inference depends on external model/provider availability and network access.
- Scientific accuracy depends on image quality, co-registration, available bands, calibration, and the suitability of the selected thresholds/model.

## 🌍 Why SatQuery Is More Than a Generic AI Chatbot

A generic chatbot primarily generates text from a prompt. SatQuery adds a **remote-sensing analysis layer** before and around language generation:

```text
Natural-language query
        ↓
Remote-sensing task identification
        ↓
Input / modality validation
        ↓
Specialist tool selection
        ↓
Spectral / temporal / spatial computation
        ↓
Optional VLM reasoning
        ↓
Evidence + visualization + execution trace
```

This makes SatQuery a system designed around **Earth-observation data and analysis**, rather than simply attaching a chatbot to an image.

## 📌 Recommended Presentation Description

> **“SatQuery is an agentic remote-sensing intelligence platform that converts natural-language questions into validated, task-specific Earth-observation analysis workflows, combining physical spectral computation, spatial and temporal analysis, optional vision-language reasoning, and auditable execution traces.”**

## 📄 License

This repository is intended for academic and Smart India Hackathon prototype development.
