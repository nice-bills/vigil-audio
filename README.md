# VigilAudio: AI-Powered Audio Moderation Engine

**A production-ready audio emotion classification system built for content moderation.**

VigilAudio is an advanced audio analysis engine designed to detect aggression, distress, and safety risks by analyzing the *tone* of voice. It is the audio foundation of a multimodal moderation suite, utilizing fine-tuned Transformers and optimized for high-speed CPU inference.

![Dashboard](docs/screenshot_placeholder.png)

## Dataset & Results

*   **Source:** [Kaggle - Audio Emotions Dataset](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions) (12,798 recordings).
*   **Architecture:** Fine-tuned `Wav2Vec2` Transformer.
*   **Accuracy:** **83%** (PyTorch) / **84%** (Optimized INT8 ONNX).
*   **Optimization:** 1.85x speedup and 67% size reduction via INT8 Quantization.

---

## Prerequisites

*   **Python 3.10+**
*   **uv:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended for environment management).
*   **FFMPEG:** Required for audio processing.
    *   *Windows:* `winget install ffmpeg`
    *   *Linux:* `sudo apt install ffmpeg`

---

## How to Run (Quick Start)

### 1. Setup Environment
```bash
git clone https://github.com/yourusername/vigilaudio.git
cd vigilaudio
uv sync
```

### 2. Download Model Weights
Because model weights are large, they are not stored in Git.
1. Download `wav2vec2_model.zip` from [Your Link/Releases].
2. Extract to `models/onnx_quantized/`.

### 3. Launch the Application
Run the standalone demo (recommended for local testing):
```bash
uv run streamlit run src/ui/app_standalone.py
```
*   **Access:** `http://localhost:8501`

---

## Development Workflow

If you want to retrain or modify the system:

### 1. Data Preparation
1. Download the [Kaggle Dataset](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions).
2. Place the folders (Angry, Happy, etc.) in `data/raw/Emotions/`.
3. Run harmonization:
```bash
uv run src/data/harmonize.py
```

### 2. Model Training (Cloud Accelerated)
We use Google Colab (T4 GPU) for high-speed fine-tuning.
*   The training script and notebook are in `docs/VigilAudio_Fine_Tuning.ipynb`.

### 3. Optimization & Benchmarking
Convert to ONNX and verify performance:
```bash
uv run src/models/optimize.py
uv run src/models/benchmark.py
```

---

## Project Structure

```text
vigilaudio/
├── data/                   # Dataset storage
│   ├── raw/                # Original audio files (excluded from Git)
│   └── processed/          # Metadata and splits
├── models/                 # Model registry
│   ├── wav2vec2-finetuned/ # PyTorch weights
│   └── onnx_quantized/     # Optimized INT8 engine
├── src/
│   ├── api/                # FastAPI backend service
│   ├── data/               # ETL and harmonization scripts
│   ├── features/           # Audio feature extraction
│   ├── models/             # Training, Inference, and Optimization logic
│   └── ui/                 # Streamlit frontend dashboards
├── docs/                   # Benchmarks, Logs, and Colab Notebooks
└── notebooks/              # Experimental EDA
```

## Performance Optimization (ONNX)

| Model Version | Accuracy | Latency (ms) | Speedup | Size (MB) |
|---------------|----------|--------------|---------|-----------|
| PyTorch (Full) | 82.0% | 370ms | 1.00x | 361MB |
| ONNX (Standard)| 82.0% | 306ms | 1.21x | 361MB |
| **ONNX (INT8)** | **84.0%** | **199ms** | **1.85x** | **116MB** |

## License
MIT