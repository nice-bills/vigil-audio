# VigilAudio: AI-Powered Audio Moderation Engine

**VigilAudio** is a production-ready audio analysis engine designed to detect aggression, distress, and safety risks by analyzing the *tone* of voice. Built as the audio foundation for a multimodal moderation suite, it utilizes fine-tuned Transformers and optimized ONNX inference to deliver high-speed, real-time emotion detection.

## Features

*   **State-of-the-Art Architecture:** Fine-tuned `Wav2Vec2` Transformer achieving **84% accuracy**.
*   **Real-time Streaming:** WebSocket API for low-latency audio processing (~199ms).
*   **Active Learning:** Automatically captures low-confidence samples to build a "Data Flywheel" for continuous improvement.
*   **Moderation Alerts:** UI automatically flags high-intensity negative emotions (Anger, Fear).
*   **Edge Optimized:** INT8 Quantized ONNX model runs **2x faster** than standard PyTorch models.
*   **Dual Deployment:** Supports both a Monolithic (Standalone) and Microservice (API + UI) architecture.

## Tech Stack

*   **Core:** `Python 3.10+`
*   **Deep Learning:** `PyTorch`, `Transformers` (Hugging Face)
*   **Inference:** `ONNX Runtime`, `Optimum`
*   **Audio Processing:** `Librosa`, `Torchaudio`, `FFmpeg`
*   **Backend API:** `FastAPI`, `WebSockets`
*   **Frontend UI:** `Streamlit`
*   **Environment:** `uv`
*   **Containerization:** `Docker`

## Live Demo

The application is deployed and available for testing on Hugging Face Spaces:

**[VigilAudio Live Demo](https://huggingface.co/spaces/nice-bill/vigilaudio)**

*Note: The demo uses the optimized INT8 ONNX model for high-performance inference on CPU.*

## Project Structure

```
.
├── data/
│   ├── raw/            # Raw audio files (from Kaggle)
│   └── processed/      # Harmonized metadata and stratified splits
├── models/
│   ├── onnx_quantized/ # Optimized INT8 model ready for deployment
│   └── wav2vec2/       # Original fine-tuned PyTorch weights
├── src/
│   ├── api/
│   │   └── app.py      # FastAPI backend with WebSocket support
│   ├── data/
│   │   └── harmonize.py # Data standardization and splitting script
│   ├── features/       # Feature extraction logic
│   ├── models/
│   │   ├── train.py    # Head-only training script
│   │   ├── predict.py  # Inference script
│   │   └── optimize.py # ONNX export and quantization script
│   └── ui/
│       ├── app.py      # Frontend for Microservice mode
│       └── app_standalone.py # Monolithic app (recommended for demos)
├── docs/
│   └── VigilAudio_Fine_Tuning.ipynb # Colab notebook for full model training
├── Dockerfile          # Multi-stage Docker build
├── pyproject.toml      # Dependency management
└── README.md           # Project documentation
```

## Dataset

The project utilizes the **Audio Emotions Dataset** sourced from Kaggle.

*   **Source:** [Audio Emotions Dataset (Kaggle)](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions)
*   **Task:** Multi-class classification of emotional states from speech.
*   **Classes:** Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.
*   **Processing:** The `src/data/harmonize.py` script standardizes folder structures, validates audio files, and generates a stratified 80/10/10 split for robust training.

## Model Training & Performance

The model development followed a rigorous experimental path, moving from a simple baseline to a highly optimized production model.

### 1. Baseline: Head-Only Training
Initially, a simple Multi-Layer Perceptron (MLP) was trained on top of frozen `Wav2Vec2` embeddings. This approach was computationally cheap but yielded poor results, demonstrating the need for full fine-tuning.

### 2. Fine-Tuning
We fine-tuned the entire `Wav2Vec2` transformer on the dataset using a T4 GPU (via Google Colab). This allowed the model to learn acoustic features specific to emotional expression, significantly boosting accuracy.

### 3. Optimization (ONNX + Quantization)
To ensure the model could run in real-time on CPUs, we exported it to **ONNX** and applied **INT8 Quantization**. This reduced the model size by **3x** and improved latency by **1.85x** with a slight *increase* in accuracy (likely due to the regularization effect of quantization).

### Performance Summary

| Model Version | Accuracy | Latency (ms) | Speedup | Size (MB) |
|---------------|----------|--------------|---------|-----------|
| **Baseline (Head-only)** | **52.0%** | **< 50ms** | **~7x** | **3.5MB** |
| PyTorch (Full) | 82.0% | 370ms | 1.00x | 361MB |
| ONNX (Standard) | 82.00% | 306.52ms | 1.21x | 361.0MB |
| **ONNX (INT8)** | **84.0%** | **199ms** | **1.85x** | **116MB** |

*Note: The Baseline represents a simple neural network head trained on frozen Wav2Vec2 embeddings, illustrating the significant gain achieved by fine-tuning the backbone.*

## Setup and Usage

### Prerequisites
*   Python 3.10+
*   `uv` (recommended) or `pip`
*   `ffmpeg` (installed via system package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/vigilaudio.git
cd vigilaudio
```

### 2. Environment Setup
```bash
uv sync
```

### 3. Download Model Weights
1. Download the quantized model (`wav2vec2_quantized.zip`) from the Releases page.
2. Extract it to `models/onnx_quantized/`.

### 4. Run the Application

#### Mode A: Standalone Demo (Recommended)
Best for quick testing or Hugging Face Spaces. Runs Model + UI in a single process.
```bash
uv run streamlit run src/ui/app_standalone.py
```
*   **Access:** `http://localhost:8501`

#### Mode B: Microservice Architecture (Production)
Decouples the API from the frontend.

**Terminal 1 (Backend API):**
```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
**Terminal 2 (Frontend UI):**
```bash
uv run streamlit run src/ui/app.py
```

## Docker Deployment

The application is fully containerized and ready for deployment.

### Build the Image
```bash
docker build -t vigilaudio .
```

### Run the Container
```bash
docker run -p 8501:8501 vigilaudio
```
Access the app at `http://localhost:8501`.

## Contributing
Contributions are welcome! Please open an issue or submit a PR for improvements to the streaming logic or additional emotion classes.

## License
MIT