# VigilAudio: AI-Powered Audio Moderation Engine

**A production-ready audio emotion classification system built for content moderation.**

VigilAudio is the first phase of a multimodal moderation suite designed to detect distress, aggression, and safety risks in user-generated content. Unlike traditional moderators that look for keywords, VigilAudio listens to the *tone* of the voice—detecting anger, fear, or distress even when the words themselves are neutral.

## Key Features

*   **State-of-the-Art Architecture:** Fine-tuned `facebook/wav2vec2-base-960h` Transformer model.
*   **High Accuracy:** Achieved **82% accuracy** on a 7-class emotion dataset (Angry, Happy, Sad, Fearful, Disgusted, Neutral, Surprised).
*   **Production Pipeline:** End-to-end data harmonization, stratified splitting, and efficient feature extraction.
*   **Cloud-Native Training:** Optimized training scripts for Google Colab (T4 GPU), reducing training time from 50+ hours to <20 minutes.

## Technology Stack

*   **Language:** Python 3.10+
*   **Environment:** `uv` (for fast dependency management)
*   **ML Framework:** PyTorch, Hugging Face Transformers, Accelerate
*   **Audio Processing:** Librosa, Soundfile
*   **Data Ops:** Pandas, Scikit-learn

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/vigilaudio.git
    cd vigilaudio
    ```

2.  **Initialize the environment:**
    We use `uv` for lightning-fast setups.
    ```bash
    uv sync
    ```

## Execution Guide

### 1. Data Pipeline (Harmonization)
Turn raw, messy folders into a clean, stratified dataset.
```bash
uv run src/data/harmonize.py
```
*   **Input:** Raw audio folders (`Emotions/Angry`, `Emotions/Happy`...)
*   **Output:** `data/processed/metadata.csv` (Unified labels + 80/10/10 splits)

### 2. Feature Extraction (Local Test)
Verify that your machine can process audio using the Wav2Vec2 processor.
```bash
uv run src/features/extractor.py
```
*   **Output:** Prints the embedding shape `(768,)` for a sample file.

### 3. Model Training (The "Professional" Way)
Training a Transformer on a CPU is too slow. We use Google Colab.

1.  Upload `train_colab.py` and your `Emotions` folder to Google Drive.
2.  Open `VigilAudio_Fine_Tuning.ipynb` in Colab.
3.  Set Runtime to **T4 GPU**.
4.  Run the training script.
    *   **Result:** A fine-tuned model saved to `wav2vec2-finetuned/`.
    *   **Performance:** ~82% Accuracy / 0.81 F1 Score.

## Dataset

The model was trained on a combined dataset of **12,798 audio recordings** across 7 emotions.
*   **Source:** [Kaggle - Audio Emotions Dataset](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions)
*   **Composition:** An amalgam of CREMA-D, TESS, RAVDESS, and SAVEE datasets.

## Results Summary

| Model | Architecture | Training Time | Accuracy |
|-------|--------------|---------------|----------|
| Baseline | Simple MLP (CPU) | ~3 hours | 54% |
| **VigilAudio** | **Fine-Tuned Wav2Vec2 (GPU)** | **17 mins** | **82%** |

## License

MIT
