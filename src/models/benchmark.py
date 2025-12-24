import os
import time
import torch
import librosa
import pandas as pd
import numpy as np
from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- CONFIG ---
MODELS = {
    "PyTorch (Full)": "models/wav2vec2-finetuned",
    "ONNX (Standard)": "models/onnx",
    "ONNX (INT8 Quantized)": "models/onnx_quantized"
}
METADATA_PATH = "data/processed/metadata.csv"
TEST_SAMPLES = 50 # Small subset for speed comparison

def get_dir_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024) # Return MB

def run_benchmark():
    print("Starting VigilAudio Benchmark...")
    df = pd.read_csv(METADATA_PATH)
    test_df = df[df['split'] == 'test'].sample(min(TEST_SAMPLES, len(df)))
    
    # Load feature extractor (shared)
    extractor = AutoFeatureExtractor.from_pretrained(MODELS["PyTorch (Full)"])
    
    # Label Map
    emotions = sorted(df['emotion'].unique())
    label_map = {name: i for i, name in enumerate(emotions)}
    
    # Prepare test data in memory to isolate inference speed
    print(f"Pre-loading {len(test_df)} audio files into memory...")
    audio_data = []
    y_true = []
    for _, row in test_df.iterrows():
        # Handle Windows paths
        path = row['path']
        if not os.path.exists(path):
            path = os.path.join("C:/dev/archive/Emotions", row['emotion'].capitalize(), row['filename'])
        
        speech, _ = librosa.load(path, sr=16000)
        audio_data.append(speech)
        y_true.append(label_map[row['emotion']])

    results = []

    for name, path in MODELS.items():
        print(f"\nBenchmarking {name}...")
        
        # 1. Load Model
        start_load = time.time()
        if "ONNX" in name:
            model = ORTModelForAudioClassification.from_pretrained(path)
        else:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(path)
        load_time = time.time() - start_load
        
        y_pred = []
        latencies = []
        
        # 2. Warmup
        model(extractor(audio_data[0], sampling_rate=16000, return_tensors="pt", padding=True).input_values)
        
        # 3. Inference Loop
        for speech in tqdm(audio_data, desc=f"Predicting with {name}"):
            inputs = extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            
            start_inf = time.time()
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            latency = (time.time() - start_inf) * 1000 # to ms
            latencies.append(latency)
            
            pred_id = torch.argmax(logits, dim=-1).item()
            y_pred.append(pred_id)
            
        # 4. Metrics
        avg_latency = np.mean(latencies)
        acc = accuracy_score(y_true, y_pred)
        model_size = get_dir_size(path)
        
        # Store baseline for speedup calc
        if name == "PyTorch (Full)":
            baseline_latency = avg_latency
            speedup = 1.0
        else:
            speedup = baseline_latency / avg_latency if 'baseline_latency' in locals() else 1.0
        
        results.append({
            "Model": name,
            "Accuracy": f"{acc:.2%}",
            "Latency (Avg ms)": f"{avg_latency:.2f}ms",
            "Speedup": f"{speedup:.2f}x",
            "Size (MB)": f"{model_size:.1f}MB"
        })

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print("VIGILAUDIO PERFORMANCE REPORT")
    print("="*60)
    report_df = pd.DataFrame(results)
    print(report_df.to_string(index=False))
    print("="*60)
    
    # Save report
    report_df.to_csv("docs/benchmark_report.csv", index=False)
    print("Report saved to docs/benchmark_report.csv")

if __name__ == "__main__":
    if os.path.exists(METADATA_PATH):
        run_benchmark()
    else:
        print("Metadata not found. Please run harmonization first.")
