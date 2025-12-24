from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import shutil
import os
import torch
import librosa
import numpy as np
from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import AutoFeatureExtractor
from typing import List, Dict
import tempfile

app = FastAPI(title="VigilAudio Optimized API")

# --- CONFIG ---
# We use the INT8 model which proved to be the fastest in benchmarks
MODEL_PATH = "models/onnx_quantized"

# --- MODEL LOADING (Optimized with ONNX) ---
print(f"Loading OPTIMIZED INT8 ONNX model into memory...")
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    # Note: we explicitly pass file_name since optimum expects model.onnx by default
    model = ORTModelForAudioClassification.from_pretrained(MODEL_PATH, file_name="model_quantized.onnx")
    
    # Label mapping from config
    id2label = model.config.id2label
    print(f"Optimized API Ready. Speedup expected: ~1.8x")
except Exception as e:
    print(f"API Failed to load model: {e}")
    model = None

# --- UTILS ---
def segment_audio(audio, sr, window_size=3.0):
    chunk_len = int(window_size * sr)
    for i in range(0, len(audio), chunk_len):
        yield audio[i:i + chunk_len]

# --- ENDPOINTS ---
@app.get("/health")
def health():
    return {
        "status": "online",
        "engine": "ONNX Runtime (INT8)",
        "model_loaded": model is not None,
        "labels": list(id2label.values()) if model else []
    }

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model weights missing on server.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        speech, sr = librosa.load(tmp_path, sr=16000)
        duration = librosa.get_duration(y=speech, sr=sr)
        
        timeline = []
        
        for i, chunk in enumerate(segment_audio(speech, sr, window_size=3.0)):
            if len(chunk) < 8000: continue
                
            inputs = feature_extractor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # ONNX Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_id = torch.argmax(logits, dim=-1).item()
            
            timeline.append({
                "start_sec": i * 3.0,
                "end_sec": min((i + 1) * 3.0, duration),
                "emotion": id2label[pred_id],
                "confidence": round(float(probs[0][pred_id]), 4)
            })

        emotions_list = [seg["emotion"] for seg in timeline]
        dominant = max(set(emotions_list), key=emotions_list.count) if emotions_list else "unknown"

        return {
            "filename": file.filename,
            "engine": "ONNX_INT8",
            "duration_seconds": round(duration, 2),
            "dominant_emotion": dominant,
            "timeline": timeline
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)