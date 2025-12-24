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
import soundfile as sf
import uuid
from datetime import datetime

app = FastAPI(title="VigilAudio: Optimized API with Active Learning")

MODEL_PATH = "models/onnx_quantized"
UPLOAD_DIR = "data/uploads/weak_predictions"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print(f"\nLoading optimized INT8 model...")
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = ORTModelForAudioClassification.from_pretrained(MODEL_PATH, file_name="model_quantized.onnx")
    id2label = model.config.id2label
    print(f"API Ready. Labels: {list(id2label.values())}")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

def segment_audio(audio, sr, window_size=3.0):
    """Splits audio into fixed-size windows."""
    chunk_len = int(window_size * sr)
    for i in range(0, len(audio), chunk_len):
        yield audio[i:i + chunk_len]

def save_training_sample(audio_chunk, sr, predicted_emotion, confidence):
    """Saves low-confidence chunks for future training (Active Learning)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{predicted_emotion}_{confidence:.2f}_{unique_id}.wav"
    path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        sf.write(path, audio_chunk, sr)
        print(f"Saved weak prediction for review: {filename}")
    except Exception as e:
        print(f"Failed to save sample: {e}")

@app.get("/health")
def health():
    return {
        "status": "online",
        "engine": "ONNX Runtime (INT8)",
        "model_loaded": model is not None,
        "active_learning_path": UPLOAD_DIR
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
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_id = torch.argmax(logits, dim=-1).item()
                confidence = float(probs[0][pred_id])
            
            emotion_label = id2label[pred_id]
            
            if confidence < 0.60:
                save_training_sample(chunk, sr, emotion_label, confidence)
            
            timeline.append({
                "start_sec": i * 3.0,
                "end_sec": min((i + 1) * 3.0, duration),
                "emotion": emotion_label,
                "confidence": round(confidence, 4)
            })

        if not timeline:
            raise HTTPException(status_code=400, detail="Audio file too short or empty.")

        emotions_list = [seg["emotion"] for seg in timeline]
        dominant = max(set(emotions_list), key=emotions_list.count)

        return {
            "filename": file.filename,
            "duration_seconds": round(duration, 2),
            "dominant_emotion": dominant,
            "timeline": timeline
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 5. Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
