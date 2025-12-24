from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import shutil
import os
import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from typing import List, Dict
import tempfile

app = FastAPI(title="VigilAudio Emotion API")

MODEL_PATH = "models/wav2vec2-finetuned"
DEVICE = torch.device("cpu")

print(f"Loading model into API memory...")
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    id2label = model.config.id2label
    print(f"API Model Ready. Labels: {list(id2label.values())}")
except Exception as e:
    print(f"API Failed to load model: {e}")
    model = None

def segment_audio(audio, sr, window_size=3.0):
    """Splits audio into fixed-size windows."""
    chunk_len = int(window_size * sr)
    for i in range(0, len(audio), chunk_len):
        yield audio[i:i + chunk_len]

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "device": str(DEVICE)
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
            if len(chunk) < 8000: 
                continue
                
            inputs = feature_extractor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                logits = model(inputs.input_values.to(DEVICE)).logits
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
            "duration_seconds": round(duration, 2),
            "dominant_emotion": dominant,
            "timeline": timeline
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
