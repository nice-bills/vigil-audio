from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
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
import asyncio

app = FastAPI(title="VigilAudio: Optimized API with Real-time Streaming")

# --- CONFIG ---
MODEL_PATH = "models/onnx_quantized"
UPLOAD_DIR = "data/uploads/weak_predictions"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- MODEL LOADING ---
print(f"Loading optimized INT8 model...")
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = ORTModelForAudioClassification.from_pretrained(MODEL_PATH, file_name="model_quantized.onnx")
    id2label = model.config.id2label
    print(f"API Ready. Labels: {list(id2label.values())}")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# --- HELPER FUNCTIONS ---
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

# --- STREAMING MANAGER ---
class AudioStreamBuffer:
    def __init__(self, sample_rate=16000, window_size_sec=2.0):
        self.sr = sample_rate
        self.window_size = int(sample_rate * window_size_sec)
        self.buffer = np.array([], dtype=np.float32)

    def add_chunk(self, chunk_bytes):
        # Convert raw bytes to float32 array (assuming 16-bit PCM for now)
        # Note: Ideally, we should resample here if input is not 16kHz
        chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.append(self.buffer, chunk)
        
        # Keep only the last window_size samples (Sliding Window)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

    def is_ready(self):
        return len(self.buffer) >= self.window_size

# --- ENDPOINTS ---
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

    # 1. Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 2. Load and Resample
        speech, sr = librosa.load(tmp_path, sr=16000)
        duration = librosa.get_duration(y=speech, sr=sr)
        
        timeline = []
        
        # 3. Process segments
        for i, chunk in enumerate(segment_audio(speech, sr, window_size=3.0)):
            if len(chunk) < 8000: continue # Skip very small fragments
                
            inputs = feature_extractor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_id = torch.argmax(logits, dim=-1).item()
                confidence = float(probs[0][pred_id])
            
            emotion_label = id2label[pred_id]
            
            # --- DATA FLYWHEEL (Active Learning) ---
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

        # 4. Overall Summary
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

@app.websocket("/stream/audio")
async def stream_audio(websocket: WebSocket, rate: int = 16000):
    await websocket.accept()
    print(f"WebSocket Connected (Input Rate: {rate}Hz)")
    buffer = AudioStreamBuffer()
    
    # Pre-configure resampler if rate != 16000
    resampler = None
    if rate != 16000:
        import torchaudio.transforms as T
        resampler = T.Resample(rate, 16000)

    try:
        while True:
            data = await websocket.receive_bytes()
            
            # 1. Convert to tensor
            chunk = torch.from_numpy(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
            
            # 2. Resample if necessary
            if resampler:
                chunk = resampler(chunk)
            
            # 3. Add to buffer
            buffer.add_chunk(chunk.numpy().tobytes()) # Convert back to bytes for the buffer manager
            
            if buffer.is_ready():
                inputs = feature_extractor(buffer.buffer, sampling_rate=16000, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred_id = torch.argmax(outputs.logits, dim=-1).item()
                    confidence = float(probs[0][pred_id])
                
                # 4. Confidence Threshold (0.85)
                # We only send if we are confident, or send a 'low_confidence' status
                response = {
                    "emotion": id2label[pred_id],
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "status": "high_confidence" if confidence > 0.85 else "low_confidence"
                }
                
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        print("WebSocket Disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        try:
            await websocket.close()
        except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)