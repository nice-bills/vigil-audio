import streamlit as st
import librosa
import numpy as np
import plotly.express as px
import pandas as pd
from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import AutoFeatureExtractor
import torch
import tempfile
import os
import soundfile as sf
import uuid
from datetime import datetime
from audio_recorder_streamlit import audio_recorder

# --- CONFIG ---
MODEL_PATH = "models/onnx_quantized"
WINDOW_SIZE_SEC = 2.0 
MAX_DURATION_SEC = 60.0 
UPLOAD_DIR = "data/uploads/weak_predictions"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="VigilAudio Standalone",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
.emotion-badge { 
    padding: 10px 20px; 
    border-radius: 10px; 
    font-weight: bold; 
    text-transform: uppercase;
    font-size: 28px;
    text-align: center;
}
.alert-banner {
    background-color: rgba(255, 75, 75, 0.15);
    border: 2px solid #ff4b4b;
    color: #ff4b4b;
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
    margin-bottom: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
        model = ORTModelForAudioClassification.from_pretrained(MODEL_PATH, file_name="model_quantized.onnx")
        return extractor, model
    except Exception as e:
        st.error(f"Critical Error: Could not load model from {MODEL_PATH}.")
        return None, None

def save_training_sample(audio_chunk, sr, predicted_emotion, confidence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{predicted_emotion}_{confidence:.2f}_{unique_id}.wav"
    path = os.path.join(UPLOAD_DIR, filename)
    try:
        sf.write(path, audio_chunk, sr)
    except: pass

feature_extractor, model = load_model()
id2label = model.config.id2label if model else {}

# --- UI HEADER ---
st.title("VigilAudio: Standalone Moderation Demo")
st.caption(f"Edge-Optimized Content Safety Engine (Limit: {MAX_DURATION_SEC}s)")
st.markdown("---")

# --- INPUT TABS ---
# This separates the logic to avoid state conflicts
tab_upload, tab_mic = st.tabs(["Upload File", "Live Microphone"])

audio_source_bytes = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload Audio Content", type=["wav", "mp3", "m4a"], key="uploader_widget")
    if uploaded_file:
        audio_source_bytes = uploaded_file.getvalue()

with tab_mic:
    st.write("Capture Live Audio")
    recorded_audio_bytes = audio_recorder(text="Start Recording", icon_size="2x", key="recorder_widget")
    if recorded_audio_bytes:
        audio_source_bytes = recorded_audio_bytes

# --- MAIN CONTENT ---
if audio_source_bytes is not None and model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_source_bytes)
        tmp_path = tmp.name

    st.audio(audio_source_bytes)
    
    if st.button("🚀 Analyze Content", type="primary", use_container_width=True):
        with st.spinner("Analyzing audio context..."):
            y, sr = librosa.load(tmp_path, sr=16000)
            original_duration = librosa.get_duration(y=y, sr=sr)
            
            if original_duration > MAX_DURATION_SEC:
                st.warning(f"Long audio detected ({original_duration:.1f}s). Analyzing only the first {MAX_DURATION_SEC} seconds.")
                y = y[:int(MAX_DURATION_SEC * sr)]
            
            duration = librosa.get_duration(y=y, sr=sr)
            
            # --- INFERENCE LOGIC ---
            timeline = []
            chunk_len = int(WINDOW_SIZE_SEC * sr)
            for i, start in enumerate(range(0, len(y), chunk_len)):
                chunk = y[start:start + chunk_len]
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
                    "start_sec": i * WINDOW_SIZE_SEC,
                    "end_sec": min((i + 1) * WINDOW_SIZE_SEC, duration),
                    "emotion": emotion_label,
                    "confidence": round(confidence, 4)
                })

            if not timeline:
                st.error("Audio content too short to analyze.")
            else:
                # --- MODERATION LOGIC ---
                flagged_emotions = ['angry', 'fearful', 'disgusted']
                is_flagged = any(seg['emotion'] in flagged_emotions and seg['confidence'] > 0.85 for seg in timeline)
                
                if is_flagged:
                    st.markdown("""
                        <div class="alert-banner">
                            MODERATION ALERT: High-intensity negative sentiment detected. Human review recommended.
                        </div>
                    """, unsafe_allow_html=True)

                # --- DISPLAY RESULTS ---
                col1, col2 = st.columns([1, 2])
                color_map = {"angry": "#f48771", "happy": "#89d185", "sad": "#4fc1ff", "fearful": "#c586c0", "disgusted": "#ce9178", "neutral": "#808080", "suprised": "#dcdcaa"}
                
                with col1:
                    st.subheader("Dominant Tone")
                    emotions_list = [seg["emotion"] for seg in timeline]
                    dominant = max(set(emotions_list), key=emotions_list.count)
                    color = color_map.get(dominant, "#ffffff")
                    st.markdown(f'<div style="background-color: {color}22; border: 2px solid {color}; color: {color};" class="emotion-badge">{dominant}</div>', unsafe_allow_html=True)
                    
                    dist_counts = pd.Series(emotions_list).value_counts().reset_index()
                    dist_counts.columns = ['emotion', 'count']
                    fig_dist = px.pie(dist_counts, values='count', names='emotion', color='emotion', color_discrete_map=color_map, hole=0.4, height=250)
                    fig_dist.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    st.subheader("Confidence Timeline")
                    timeline_df = pd.DataFrame(timeline)
                    fig_timeline = px.bar(timeline_df, x="start_sec", y="confidence", color="emotion", color_discrete_map=color_map, labels={"start_sec": "Time (s)", "confidence": "Confidence"})
                    fig_timeline.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0), height=300)
                    st.plotly_chart(fig_timeline, use_container_width=True)

    os.remove(tmp_path)

else:
    st.info("System standby. Please provide audio content via upload or microphone.")
