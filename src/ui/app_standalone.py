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
import matplotlib.pyplot as plt 
# --- CONFIG ---
MODEL_PATH = "models/onnx_quantized"
WINDOW_SIZE_SEC = 3.0

st.set_page_config(
    page_title="VigilAudio Standalone",
    page_icon="🎙️",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
.emotion-badge { 
    padding: 5px 15px; 
    border-radius: 20px; 
    font-weight: bold; 
    text-transform: uppercase;
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (Singleton) ---
@st.cache_resource
def load_model():
    """Loads the ONNX model and feature extractor into memory."""
    print("Loading ONNX model into Streamlit memory...")
    try:
        extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
        model = ORTModelForAudioClassification.from_pretrained(MODEL_PATH, file_name="model_quantized.onnx")
        print("Model loaded successfully.")
        return extractor, model
    except Exception as e:
        print(f"Failed to load model: {e}")
        st.error(f"Critical Error: Could not load model from {MODEL_PATH}. Please check the files.")
        return None, None

feature_extractor, model = load_model()
id2label = model.config.id2label if model else {}

# --- UI HEADER ---
st.title("VigilAudio: Standalone Emotion Analyzer")
st.markdown("---")

# --- MAIN UPLOAD ---
uploaded_file = st.file_uploader("Upload Audio File (WAV, MP3)", type=["wav", "mp3", "m4a"])

# --- MAIN CONTENT ---
if uploaded_file is not None and model is not None:
    # Save to temp file to load with librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.audio(uploaded_file)
    
    # Analyze Button
    if st.button("Analyze Emotion", type="primary"):
        with st.spinner("Processing audio..."):
            y, sr = librosa.load(tmp_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # --- INFERENCE LOGIC (Embedded in UI) ---
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
                
                timeline.append({
                    "start_sec": i * WINDOW_SIZE_SEC,
                    "end_sec": min((i + 1) * WINDOW_SIZE_SEC, duration),
                    "emotion": id2label[pred_id],
                    "confidence": round(float(probs[0][pred_id]), 4)
                })

            if not timeline:
                st.error("Audio too short to analyze.")
            else:
                emotions_list = [seg["emotion"] for seg in timeline]
                dominant = max(set(emotions_list), key=emotions_list.count)
                
                # --- DISPLAY RESULTS ---
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    color_map = {
                        "angry": "#f48771", "happy": "#89d185", "sad": "#4fc1ff",
                        "fearful": "#c586c0", "disgusted": "#ce9178", "neutral": "#808080",
                        "surprised": "#dcdcaa"
                    }
                    color = color_map.get(dominant, "#ffffff")
                    
                    st.markdown(f"""
                        <div style="background-color: {color}22; border: 2px solid {color}; text-align: center;" class="emotion-badge">
                            {dominant}
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    timeline_df = pd.DataFrame(timeline)
                    fig_timeline = px.bar(
                        timeline_df, x="start_sec", y="confidence", color="emotion",
                        color_discrete_map=color_map, title="Emotion Timeline",
                        labels={"start_sec": "Time (s)", "confidence": "Confidence"}
                    )
                    fig_timeline.update_layout(
                        template="plotly_dark", 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

    # Cleanup temp file
    os.remove(tmp_path)

else:
    st.info("Upload an audio file to begin.")
