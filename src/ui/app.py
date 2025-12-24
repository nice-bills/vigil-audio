import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np
import os

# --- CONFIG ---
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="VigilAudio Dashboard",
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

# --- UI HEADER ---
st.title("🛡️ VigilAudio: Emotion Analytics")
st.markdown("---")

# --- MAIN UPLOAD & RECORD ---
col_upload, col_record = st.columns(2)
with col_upload:
    uploaded_file = st.file_uploader("📂 Upload Audio File", type=["wav", "mp3", "m4a"])
with col_record:
    recorded_audio = st.audio_input("🎙️ Record Voice")

# Logic to prioritize recording if both exist
audio_source = recorded_audio if recorded_audio else uploaded_file

# --- MAIN CONTENT ---
if audio_source is not None:
    st.audio(audio_source)

    if st.button("🚀 Analyze Emotion", type="primary"):
        with st.spinner("Sending to VigilAudio API..."):
            # 1. Send to FastAPI
            # We use a generic filename if recording
            filename = audio_source.name if hasattr(audio_source, 'name') else "recording.wav"
            files = {"file": (filename, audio_source.getvalue())}
            
            try:
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # --- PROCESS RESULTS ---
                    dominant = data['dominant_emotion']
                    timeline = data['timeline']
                    
                    # Calculate Distribution
                    emotions_list = [seg["emotion"] for seg in timeline]
                    dist_counts = pd.Series(emotions_list).value_counts().reset_index()
                    dist_counts.columns = ['emotion', 'count']
                    
                    # --- DISPLAY ---
                    col1, col2 = st.columns([1, 2])
                    
                    color_map = {
                        "angry": "#f48771", "happy": "#89d185", "sad": "#4fc1ff",
                        "fearful": "#c586c0", "disgusted": "#ce9178", "neutral": "#808080",
                        "surprised": "#dcdcaa"
                    }
                    
                    with col1:
                        st.subheader("Summary")
                        color = color_map.get(dominant, "#ffffff")
                        
                        st.markdown(f"""
                            <div style="background-color: {color}22; border: 2px solid {color}; text-align: center; margin-bottom: 20px;" class="emotion-badge">
                                {dominant}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Pie Chart
                        fig_dist = px.pie(
                            dist_counts, values='count', names='emotion',
                            color='emotion', color_discrete_map=color_map,
                            hole=0.4
                        )
                        fig_dist.update_layout(
                            template="plotly_dark",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False,
                            height=200
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                    with col2:
                        st.subheader("Timeline Analysis")
                        timeline_df = pd.DataFrame(timeline)
                        fig_timeline = px.bar(
                            timeline_df, x="start_sec", y="confidence", color="emotion",
                            color_discrete_map=color_map, 
                            labels={"start_sec": "Time (s)", "confidence": "Confidence"}
                        )
                        fig_timeline.update_layout(
                            template="plotly_dark", 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=300
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
                        
                    # Raw JSON
                    with st.expander("View API Response"):
                        st.json(data)
                        
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}. Is the API running?")

else:
    st.info("👆 Upload an audio file or record your voice to begin.")