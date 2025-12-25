import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np
import os
import json
from audio_recorder_streamlit import audio_recorder

# --- CONFIG ---
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="VigilAudio Moderation Dashboard",
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

# --- UI HEADER ---
st.title("VigilAudio: Content Moderation Suite")
st.caption("Automated Tone and Emotion Analysis for Audio Safety")
st.markdown("---")

# --- INPUT SECTION ---
col_input, col_info = st.columns([2, 1])

with col_input:
    st.subheader("Input Source")
    c1, c2 = st.columns(2)
    with c1:
        uploaded_file = st.file_uploader("Upload Audio Content", type=["wav", "mp3", "m4a"])
    with c2:
        st.write("Record Direct Input")
        recorded_audio_bytes = audio_recorder(text="Start Recording", icon_size="2x")

# Select active source
audio_source = None
if recorded_audio_bytes:
    audio_source = {"name": "live_stream.wav", "bytes": recorded_audio_bytes}
elif uploaded_file:
    audio_source = {"name": uploaded_file.name, "bytes": uploaded_file.getvalue()}

with col_info:
    st.subheader("System Health")
    try:
        health_resp = requests.get("http://localhost:8000/health")
        health = health_resp.json()
        st.success(f"Status: {health['status'].upper()}")
        st.info(f"Engine: {health['engine']}")
    except:
        st.error("Backend Server Offline")

# --- ANALYSIS AND RESULTS ---
if audio_source:
    st.audio(audio_source["bytes"])
    
    if st.button("Analyze Content", type="primary", use_container_width=True):
        with st.spinner("Performing high-precision audit..."):
            files = {"file": (audio_source["name"], audio_source["bytes"])}
            try:
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process results
                    dominant = data['dominant_emotion']
                    timeline = data['timeline']
                    
                    # Moderation Logic: Flag high-confidence negative emotions
                    flagged_emotions = ['angry', 'fearful', 'disgusted']
                    is_flagged = any(seg['emotion'] in flagged_emotions and seg['confidence'] > 0.85 for seg in timeline)
                    
                    if is_flagged:
                        st.markdown("""
                            <div class="alert-banner">
                                MODERATION ALERT: High-intensity negative sentiment detected. Human review recommended.
                            </div>
                        """, unsafe_allow_html=True)

                    # Layout for charts
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    color_map = {
                        "angry": "#f48771", "happy": "#89d185", "sad": "#4fc1ff",
                        "fearful": "#c586c0", "disgusted": "#ce9178", "neutral": "#808080",
                        "suprised": "#dcdcaa"
                    }
                    
                    with res_col1:
                        st.subheader("Tone Summary")
                        color = color_map.get(dominant, "#ffffff")
                        st.markdown(f"""
                            <div style="background-color: {color}22; border: 2px solid {color}; color: {color};" class="emotion-badge">
                                {dominant}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Distribution Pie
                        emotions_list = [seg["emotion"] for seg in timeline]
                        dist_counts = pd.Series(emotions_list).value_counts().reset_index()
                        dist_counts.columns = ['emotion', 'count']
                        fig_dist = px.pie(
                            dist_counts, values='count', names='emotion',
                            color='emotion', color_discrete_map=color_map,
                            hole=0.4, height=250
                        )
                        fig_dist.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with res_col2:
                        st.subheader("Confidence Timeline")
                        timeline_df = pd.DataFrame(timeline)
                        fig_timeline = px.bar(
                            timeline_df, x="start_sec", y="confidence", color="emotion",
                            color_discrete_map=color_map,
                            labels={"start_sec": "Time (s)", "confidence": "Confidence"}
                        )
                        fig_timeline.update_layout(
                            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0)
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)

                    with st.expander("Detailed System Logs"):
                        st.json(data)
                else:
                    st.error(f"Prediction failed: {response.text}")
            except Exception as e:
                st.error(f"System Error: {e}")

else:
    st.info("System standby. Please provide audio content for safety analysis.")
