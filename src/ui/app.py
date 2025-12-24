import streamlit as st
import requests
import plotly.express as pd
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt

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
st.title("VigilAudio: Emotion Analytics")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Upload Control")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
    st.info("The system analyzes tone and distress patterns across the entire duration.")

# --- MAIN CONTENT ---
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Audio Source")
        st.audio(uploaded_file)
        
        # Load audio for waveform visualization
        y, sr = librosa.load(uploaded_file, sr=16000)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#1f77b4', alpha=0.7)
        ax.set_axis_off()
        fig.patch.set_facecolor('#0e1117')
        st.pyplot(fig)

    if st.button("Analyze Emotion"):
        with st.spinner("Analyzing audio segments..."):
            # 1. Send to FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                
                with col2:
                    st.subheader("Classification Results")
                    
                    # 2. Prominent Result
                    dominant = data['dominant_emotion']
                    color_map = {
                        "angry": "#f48771", "happy": "#89d185", "sad": "#4fc1ff",
                        "fearful": "#c586c0", "disgusted": "#ce9178", "neutral": "#808080",
                        "surprised": "#dcdcaa"
                    }
                    color = color_map.get(dominant, "#ffffff")
                    
                    st.markdown(f"""
                        <div style="background-color: {color}22; border: 2px solid {color};" class="emotion-badge">
                            Detected Emotion: <span style="color: {color};">{dominant}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 3. Timeline Plot
                    st.markdown("#### Emotion Timeline")
                    timeline_df = pd.DataFrame(data['timeline'])
                    
                    # Plotly chart
                    fig_timeline = pd.bar(
                        timeline_df, 
                        x="start_sec", 
                        y="confidence", 
                        color="emotion",
                        color_discrete_map=color_map,
                        title="Confidence by Segment (3s windows)",
                        labels={"start_sec": "Time (seconds)", "confidence": "Probability"}
                    )
                    fig_timeline.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # 4. Detailed Data
                    with st.expander("View Raw JSON Response"):
                        st.json(data)
            else:
                st.error(f"API Error: {response.text}")

else:
    st.info("Welcome! Please upload an audio file in the sidebar to begin moderation analysis.")
    st.image("https://images.unsplash.com/photo-1589254065878-42c9da997008?auto=format&fit=crop&w=1000&q=80", caption="VigilAudio ensures safety in audio streams.")
