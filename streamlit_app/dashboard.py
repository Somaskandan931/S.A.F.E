# Edited dashboard.py - Backend API Integration Version
# All model loading and prediction moved to FastAPI backend
# Frontend now lightweight - only UI + HTTP calls

import streamlit as st
import requests
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import time
from pathlib import Path
from datetime import datetime
from collections import deque
import io

st.set_page_config(page_title="S.A.F.E Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ========== API CONFIGURATION ==========
API_BASE = "http://localhost:8000"

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    padding: 1rem;
    background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.alert-critical {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
    animation: pulse 2s infinite;
}
.alert-warning {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
.alert-normal {
    background-color: #e8f5e9;
    border-left: 5px solid #4caf50;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4caf50 0%, #ff9800 50%, #f44336 100%);
}
</style>
""", unsafe_allow_html=True)

# ========== INITIALIZE SESSION STATE ==========
if 'models' not in st.session_state:
    st.session_state.models = []
if 'current_model_id' not in st.session_state:
    st.session_state.current_model_id = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = deque(maxlen=100)
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'framecount' not in st.session_state:
    st.session_state.framecount = 0
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = {'timestamps': [], 'scores': [], 'levels': []}

# ========== API HELPER FUNCTIONS ==========
@st.cache_data(ttl=300)
def fetch_models():
    """Fetch available models from backend API"""
    try:
        r = requests.get(f"{API_BASE}/api/models", timeout=5)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Backend error: {r.json().get('detail', 'Unknown error')}")
            return []
    except Exception as e:
        st.error(f"Backend not reachable: {e}")
        return []

def load_model(model_id):
    """Load model via backend API"""
    try:
        r = requests.post(f"{API_BASE}/api/models/{model_id}/load", timeout=10)
        if r.status_code == 200:
            st.success(r.json()["message"])
            return True
        else:
            st.error(r.json().get("detail", "Load failed"))
            return False
    except Exception as e:
        st.error(f"Load error: {e}")
        return False

def predict_image(file, model_id=None):
    """Predict via backend API"""
    try:
        files = {"file": file}
        params = {"model_id": model_id} if model_id else {}
        r = requests.post(f"{API_BASE}/api/predict/image", files=files, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Prediction error: {r.json().get('detail', 'Unknown')}")
            return None
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

# ========== SIDEBAR ==========
st.sidebar.markdown("🛡️ **S.A.F.E**")
st.sidebar.markdown("Signal-based Anomaly Flow Evaluation")

# Fetch and display models
st.sidebar.markdown("---")
st.sidebar.subheader("Model Selection")
models = fetch_models()

if models:
    model_options = {m['name']: m['model_id'] for m in models if m['available']}
    selected_model_name = st.sidebar.selectbox("Select Detection Model", list(model_options.keys()))
    selected_model_id = model_options[selected_model_name]
    
    if st.sidebar.button("Load Model", use_container_width=True):
        with st.spinner(f"Loading {selected_model_name}..."):
            if load_model(selected_model_id):
                st.session_state.current_model_id = selected_model_id
                st.sidebar.success(f"✅ {selected_model_name} loaded!")
            else:
                st.sidebar.error("❌ Failed to load model")
    
    if st.session_state.current_model_id:
        st.sidebar.success(f"🎯 Active: {selected_model_name}")
else:
    st.sidebar.warning("⚠️ No models available. Check backend.")

# Detection settings
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Settings")
risk_threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
frame_skip = st.sidebar.slider("Process Every N Frames", 1, 10, 3)
alert_cooldown = st.sidebar.number_input("Alert Cooldown (seconds)", 1, 60, 5)

# Navigation
page = st.sidebar.radio("Navigation", ["Overview", "Live Detection", "Analytics", "Alerts", "Model Insights"])

# ========== MAIN CONTENT ==========
if page == "Overview":
    st.markdown('<div class="main-header">S.A.F.E Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Real-time Crowd Anomaly Detection System")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Models", len(models))
    with col2:
        model_status = "Loaded" if st.session_state.current_model_id else "Not Loaded"
        st.metric("Model Status", model_status)
    with col3:
        st.metric("Alerts Generated", len(st.session_state.alerts))
    
    st.markdown("---")
    st.markdown("### Quick Start Guide")
    st.markdown("""
    1. **Load a Model**: Select and load from sidebar
    2. **Upload Video**: Go to Live Detection
    3. **Monitor**: Watch real-time risk scores
    4. **Review**: Check Alerts tab
    5. **Analyze**: Explore Analytics
    """)

elif page == "Live Detection":
    st.title("🔴 Live Video Analysis")
    
    if not st.session_state.current_model_id:
        st.warning("⚠️ Please load a model from the sidebar first")
    else:
        input_method = st.radio("Select Input Method", ["Upload Video File", "Use Webcam"], horizontal=True)
        
        if input_method == "Upload Video File":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
            
            if uploaded_file is not None:
                col1, col2 = st.columns([3, 1])
                with col1:
                    start_processing = st.button("Start Analysis", use_container_width=True)
                with col2:
                    stop_processing = st.button("Stop", use_container_width=True)
                
                if stop_processing:
                    st.session_state.processing = False
                
                if start_processing or st.session_state.processing:
                    st.session_state.processing = True
                    video_placeholder = st.empty()
                    metrics_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process video file frame by frame
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    videopath = tfile.name
                    
                    cap = cv2.VideoCapture(videopath)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frame_idx = 0
                    
                    try:
                        while cap.isOpened() and st.session_state.processing:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_idx += 1
                            if frame_idx % frame_skip == 0:
                                # Convert frame to bytes for API
                                _, buffer = cv2.imencode('.jpg', frame)
                                file_bytes = io.BytesIO(buffer)
                                
                                # API prediction
                                result = predict_image(file_bytes, st.session_state.current_model_id)
                                
                                if result:
                                    risk_score = result["risk_score"]
                                    risk_level = result["risk_level"]
                                    anomaly = result["anomaly"]
                                    
                                    current_time = datetime.now()
                                    st.session_state.anomaly_history['timestamps'].append(current_time)
                                    st.session_state.anomaly_history['scores'].append(risk_score)
                                    st.session_state.anomaly_history['levels'].append(risk_level)
                                    
                                    # Limit history
                                    if len(st.session_state.anomaly_history['timestamps']) > 100:
                                        st.session_state.anomaly_history['timestamps'].pop(0)
                                        st.session_state.anomaly_history['scores'].pop(0)
                                        st.session_state.anomaly_history['levels'].pop(0)
                                    
                                    # Create alert
                                    if risk_score > risk_threshold:
                                        # Simple cooldown check
                                        if not st.session_state.alerts or (time.time() - st.session_state.alerts[-1].get('timestamp', 0)) > alert_cooldown:
                                            alert = {
                                                'timestamp': current_time,
                                                'risk_level': risk_level,
                                                'risk_score': risk_score,
                                                'message': f"{risk_level} risk detected"
                                            }
                                            st.session_state.alerts.append(alert)
                                
                                # Display frame with overlay
                                annotated_frame = frame.copy()
                                color = (0, 255, 0) if risk_level == "NORMAL" else (0, 165, 255) if risk_level == "WARNING" else (0, 0, 255)
                                cv2.putText(annotated_frame, f"Risk: {risk_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                cv2.putText(annotated_frame, f"Score: {risk_score:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                
                                video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                                
                                # Metrics
                                with metrics_placeholder.container():
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Frame", f"{frame_idx}/{total_frames}")
                                    with col2:
                                        st.metric("Risk Level", risk_level)
                                    with col3:
                                        st.metric("Risk Score", f"{risk_score:.3f}")
                                    with col4:
                                        st.metric("Alerts", len(st.session_state.alerts))
                                
                                st.session_state.framecount += 1
                            
                            progress = frame_idx / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Processing... {progress*100:.1f}% complete")
                            time.sleep(1/fps * 2)  # Real-time speed
                        
                        status_text.success("✅ Video processing complete!")
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                    finally:
                        cap.release()
                        st.session_state.processing = False
        else:  # Webcam
            st.info("🖥️ Webcam mode - Click Start for real-time detection")
            run_webcam = st.button("Start Webcam", use_container_width=True)
            stop_webcam = st.button("Stop Webcam", use_container_width=True)
            
            if run_webcam:
                st.session_state.processing = True
            if stop_webcam:
                st.session_state.processing = False
            
            if st.session_state.processing:
                video_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                cap = cv2.VideoCapture(0)
                try:
                    while st.session_state.processing:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to access webcam")
                            break
                        
                        # API prediction for webcam frame
                        _, buffer = cv2.imencode('.jpg', frame)
                        file_bytes = io.BytesIO(buffer)
                        result = predict_image(file_bytes, st.session_state.current_model_id)
                        
                        if result:
                            risk_score = result["risk_score"]
                            risk_level = result["risk_level"]
                            
                            # Update history and alerts (same logic as above)
                            current_time = datetime.now()
                            st.session_state.anomaly_history['timestamps'].append(current_time)
                            st.session_state.anomaly_history['scores'].append(risk_score)
                            st.session_state.anomaly_history['levels'].append(risk_level)
                            
                            if len(st.session_state.anomaly_history['timestamps']) > 100:
                                st.session_state.anomaly_history['timestamps'].pop(0)
                                st.session_state.anomaly_history['scores'].pop(0)
                                st.session_state.anomaly_history['levels'].pop(0)
                            
                            if risk_score > risk_threshold and (not st.session_state.alerts or 
                                (time.time() - st.session_state.alerts[-1].get('timestamp', 0)) > alert_cooldown):
                                alert = {
                                    'timestamp': current_time,
                                    'risk_level': risk_level,
                                    'risk_score': risk_score,
                                    'message': f"{risk_level} risk detected"
                                }
                                st.session_state.alerts.append(alert)
                        
                        # Display with overlay
                        annotated_frame = frame.copy()
                        color = (0, 255, 0) if risk_level == "NORMAL" else (0, 165, 255) if risk_level == "WARNING" else (0, 0, 255)
                        cv2.putText(annotated_frame, f"Risk: {risk_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(annotated_frame, f"Score: {risk_score:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                        
                        with metrics_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Risk Level", risk_level)
                            with col2:
                                st.metric("Risk Score", f"{risk_score:.3f}")
                            with col3:
                                st.metric("Total Alerts", len(st.session_state.alerts))
                        
                        time.sleep(0.03)
                except Exception as e:
                    st.error(f"Webcam error: {e}")
                finally:
                    cap.release()

elif page == "Analytics":
    st.title("📊 Analytics Insights")
    
    if not st.session_state.anomaly_history['timestamps']:
        st.info("No data available yet. Process some video first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Level Distribution")
            level_counts = pd.Series(st.session_state.anomaly_history['levels']).value_counts()
            fig = px.pie(names=level_counts.index, values=level_counts.values, 
                        color=level_counts.index, color_discrete_map={
                            "CRITICAL": "#f44336", "WARNING": "#ff9800", "NORMAL": "#4caf50"
                        })
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Score Distribution")
            fig = px.histogram(x=st.session_state.anomaly_history['scores'], nbins=30, 
                             labels={'x': 'Risk Score', 'y': 'Frequency'})
            fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Critical")
            fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Warning")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Risk Score Over Time")
        df_history = pd.DataFrame({
            'Timestamp': st.session_state.anomaly_history['timestamps'],
            'Risk Score': st.session_state.anomaly_history['scores'],
            'Risk Level': st.session_state.anomaly_history['levels']
        })
        fig = px.line(df_history, x='Timestamp', y='Risk Score', color='Risk Level',
                     color_discrete_map={"CRITICAL": "#f44336", "WARNING": "#ff9800", "NORMAL": "#4caf50"})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)
        scores = st.session_state.anomaly_history['scores']
        with col1:
            st.metric("Mean Risk", f"{np.mean(scores):.3f}")
        with col2:
            st.metric("Max Risk", f"{np.max(scores):.3f}")
        with col3:
            st.metric("Min Risk", f"{np.min(scores):.3f}")
        with col4:
            st.metric("Std Dev", f"{np.std(scores):.3f}")

elif page == "Alerts":
    st.title("🚨 Alert Management")
    
    if not st.session_state.alerts:
        st.info("No alerts generated yet. Start processing video!")
    else:
        col1, col2, col3 = st.columns(3)
        critical_count = sum(1 for a in st.session_state.alerts if a['risk_level'] == 'CRITICAL')
        warning_count = sum(1 for a in st.session_state.alerts if a['risk_level'] == 'WARNING')
        with col1:
            st.metric("Critical Alerts", critical_count)
        with col2:
            st.metric("Warnings", warning_count)
        with col3:
            st.metric("Normal", len(st.session_state.alerts) - critical_count - warning_count)
        
        st.markdown("---")
        filter_level = st.multiselect("Filter by Risk Level", ['CRITICAL', 'WARNING', 'NORMAL'], 
                                    default=['CRITICAL', 'WARNING'])
        
        st.subheader(f"Alert Log ({len(st.session_state.alerts)} total)")
        for idx, alert in enumerate(reversed(list(st.session_state.alerts))):
            if alert['risk_level'] in filter_level:
                with st.expander(f"Alert #{len(st.session_state.alerts) - idx} - {alert['risk_level']} - {alert['timestamp'].strftime('%H:%M:%S')}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Time**: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown(f"**Risk Level**: {alert['risk_level']}")
                        st.markdown(f"**Risk Score**: {alert['risk_score']:.4f}")
                        st.markdown(f"**Message**: {alert['message']}")
                    
                    with col2:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=alert['risk_score'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 1]},
                                'bar': {'color': "#f44336" if alert['risk_level'] == 'CRITICAL' 
                                        else "#ff9800" if alert['risk_level'] == 'WARNING' else "#4caf50"},
                                'steps': [
                                    {'range': [0, 0.4], 'color': "lightgreen"},
                                    {'range': [0.4, 0.7], 'color': "yellow"},
                                    {'range': [0.7, 1], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': risk_threshold
                                }
                            }
                        ))
                        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        if st.button("Export Alerts to CSV"):
            df_alerts = pd.DataFrame(list(st.session_state.alerts))
            csv = df_alerts.to_csv(index=False)
            st.download_button("Download CSV", csv, "alerts.csv", "text/csv")

elif page == "Model Insights":
    st.title("🔍 Model Insights & Explainability")
    
    if not st.session_state.current_model_id:
        st.warning("Please load a model first")
    else:
        st.subheader("Active Model Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Details**")
            # Show current model info
            st.markdown(f"- **ID**: {st.session_state.current_model_id}")
            st.markdown(f"- **Status**: ✅ Loaded & Active")
            st.markdown(f"- **Frames Analyzed**: {st.session_state.framecount}")
        
        with col2:
            if st.session_state.framecount > 0:
                detection_rate = len(st.session_state.alerts) / st.session_state.framecount * 100
                st.metric("Detection Rate", f"{detection_rate:.2f}%")
        
        st.markdown("---")
        st.markdown("**How Detection Works**")
        st.markdown("""
        1. **Frame Capture** - Extract visual features (density, motion, intensity)
        2. **API Prediction** - Send to backend ML model
        3. **Risk Scoring** - 0.0 (normal) to 1.0 (critical)
        4. **Alert Generation** - Trigger if above threshold
        """)
        
        st.markdown("**Active Features**")
        st.markdown("- **Density**: Edge density from Canny detection")
        st.markdown("- **Speed**: Motion intensity between frames")
        st.markdown("- **Direction Variance**: Movement pattern consistency")
        st.markdown("- **Intensity**: Frame brightness patterns")
