"""
S.A.F.E Streamlit Dashboard - COMPLETE VERSION
Signal-based Anomaly & Flow Evaluation System
Production-Ready Interface for Video-Based Anomaly Detection
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import tempfile
import time
from pathlib import Path
from datetime import datetime
from collections import deque
import io

# Page config
st.set_page_config(
    page_title="S.A.F.E Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = deque(maxlen=100)
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = {'timestamps': [], 'scores': [], 'levels': []}

# Model paths
MODELS_PATH = Path("C:/Users/somas/PycharmProjects/S.A.F.E/scripts/models")

AVAILABLE_MODELS = {
    'Simple Threshold (Demo)': None,  # Built-in demo model
    'Isolation Forest': MODELS_PATH / 'isolation_forest.pkl',
    'LSTM Autoencoder': MODELS_PATH / 'lstm_autoencoder.pkl',
    'Moving Average Detector': MODELS_PATH / 'mad.pkl',
    'One-Class SVM': MODELS_PATH / 'oneclass_svm.pkl',
    'Z-Score Detector': MODELS_PATH / 'zscore.pkl'
}

# Simple demo model class
class SimpleThresholdModel:
    """Simple threshold-based model that works without training"""
    def __init__(self):
        self.name = "SimpleThreshold"
        self.is_fitted = True

    def predict_score(self, X):
        """Calculate simple anomaly score based on thresholds"""
        # X shape: (n_samples, 5) - [density, speed_mean, direction_variance, mean_intensity, std_intensity]
        scores = []
        for sample in X:
            density, speed, direction_var, mean_int, std_int = sample

            # Simple scoring logic
            score = 0.0

            # High density contributes to risk
            if density > 50:
                score += 0.3

            # Low speed indicates congestion
            if speed < 5:
                score += 0.2

            # High direction variance indicates chaos
            if direction_var > 20:
                score += 0.3

            # Unusual intensity patterns
            if mean_int < 50 or mean_int > 200:
                score += 0.1

            if std_int > 50:
                score += 0.1

            scores.append(min(score, 1.0))

        return np.array(scores)

    def predict(self, X):
        """Predict binary anomaly labels"""
        scores = self.predict_score(X)
        return (scores > 0.5).astype(int)

# Helper functions
@st.cache_resource
def load_model(model_path):
    """Load trained model from disk with fallback handling"""
    try:
        # Try direct pickle load
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except ModuleNotFoundError as e:
        st.error(f"Model dependency error: {e}")
        st.info("💡 Models were trained with 'src' module. Loading with compatibility mode...")

        # Try with compatibility - create mock src module
        import sys
        import types

        # Create mock modules
        if 'src' not in sys.modules:
            sys.modules['src'] = types.ModuleType('src')
            sys.modules['src.models'] = types.ModuleType('src.models')
            sys.modules['src.models.base_model'] = types.ModuleType('src.models.base_model')

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            st.success("✅ Model loaded in compatibility mode")
            return model
        except Exception as e2:
            st.error(f"Failed to load model: {e2}")
            st.warning("⚠️ Please retrain models or use alternative loading method")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_features_from_frame(frame, prev_frame=None):
    """Extract basic features from video frame for anomaly detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    features = {
        'mean_intensity': np.mean(gray),
        'std_intensity': np.std(gray),
        'edge_density': np.sum(cv2.Canny(gray, 100, 200)) / gray.size,
    }

    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        features['motion_intensity'] = np.mean(diff)
        features['motion_variance'] = np.std(diff)
    else:
        features['motion_intensity'] = 0
        features['motion_variance'] = 0

    features['density'] = features['edge_density'] * 100
    features['speed_mean'] = features['motion_intensity'] / 10
    features['direction_variance'] = features['motion_variance']

    return features

def calculate_risk_score(features, model, threshold=0.5):
    """Calculate risk score from features"""
    try:
        feature_vector = np.array([[
            features['density'],
            features['speed_mean'],
            features['direction_variance'],
            features['mean_intensity'],
            features['std_intensity']
        ]])

        if hasattr(model, 'predict_score'):
            score = model.predict_score(feature_vector)[0]
        elif hasattr(model, 'decision_function'):
            score = -model.decision_function(feature_vector)[0]
            score = (score - score.min()) / (score.max() - score.min() + 1e-8)
        else:
            prediction = model.predict(feature_vector)[0]
            score = 1.0 if prediction == -1 else 0.0

        score = np.clip(score, 0, 1)

        if score >= 0.7:
            risk_level = "CRITICAL"
        elif score >= 0.4:
            risk_level = "WARNING"
        else:
            risk_level = "NORMAL"

        return score, risk_level
    except Exception as e:
        st.warning(f"Risk calculation error: {e}")
        return 0.0, "NORMAL"

def process_video_frame(frame, model, prev_frame=None):
    """Process single video frame"""
    features = extract_features_from_frame(frame, prev_frame)
    risk_score, risk_level = calculate_risk_score(features, model)
    annotated_frame = frame.copy()

    color = (0, 255, 0) if risk_level == "NORMAL" else (0, 165, 255) if risk_level == "WARNING" else (0, 0, 255)
    cv2.putText(annotated_frame, f"Risk: {risk_level}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(annotated_frame, f"Score: {risk_score:.3f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if 'motion_intensity' in features:
        motion_text = f"Motion: {features['motion_intensity']:.2f}"
        cv2.putText(annotated_frame, motion_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated_frame, risk_score, risk_level, features

def create_alert(risk_level, risk_score, features, timestamp):
    """Create alert entry"""
    return {
        'timestamp': timestamp,
        'risk_level': risk_level,
        'risk_score': risk_score,
        'density': features.get('density', 0),
        'motion': features.get('motion_intensity', 0),
        'message': f"{risk_level} risk detected - Score: {risk_score:.3f}"
    }

# Sidebar
st.sidebar.markdown("# 🛡️ S.A.F.E")
st.sidebar.markdown("### Signal-based Anomaly & Flow Evaluation")
st.sidebar.markdown("---")

# Model selection
st.sidebar.markdown("### 🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Detection Model",
    list(AVAILABLE_MODELS.keys()),
    help="Choose the AI model for anomaly detection"
)

if st.sidebar.button("🔄 Load Model", use_container_width=True):
    with st.spinner(f"Loading {model_choice}..."):
        model_path = AVAILABLE_MODELS[model_choice]

        # Check if it's the demo model
        if model_path is None:
            model = SimpleThresholdModel()
            st.session_state.current_model = model
            st.session_state.models_loaded = True
            st.sidebar.success(f"✅ {model_choice} loaded!")
        else:
            model = load_model(model_path)
            if model is not None:
                st.session_state.current_model = model
                st.session_state.models_loaded = True
                st.sidebar.success(f"✅ {model_choice} loaded!")
            else:
                st.sidebar.error("❌ Failed to load model")
                st.sidebar.info("💡 Try using 'Simple Threshold (Demo)' model")

if st.session_state.models_loaded:
    st.sidebar.success(f"✅ Model Active: {model_choice}")

st.sidebar.markdown("---")

# Detection settings
st.sidebar.markdown("### ⚙️ Detection Settings")
risk_threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
frame_skip = st.sidebar.slider("Process Every N Frames", 1, 10, 3)
alert_cooldown = st.sidebar.number_input("Alert Cooldown (seconds)", 1, 60, 5)

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🎥 Live Detection", "📊 Analytics", "🚨 Alerts", "🧠 Model Insights"],
    help="Navigate between dashboard sections"
)

# Main content
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🛡️ S.A.F.E Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Real-time Crowd Anomaly Detection System")

    if not st.session_state.models_loaded:
        st.info("👈 Please load a model from the sidebar to begin")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Available Models", len(AVAILABLE_MODELS))
        with col2:
            st.metric("Model Status", "Not Loaded" if not st.session_state.models_loaded else "Loaded")
        with col3:
            st.metric("Alerts Generated", len(st.session_state.alerts))

        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **Load a Model**: Select and load a detection model from the sidebar
        2. **Upload Video**: Go to 🎥 Live Detection and upload a video file
        3. **Monitor**: Watch real-time anomaly detection and risk scores
        4. **Review**: Check 🚨 Alerts for detected incidents
        5. **Analyze**: Explore 📊 Analytics for patterns and trends
        """)

        st.markdown("---")
        st.markdown("### 📚 Available Models")

        models_df = pd.DataFrame({
            'Model': list(AVAILABLE_MODELS.keys()),
            'Type': ['Demo/Baseline', 'Ensemble', 'Deep Learning', 'Statistical', 'ML', 'Statistical'],
            'Use Case': [
                'Quick demo without training',
                'Best overall performance',
                'Complex temporal patterns',
                'Quick anomalies detection',
                'Robust outlier detection',
                'Fast statistical analysis'
            ]
        })
        st.dataframe(models_df, use_container_width=True)
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Frames Processed", st.session_state.frame_count)
        with col2:
            critical_alerts = sum(1 for a in st.session_state.alerts if a['risk_level'] == 'CRITICAL')
            st.metric("Critical Alerts", critical_alerts)
        with col3:
            warning_alerts = sum(1 for a in st.session_state.alerts if a['risk_level'] == 'WARNING')
            st.metric("Warnings", warning_alerts)
        with col4:
            avg_risk = np.mean([a['risk_score'] for a in st.session_state.alerts]) if st.session_state.alerts else 0
            st.metric("Avg Risk Score", f"{avg_risk:.3f}")

        if st.session_state.anomaly_history['timestamps']:
            st.markdown("---")
            st.subheader("📈 Risk Score Trend")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.anomaly_history['timestamps'],
                y=st.session_state.anomaly_history['scores'],
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#ff9800', width=2),
                marker=dict(size=6)
            ))

            fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Critical")
            fig.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Warning")

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Risk Score",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🎥 Live Detection":
    st.title("🎥 Live Video Analysis")

    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load a model from the sidebar first")
    else:
        input_method = st.radio("Select Input Method", ["Upload Video File", "Use Webcam"], horizontal=True)

        if input_method == "Upload Video File":
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])

            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name

                col1, col2 = st.columns([3, 1])

                with col1:
                    start_processing = st.button("▶️ Start Analysis", use_container_width=True)
                with col2:
                    stop_processing = st.button("⏹️ Stop", use_container_width=True)

                if stop_processing:
                    st.session_state.processing = False

                if start_processing:
                    st.session_state.processing = True

                    video_placeholder = st.empty()
                    metrics_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    frame_idx = 0
                    prev_frame = None
                    last_alert_time = 0

                    try:
                        while cap.isOpened() and st.session_state.processing:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_idx += 1

                            if frame_idx % frame_skip == 0:
                                annotated_frame, risk_score, risk_level, features = process_video_frame(
                                    frame, st.session_state.current_model, prev_frame
                                )

                                current_time = datetime.now()
                                st.session_state.anomaly_history['timestamps'].append(current_time)
                                st.session_state.anomaly_history['scores'].append(risk_score)
                                st.session_state.anomaly_history['levels'].append(risk_level)

                                if len(st.session_state.anomaly_history['timestamps']) > 100:
                                    st.session_state.anomaly_history['timestamps'].pop(0)
                                    st.session_state.anomaly_history['scores'].pop(0)
                                    st.session_state.anomaly_history['levels'].pop(0)

                                if risk_score >= risk_threshold:
                                    time_since_alert = time.time() - last_alert_time
                                    if time_since_alert >= alert_cooldown:
                                        alert = create_alert(risk_level, risk_score, features, current_time)
                                        st.session_state.alerts.append(alert)
                                        last_alert_time = time.time()

                                video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

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

                                prev_frame = frame
                                st.session_state.frame_count += 1

                            progress = frame_idx / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {progress*100:.1f}% complete")

                            time.sleep(1 / (fps * 2))

                        status_text.success("✅ Video processing complete!")

                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                    finally:
                        cap.release()
                        st.session_state.processing = False

        else:
            st.info("📷 Webcam mode - Click 'Start' to begin real-time detection")

            run_webcam = st.button("🎥 Start Webcam", use_container_width=True)
            stop_webcam = st.button("⏹️ Stop Webcam", use_container_width=True)

            if run_webcam:
                st.session_state.processing = True

            if stop_webcam:
                st.session_state.processing = False

            if st.session_state.processing:
                video_placeholder = st.empty()
                metrics_placeholder = st.empty()

                cap = cv2.VideoCapture(0)
                prev_frame = None
                last_alert_time = 0

                try:
                    while st.session_state.processing:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to access webcam")
                            break

                        annotated_frame, risk_score, risk_level, features = process_video_frame(
                            frame, st.session_state.current_model, prev_frame
                        )

                        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                        with metrics_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Risk Level", risk_level)
                            with col2:
                                st.metric("Risk Score", f"{risk_score:.3f}")
                            with col3:
                                st.metric("Total Alerts", len(st.session_state.alerts))

                        if risk_score >= risk_threshold:
                            time_since_alert = time.time() - last_alert_time
                            if time_since_alert >= alert_cooldown:
                                alert = create_alert(risk_level, risk_score, features, datetime.now())
                                st.session_state.alerts.append(alert)
                                last_alert_time = time.time()

                        prev_frame = frame
                        time.sleep(0.03)

                except Exception as e:
                    st.error(f"Webcam error: {e}")
                finally:
                    cap.release()
                    st.session_state.processing = False

elif page == "📊 Analytics":
    st.title("📊 Analytics & Insights")

    if not st.session_state.anomaly_history['timestamps']:
        st.info("No data available yet. Process some video first!")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔴 Risk Level Distribution")
            level_counts = pd.Series(st.session_state.anomaly_history['levels']).value_counts()

            fig = px.pie(
                names=level_counts.index,
                values=level_counts.values,
                color=level_counts.index,
                color_discrete_map={
                    'CRITICAL': '#f44336',
                    'WARNING': '#ff9800',
                    'NORMAL': '#4caf50'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 Score Distribution")
            fig = px.histogram(
                x=st.session_state.anomaly_history['scores'],
                nbins=30,
                labels={'x': 'Risk Score', 'y': 'Frequency'}
            )
            fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Critical")
            fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Warning")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("⏱️ Risk Score Over Time")

        df_history = pd.DataFrame({
            'Timestamp': st.session_state.anomaly_history['timestamps'],
            'Risk Score': st.session_state.anomaly_history['scores'],
            'Risk Level': st.session_state.anomaly_history['levels']
        })

        fig = px.line(df_history, x='Timestamp', y='Risk Score', color='Risk Level',
                      color_discrete_map={
                          'CRITICAL': '#f44336',
                          'WARNING': '#ff9800',
                          'NORMAL': '#4caf50'
                      })
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Statistical Summary")

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

elif page == "🚨 Alerts":
    st.title("🚨 Alert Management")

    if not st.session_state.alerts:
        st.info("No alerts generated yet. Start processing video to generate alerts!")
    else:
        col1, col2, col3 = st.columns(3)

        critical_count = sum(1 for a in st.session_state.alerts if a['risk_level'] == 'CRITICAL')
        warning_count = sum(1 for a in st.session_state.alerts if a['risk_level'] == 'WARNING')
        normal_count = len(st.session_state.alerts) - critical_count - warning_count

        with col1:
            st.metric("🔴 Critical Alerts", critical_count)
        with col2:
            st.metric("🟡 Warnings", warning_count)
        with col3:
            st.metric("🟢 Normal", normal_count)

        st.markdown("---")

        filter_level = st.multiselect(
            "Filter by Risk Level",
            ['CRITICAL', 'WARNING', 'NORMAL'],
            default=['CRITICAL', 'WARNING']
        )

        st.subheader(f"📋 Alert Log ({len(st.session_state.alerts)} total)")

        for idx, alert in enumerate(reversed(list(st.session_state.alerts))):
            if alert['risk_level'] in filter_level:
                with st.expander(f"Alert #{len(st.session_state.alerts) - idx} - {alert['risk_level']} - {alert['timestamp'].strftime('%H:%M:%S')}"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown(f"**Risk Level:** {alert['risk_level']}")
                        st.markdown(f"**Risk Score:** {alert['risk_score']:.4f}")
                        st.markdown(f"**Message:** {alert['message']}")

                        if 'density' in alert:
                            st.markdown(f"**Density:** {alert['density']:.2f}")
                        if 'motion' in alert:
                            st.markdown(f"**Motion Intensity:** {alert['motion']:.2f}")

                    with col2:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=alert['risk_score'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 1]},
                                'bar': {'color': '#f44336' if alert['risk_level'] == 'CRITICAL' else '#ff9800' if alert['risk_level'] == 'WARNING' else '#4caf50'},
                                'steps': [
                                    {'range': [0, 0.4], 'color': 'lightgreen'},
                                    {'range': [0.4, 0.7], 'color': 'yellow'},
                                    {'range': [0.7, 1], 'color': 'lightcoral'}
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
        if st.button("📥 Export Alerts to CSV"):
            df_alerts = pd.DataFrame(list(st.session_state.alerts))
            csv = df_alerts.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "safe_alerts.csv",
                "text/csv",
                key='download-csv'
            )

elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights & Explainability")

    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load a model first")
    else:
        st.subheader(f"Active Model: {model_choice}")

        model = st.session_state.current_model

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Model Details")
            st.markdown(f"**Type:** {type(model).__name__}")
            st.markdown(f"**Status:** {'Trained' if hasattr(model, 'is_fitted') and model.is_fitted else 'Ready'}")

            if hasattr(model, 'feature_names'):
                st.markdown(f"**Features:** {len(model.feature_names)}")

        with col2:
            st.markdown("### Performance Stats")
            if st.session_state.frame_count > 0:
                st.metric("Frames Analyzed", st.session_state.frame_count)
                detection_rate = len(st.session_state.alerts) / st.session_state.frame_count * 100
                st.metric("Detection Rate", f"{detection_rate:.2f}%")

        st.markdown("---")
        st.subheader("📊 Feature Analysis")

        if st.session_state.anomaly_history['timestamps']:
            st.markdown("**Active Features in Detection:**")
            st.markdown("""
            - **Density**: Edge density from frame analysis
            - **Speed Mean**: Average motion intensity
            - **Direction Variance**: Movement pattern consistency
            - **Mean Intensity**: Overall frame brightness
            - **Std Intensity**: Frame brightness variation
            """)

            st.markdown("---")
            st.markdown("**How Anomalies Are Detected:**")
            st.markdown("""
            1. Extract visual features from each frame
            2. Calculate motion patterns compared to previous frames
            3. Score anomaly using trained ML model
            4. Classify risk level based on threshold
            5. Generate alert if risk exceeds configured threshold
            """)

        if hasattr(model, 'get_feature_importance'):
            st.markdown("---")
            st.subheader("🎯 Feature Importance")
            importance = model.get_feature_importance()
            if importance:
                df_importance = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                }).sort_values('Importance', ascending=False)

                fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
                            title="Feature Importance Ranking")
                st.plotly_chart(fig, use_container_width=True)