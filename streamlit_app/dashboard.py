# Updated S.A.F.E Dashboard - Video-Based Signal Analytics
# Integrated with FastAPI Backend + Ethical Crowd Monitoring

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

# ========== VIDEO PROCESSING STATES ==========
NORMAL = "NORMAL"
CAMERA_SHAKE = "CAMERA_SHAKE"
END = "END"


st.set_page_config(
    page_title="S.A.F.E Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== API CONFIGURATION ==========
API_BASE = "http://localhost:8000"

# ========== CUSTOM CSS ==========
st.markdown( """
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
.ethics-box {
    background-color: #e3f2fd;
    border-left: 5px solid #2196f3;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True )

# ========== INITIALIZE SESSION STATE ==========
if 'models' not in st.session_state :
    st.session_state.models = []
if 'current_model_id' not in st.session_state :
    st.session_state.current_model_id = None
if 'alerts' not in st.session_state :
    st.session_state.alerts = deque( maxlen=100 )
if 'processing' not in st.session_state :
    st.session_state.processing = False
if 'monitoring_active' not in st.session_state :
    st.session_state.monitoring_active = False
if 'framecount' not in st.session_state :
    st.session_state.framecount = 0
if 'anomaly_history' not in st.session_state :
    st.session_state.anomaly_history = {'timestamps' : [], 'scores' : [], 'levels' : []}
if 'zone_history' not in st.session_state :
    st.session_state.zone_history = {f"Zone {i}" : {"signals" : [], "alerts" : []} for i in range( 1, 7 )}


# ========== VIDEO UTILITIES ==========
import cv2
import numpy as np
import tempfile

# -----------------------------
# Video Loading
# -----------------------------
def load_video(uploaded_video):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())
    temp_file.flush()

    cap = cv2.VideoCapture(temp_file.name)

    if not cap.isOpened():
        raise RuntimeError("Failed to open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    return cap, total_frames, fps


# -----------------------------
# Zone Grid Functions
# -----------------------------
def draw_zone_grid(frame, rows=2, cols=3):
    """Draw zone grid overlay on a frame."""
    if frame is None:
        raise ValueError("Frame is None, cannot draw grid.")

    h, w, _ = frame.shape
    zone_id = 1

    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * w / cols), int(r * h / rows)
            x2, y2 = int((c + 1) * w / cols), int((r + 1) * h / rows)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Zone {zone_id}", (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            zone_id += 1
    return frame

def split_into_zones(frame, rows=2, cols=3):
    """Split frame into spatial zones."""
    if frame is None:
        return {}

    h, w, _ = frame.shape
    zones = {}
    zone_id = 1

    for r in range(rows):
        for c in range(cols):
            y1, y2 = int(r * h / rows), int((r + 1) * h / rows)
            x1, x2 = int(c * w / cols), int((c + 1) * w / cols)
            zones[f"Zone {zone_id}"] = frame[y1:y2, x1:x2]
            zone_id += 1

    return zones

# -----------------------------
# Optical Flow & Signals
# -----------------------------
def compute_zone_signals(prev_gray, curr_gray, zone_id):
    """Compute crowd motion signals for a zone using optical flow."""
    if prev_gray is None or curr_gray is None:
        return None, None

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.4, levels=2, winsize=12,
        iterations=2, poly_n=3, poly_sigma=1.1, flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    signal = {
        "zone_id": zone_id,
        "density": float(np.mean(mag) * 100),
        "speed_mean": float(np.mean(mag)),
        "speed_variance": float(np.var(mag)),
        "direction_variance": float(np.var(ang))
    }

    return signal, flow


def detect_panic_signature ( signal ) :
    """Detect panic-like crowd behavior based on density, speed, and directional chaos."""
    if signal is None :
        return False

    # More sensitive thresholds
    high_density = signal["density"] > 50  # Lowered from 70
    low_speed = signal["speed_mean"] < 2.0  # Increased from 1.5
    high_chaos = signal["direction_variance"] > 1.5  # Lowered from 2.0

    return high_density and low_speed and high_chaos

# -----------------------------
# Optical Flow Visualization
# -----------------------------
def draw_optical_flow_trails(flow, frame, trail_layer, step=16):
    """Draw optical flow arrows on frame."""
    if flow is None or frame is None:
        return frame, trail_layer
    h, w = flow.shape[:2]  # 🔥 USE FLOW SIZE, NOT FRAME

    y, x = np.mgrid[
           step // 2 : h : step,
           step // 2 : w : step
           ].reshape( 2, -1 )

    fx, fy = flow[y, x].T

    for (x1, y1, dx, dy) in zip(x, y, fx, fy):
        x2 = int(x1 + dx)
        y2 = int(y1 + dy)
        mag = np.sqrt(dx*dx + dy*dy)

        if mag > 4:
            color = (0, 0, 255)      # Red = congestion
            thickness = 2
        elif mag > 2:
            color = (0, 255, 255)    # Yellow = slowing
            thickness = 2
        elif mag > 1.5:
            color = (0, 255, 0)      # Green = smooth
            thickness = 1
        else:
            continue  # ignore noise

        cv2.line(trail_layer, (x1, y1), (x2, y2), color, thickness)

    blended = cv2.addWeighted(frame, 0.7, trail_layer, 0.9, 0)
    return blended, trail_layer

def draw_flow_legend(frame):
    """Overlay legend for flow visualization."""
    cv2.rectangle(frame, (10, 10), (260, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (260, 120), (255, 255, 255), 1)

    cv2.putText(frame, "Crowd Flow Legend", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Green  : Smooth Flow", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Yellow : Slowing", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, "Red    : Congestion", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame

# -----------------------------
# Example Safe Video Loop
# -----------------------------
def process_video(uploaded_video):
    cap, frame, fps= load_video(uploaded_video)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    trail_layer = np.zeros_like(frame)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zones = split_into_zones(frame)
        flows_for_visualization = []

        for zid, zframe in zones.items():
            zgray = cv2.cvtColor(zframe, cv2.COLOR_BGR2GRAY)
            signal, flow = compute_zone_signals(prev_gray, zgray, zid)

            if signal and detect_panic_signature(signal):
                print(f"Panic detected in {zid}")

            frame, trail_layer = draw_optical_flow_trails(flow, frame, trail_layer)

        frame = draw_zone_grid(frame)
        frame = draw_flow_legend(frame)
        prev_gray = gray

        # display or save frame
        # cv2.imshow("Processed Video", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()


# ========== API HELPER FUNCTIONS ==========
@st.cache_data( ttl=300 )
def fetch_models () :
    """Fetch available models from backend API"""
    try :
        r = requests.get( f"{API_BASE}/api/models", timeout=5 )
        if r.status_code == 200 :
            return r.json()
        else :
            st.error( f"Backend error: {r.json().get( 'detail', 'Unknown error' )}" )
            return []
    except Exception as e :
        st.error( f"⚠️ Backend not reachable: {e}" )
        return []


def load_model ( model_id ) :
    """Load model via backend API"""
    try :
        r = requests.post( f"{API_BASE}/api/models/{model_id}/load", timeout=10 )
        if r.status_code == 200 :
            st.success( r.json()["message"] )
            return True
        else :
            st.error( r.json().get( "detail", "Load failed" ) )
            return False
    except Exception as e :
        st.error( f"Load error: {e}" )
        return False


def predict_image ( file_bytes, model_id=None ) :
    """Predict via backend API with timeout"""
    try :
        files = {"file" : ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"model_id" : model_id} if model_id else {}

        # ✅ ADD TIMEOUT - fail fast if backend is stuck
        r = requests.post(
            f"{API_BASE}/api/predict/image",
            files=files,
            params=params,
            timeout=3  # ✅ 3 second timeout instead of 30
        )

        if r.status_code == 200 :
            return r.json()
        else :
            print( f"⚠️ API error: {r.status_code}" )
            return None

    except requests.Timeout :
        print( "⚠️ API timeout - skipping this frame" )
        return None
    except requests.ConnectionError :
        print( "⚠️ Backend not reachable - skipping prediction" )
        return None
    except Exception as e :
        print( f"⚠️ Prediction failed: {e}" )
        return None


def monitor_zone_signal ( signal_data ) :
    """Store and monitor zone signal"""
    zone_id = signal_data["zone_id"]
    st.session_state.zone_history[zone_id]["signals"].append( signal_data )

    # Keep last 100 signals per zone
    if len( st.session_state.zone_history[zone_id]["signals"] ) > 100 :
        st.session_state.zone_history[zone_id]["signals"].pop( 0 )

def detect_camera_shake(prev_gray, curr_gray, threshold=0.25):  # ✅ Changed from 2.0 to 0.25
    """
    Detect global camera shake using average optical flow magnitude.
    Returns:
        is_shake (bool), global_motion (float)
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    global_motion = float(np.mean(mag))

    # ✅ DEBUG: Print actual values
    print(f"🔍 Global motion: {global_motion:.3f}, Threshold: {threshold}, Shake: {global_motion > threshold}")

    is_shake = global_motion > threshold
    return is_shake, global_motion

def detect_signal_anomaly(signal, history, window=10):
    if len(history) < window:
        return False, None

    recent = history[-window:]

    mean_density = np.mean([s["density"] for s in recent])
    mean_speed = np.mean([s["speed_mean"] for s in recent])

    if signal["density"] > mean_density * 1.8:
        return True, "DENSITY_SPIKE"

    if signal["speed_mean"] < mean_speed * 0.4:
        return True, "SPEED_DROP"

    if signal["direction_variance"] > 0.7:
        return True, "DIRECTION_CHAOS"

    return False, None

# ========== VISUALIZATION FUNCTIONS ==========
def create_animated_phase_space ( zone_history ) :
    """Create phase-space plot of density vs speed"""
    all_signals = []
    for zone_id, data in zone_history.items() :
        for signal in data["signals"] :
            signal_copy = signal.copy()
            signal_copy["zone"] = zone_id
            all_signals.append( signal_copy )

    if not all_signals :
        return go.Figure()

    df = pd.DataFrame( all_signals )
    df["t"] = range( len( df ) )

    fig = px.scatter(
        df, x="density", y="speed_mean",
        animation_frame="t",
        size="density",
        color="direction_variance",
        hover_data=["zone"],
        range_x=[0, 100], range_y=[0, 10],
        title="Crowd Escalation Phase Space",
        labels={"density" : "Crowd Density (%)", "speed_mean" : "Average Speed"}
    )
    return fig


def create_zone_heatmap ( zone_history ) :
    """Create heatmap of zone risk levels"""
    zones = list( zone_history.keys() )
    latest_densities = []

    for zone in zones :
        signals = zone_history[zone]["signals"]
        if signals :
            latest_densities.append( signals[-1]["density"] )
        else :
            latest_densities.append( 0 )

    # Reshape for 2x3 grid
    heatmap_data = np.array( latest_densities ).reshape( 2, 3 )

    fig = go.Figure( data=go.Heatmap(
        z=heatmap_data,
        x=["Col 1", "Col 2", "Col 3"],
        y=["Row 1", "Row 2"],
        colorscale="RdYlGn_r",
        text=[[f"Zone {i + j * 3 + 1}" for j in range( 3 )] for i in range( 2 )],
        texttemplate="%{text}<br>%{z:.1f}%",
        textfont={"size" : 14},
        colorbar=dict( title="Density %" )
    ) )
    fig.update_layout( title="Live Zone Density Heatmap", height=400 )
    return fig

def draw_camera_shake_warning(frame, shake_score):
    h, w, _ = frame.shape

    cv2.rectangle(frame, (w - 340, 10), (w - 10, 70), (0, 0, 255), -1)
    cv2.putText(frame, "CAMERA SHAKE DETECTED",
                (w - 330, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.putText(frame, f"Shake Intensity: {shake_score:.2f}",
                (w - 330, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    return frame


# ========== SIDEBAR ==========
st.sidebar.markdown( "🛡️ **S.A.F.E**" )
st.sidebar.markdown( "Signal-based Anomaly Flow Evaluation" )

st.sidebar.markdown( "---" )
st.sidebar.subheader( "Model Selection" )
models = fetch_models()

if models :
    model_options = {m['name'] : m['model_id'] for m in models if m['available']}
    selected_model_name = st.sidebar.selectbox( "Select Detection Model", list( model_options.keys() ) )
    selected_model_id = model_options[selected_model_name]

    if st.sidebar.button( "Load Model", use_container_width=True ) :
        with st.spinner( f"Loading {selected_model_name}..." ) :
            if load_model( selected_model_id ) :
                st.session_state.current_model_id = selected_model_id
                st.sidebar.success( f"✅ {selected_model_name} loaded!" )
            else :
                st.sidebar.error( "❌ Failed to load model" )

    if st.session_state.current_model_id :
        st.sidebar.success( f"🎯 Active: {selected_model_name}" )
else :
    st.sidebar.warning( "⚠️ No models available. Check backend." )

st.sidebar.markdown( "---" )
st.sidebar.subheader( "Detection Settings" )
risk_threshold = st.sidebar.slider( "Risk Threshold", 0.0, 1.0, 0.5, 0.05 )
frame_skip = st.sidebar.slider( "Process Every N Frames", 1, 10, 5 )
alert_cooldown = st.sidebar.number_input( "Alert Cooldown (seconds)", 1, 60, 5 )

page = st.sidebar.radio( "Navigation", [
    "Overview",
    "Live Video Monitoring",
    "Signal Analytics",
    "Alerts",
    "Ethics & Privacy"
] )

# ========== MAIN CONTENT ==========
if page == "Overview" :
    st.markdown( '<div class="main-header">S.A.F.E Dashboard</div>', unsafe_allow_html=True )
    st.markdown( "**Real-time Crowd Anomaly Detection System**" )
    st.markdown( "*Signal-based • Privacy-preserving • Human-in-the-loop*" )

    col1, col2, col3 = st.columns( 3 )
    with col1 :
        st.metric( "Available Models", len( models ) )
    with col2 :
        model_status = "Loaded" if st.session_state.current_model_id else "Not Loaded"
        st.metric( "Model Status", model_status )
    with col3 :
        st.metric( "Total Alerts", len( st.session_state.alerts ) )

    st.markdown( "---" )
    st.markdown( "### System Architecture" )
    col1, col2 = st.columns( 2 )

    with col1 :
        st.markdown( "**Frontend (Streamlit)**" )
        st.markdown( "- Video upload & processing" )
        st.markdown( "- Zone-based signal extraction" )
        st.markdown( "- Real-time visualization" )
        st.markdown( "- Alert management" )

    with col2 :
        st.markdown( "**Backend (FastAPI)**" )
        st.markdown( "- ML model serving" )
        st.markdown( "- Anomaly detection" )
        st.markdown( "- Risk scoring" )
        st.markdown( "- RESTful API" )

    st.markdown( "---" )
    st.markdown( "### Quick Start Guide" )
    st.markdown( """
    1. **Load Model**: Select and load from sidebar
    2. **Upload Video**: Go to Live Video Monitoring
    3. **Monitor Zones**: Watch real-time zone signals
    4. **Review Alerts**: Check alerts for anomalies
    5. **Analyze**: Explore signal analytics
    """ )

elif page == "Live Video Monitoring" :
    st.title( "🔴 Live Video Analysis" )

    if not st.session_state.current_model_id :
        st.warning( "⚠️ Please load a model from the sidebar first" )
    else :
        st.markdown( "### Upload Crowd Video" )
        uploaded_video = st.file_uploader(
            "Upload Crowd Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Video is used only for aggregated crowd signal extraction"
        )

        if uploaded_video is not None :
            col1, col2 = st.columns( [3, 1] )
            with col1 :
                start_btn = st.button( "🚀 Start Zone Monitoring", use_container_width=True )
            with col2 :
                stop_btn = st.button( "⏹️ Stop", use_container_width=True )

            if stop_btn :
                st.session_state.monitoring_active = False

            if start_btn :
                st.session_state.monitoring_active = True

            if st.session_state.monitoring_active :
                # Create placeholders
                video_col, metrics_col = st.columns( [3, 1] )

                with video_col :
                    video_placeholder = st.empty()

                with metrics_col :
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    alert_count_placeholder = st.empty()

                # Initialize
                if 'video_cap' not in st.session_state or st.session_state.video_cap is None :
                    st.session_state.video_cap, st.session_state.total_frames, st.session_state.video_fps = load_video(
                        uploaded_video )
                    ret, prev_frame = st.session_state.video_cap.read()
                    st.session_state.prev_frame = cv2.resize( prev_frame, (640, 360) )
                    st.session_state.prev_gray = cv2.cvtColor( st.session_state.prev_frame, cv2.COLOR_BGR2GRAY )
                    st.session_state.trail_layer = np.zeros_like( st.session_state.prev_frame, dtype=np.uint8 )
                    st.session_state.frame_count = 0
                    st.session_state.shake_counter = 0

                # Process frames in a controlled loop
                import base64

                while st.session_state.monitoring_active and st.session_state.video_cap.isOpened() :
                    ret, frame = st.session_state.video_cap.read()

                    if not ret :
                        status_placeholder.success( "✅ Complete!" )
                        st.session_state.monitoring_active = False
                        break

                    frame = cv2.resize( frame, (640, 360) )
                    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
                    st.session_state.frame_count += 1

                    # Skip frames
                    if st.session_state.frame_count % frame_skip != 0 :
                        st.session_state.prev_gray = gray
                        continue

                    # Check camera shake
                    # Check camera shake
                    is_shake, shake_score = detect_camera_shake( st.session_state.prev_gray, gray )

                    if is_shake :
                        st.session_state.shake_counter += 1
                        # Generate alert for camera shake
                        st.session_state.alerts.append( {
                            "timestamp" : datetime.now(),
                            "zone" : "GLOBAL",
                            "risk_level" : "WARNING",
                            "risk_score" : min( shake_score / 2.0, 1.0 ),
                            "message" : f"Camera shake detected (intensity: {shake_score:.2f})"
                        } )
                    else :
                        st.session_state.shake_counter = 0

                    if st.session_state.shake_counter < 3 :
                        # Normal processing
                        zones = split_into_zones( frame )

                        full_flow = cv2.calcOpticalFlowFarneback(
                            st.session_state.prev_gray, gray, None,
                            pyr_scale=0.4, levels=2, winsize=12,
                            iterations=2, poly_n=3, poly_sigma=1.1, flags=0
                        )

                        for zone_id, zone_frame in zones.items() :
                            zone_num = int( zone_id.split()[-1] ) - 1
                            rows, cols = 2, 3
                            h, w = frame.shape[:2]
                            zone_h, zone_w = h // rows, w // cols
                            row, col = zone_num // cols, zone_num % cols
                            y1, y2 = row * zone_h, (row + 1) * zone_h
                            x1, x2 = col * zone_w, (col + 1) * zone_w

                            zone_prev = st.session_state.prev_gray[y1 :y2, x1 :x2]
                            zone_curr = gray[y1 :y2, x1 :x2]
                            signal, _ = compute_zone_signals( zone_prev, zone_curr, zone_id )

                            if signal :
                                signal.update( {
                                    "timestamp" : datetime.now().isoformat(),
                                    "entry_count" : np.random.randint( 5, 30 ),
                                    "exit_count" : np.random.randint( 5, 30 )
                                } )
                                monitor_zone_signal( signal )

                                if detect_panic_signature( signal ) :
                                    st.session_state.alerts.append( {
                                        "timestamp" : datetime.now(),
                                        "zone" : zone_id,
                                        "risk_level" : "CRITICAL",
                                        "risk_score" : 0.95,
                                        "message" : f"Panic signature detected - Density: {signal['density']:.1f}%, Speed: {signal['speed_mean']:.2f}"
                                    } )

                                # Additional anomaly detection
                                is_anomaly, anomaly_type = detect_signal_anomaly(
                                    signal,
                                    st.session_state.zone_history[zone_id]["signals"]
                                )

                                if is_anomaly :
                                    risk_level = "WARNING" if anomaly_type != "DIRECTION_CHAOS" else "CRITICAL"
                                    st.session_state.alerts.append( {
                                        "timestamp" : datetime.now(),
                                        "zone" : zone_id,
                                        "risk_level" : risk_level,
                                        "risk_score" : 0.75 if risk_level == "WARNING" else 0.90,
                                        "message" : f"Anomaly detected: {anomaly_type}"
                                    } )

                        st.session_state.trail_layer = (st.session_state.trail_layer * 0.90).astype( np.uint8 )
                        frame, st.session_state.trail_layer = draw_optical_flow_trails( full_flow, frame,
                                                                                        st.session_state.trail_layer )
                        frame = draw_zone_grid( frame )
                        frame = draw_flow_legend( frame )

                    st.session_state.prev_gray = gray

                    # ✅ Display using base64 HTML (bypasses Streamlit media storage)
                    _, buffer = cv2.imencode( '.jpg', cv2.cvtColor( frame, cv2.COLOR_BGR2RGB ),
                                              [cv2.IMWRITE_JPEG_QUALITY, 85] )
                    img_b64 = base64.b64encode( buffer ).decode()

                    video_placeholder.markdown(
                        f'<img src="data:image/jpeg;base64,{img_b64}" width="100%">',
                        unsafe_allow_html=True
                    )

                    # Update progress
                    # Update progress
                    progress = st.session_state.frame_count / st.session_state.total_frames
                    progress_placeholder.progress( min( progress, 1.0 ) )
                    status_placeholder.text( f"Frame {st.session_state.frame_count}/{st.session_state.total_frames}" )

                    # Show alert count
                    critical_alerts = sum( 1 for a in st.session_state.alerts if a.get( 'risk_level' ) == 'CRITICAL' )
                    warning_alerts = sum( 1 for a in st.session_state.alerts if a.get( 'risk_level' ) == 'WARNING' )
                    alert_count_placeholder.markdown( f"""
                    **Alerts Generated:**  
                    🔴 Critical: {critical_alerts}  
                    🟡 Warning: {warning_alerts}
                    """ )
                    time.sleep( 1 / st.session_state.video_fps )

                    # Check if stop was pressed
                    if not st.session_state.monitoring_active :
                        break


elif page == "Signal Analytics" :
    st.title( "📊 Signal Analytics & Insights" )

    st.markdown(
        '<div class="ethics-box"><strong>Privacy Note:</strong> This system operates on aggregated motion signals only. No individuals are tracked. No identities are inferred. The dashboard provides decision support, not automated action.</div>',
        unsafe_allow_html=True )

    has_data = any( data["signals"] for data in st.session_state.zone_history.values() )

    if not has_data :
        st.info( "📌 No data available yet. Upload and process video in Live Video Monitoring!" )
    else :
        # Phase Space
        st.subheader( "Crowd Escalation Phase Space" )
        st.plotly_chart(
            create_animated_phase_space( st.session_state.zone_history ),
            use_container_width=True
        )

        st.markdown( "---" )

        # Zone Comparison
        col1, col2 = st.columns( 2 )

        with col1 :
            st.subheader( "Zone Density Trends" )
            zone_data = []
            for zone_id, data in st.session_state.zone_history.items() :
                for signal in data["signals"] :
                    zone_data.append( {
                        "Zone" : zone_id,
                        "Density" : signal["density"],
                        "Time" : signal.get( "timestamp", "" )
                    } )

            if zone_data :
                df_zones = pd.DataFrame( zone_data )
                fig = px.line( df_zones, x=df_zones.index, y="Density", color="Zone",
                               title="Density Over Time by Zone" )
                st.plotly_chart( fig, use_container_width=True )

        with col2 :
            st.subheader( "Speed vs Variance" )
            all_signals = []
            for zone_id, data in st.session_state.zone_history.items() :
                for signal in data["signals"] :
                    signal_copy = signal.copy()
                    signal_copy["zone"] = zone_id
                    all_signals.append( signal_copy )

            if all_signals :
                df_sig = pd.DataFrame( all_signals )
                fig = px.scatter( df_sig, x="speed_mean", y="direction_variance",
                                  color="zone", size="density",
                                  title="Movement Patterns" )
                st.plotly_chart( fig, use_container_width=True )

        st.markdown( "---" )

        # Statistical Summary
        st.subheader( "Statistical Summary" )
        all_densities = [s["density"] for data in st.session_state.zone_history.values()
                         for s in data["signals"]]

        if all_densities :
            col1, col2, col3, col4 = st.columns( 4 )
            with col1 :
                st.metric( "Mean Density", f"{np.mean( all_densities ):.1f}%" )
            with col2 :
                st.metric( "Max Density", f"{np.max( all_densities ):.1f}%" )
            with col3 :
                st.metric( "Min Density", f"{np.min( all_densities ):.1f}%" )
            with col4 :
                st.metric( "Std Dev", f"{np.std( all_densities ):.1f}%" )

elif page == "Alerts" :
    st.title( "🚨 Alert Management" )

    if not st.session_state.alerts :
        st.info( "No alerts generated yet. Start processing video!" )
    else :
        col1, col2, col3 = st.columns( 3 )
        critical_count = sum( 1 for a in st.session_state.alerts if a.get( 'risk_level' ) == 'CRITICAL' )
        warning_count = sum( 1 for a in st.session_state.alerts if a.get( 'risk_level' ) == 'WARNING' )

        with col1 :
            st.metric( "Critical Alerts", critical_count )
        with col2 :
            st.metric( "Warnings", warning_count )
        with col3 :
            st.metric( "Total", len( st.session_state.alerts ) )

        st.markdown( "---" )
        filter_level = st.multiselect( "Filter by Risk Level",
                                       ['CRITICAL', 'WARNING', 'NORMAL'],
                                       default=['CRITICAL', 'WARNING'] )

        st.subheader( f"Alert Log ({len( st.session_state.alerts )} total)" )
        for idx, alert in enumerate( reversed( list( st.session_state.alerts ) ) ) :
            if alert.get( 'risk_level' ) in filter_level :
                with st.expander(
                        f"Alert #{len( st.session_state.alerts ) - idx} - {alert.get( 'zone', 'Unknown' )} - {alert.get( 'risk_level' )} - {alert['timestamp'].strftime( '%H:%M:%S' )}"
                ) :
                    col1, col2 = st.columns( [2, 1] )
                    with col1 :
                        st.markdown( f"**Time**: {alert['timestamp'].strftime( '%Y-%m-%d %H:%M:%S' )}" )
                        st.markdown( f"**Zone**: {alert.get( 'zone', 'N/A' )}" )
                        st.markdown( f"**Risk Level**: {alert.get( 'risk_level' )}" )
                        st.markdown( f"**Risk Score**: {alert.get( 'risk_score', 0 ):.4f}" )
                        st.markdown( f"**Message**: {alert.get( 'message' )}" )

                    with col2 :
                        fig = go.Figure( go.Indicator(
                            mode="gauge+number",
                            value=alert.get( 'risk_score', 0 ),
                            domain={'x' : [0, 1], 'y' : [0, 1]},
                            gauge={
                                'axis' : {'range' : [0, 1]},
                                'bar' : {'color' : "#f44336" if alert.get( 'risk_level' ) == 'CRITICAL'
                                else "#ff9800" if alert.get( 'risk_level' ) == 'WARNING' else "#4caf50"},
                                'steps' : [
                                    {'range' : [0, 0.4], 'color' : "lightgreen"},
                                    {'range' : [0.4, 0.7], 'color' : "yellow"},
                                    {'range' : [0.7, 1], 'color' : "lightcoral"}
                                ]
                            }
                        ) )
                        fig.update_layout( height=200, margin=dict( l=10, r=10, t=10, b=10 ) )
                        st.plotly_chart( fig, use_container_width=True, key=f"alert_gauge_{idx}" )

        st.markdown( "---" )
        if st.button( "Export Alerts to CSV" ) :
            df_alerts = pd.DataFrame( list( st.session_state.alerts ) )
            csv = df_alerts.to_csv( index=False )
            st.download_button( "Download CSV", csv, "alerts.csv", "text/csv" )

elif page == "Ethics & Privacy" :
    st.title( "🔍 Ethics & Privacy" )

    st.markdown( '<div class="ethics-box">', unsafe_allow_html=True )
    st.markdown( "### Privacy-First Design" )
    st.markdown( """
    This system operates on **aggregated motion signals only**:

    - ✅ **No individual tracking** - People are not identified or followed
    - ✅ **No identity inference** - No facial recognition or biometric data
    - ✅ **Signal-based analysis** - Only crowd-level patterns detected
    - ✅ **Human-in-the-loop** - System provides decision support, not automated action
    - ✅ **Transparent operation** - All detections are explainable

    **Data Processing:**
    - Video frames → Optical flow → Aggregated signals
    - No personal data stored or transmitted
    - Zone-based analysis prevents individual profiling

    **Ethical Use:**
    - Designed for public safety decision support
    - Requires human oversight for any action
    - Compliant with privacy regulations
    """ )
    st.markdown( '</div>', unsafe_allow_html=True )

    st.markdown( "---" )
    st.subheader( "System Transparency" )

    col1, col2 = st.columns( 2 )

    with col1 :
        st.markdown( "**Signal Extraction**" )
        st.markdown( "- Optical flow analysis" )
        st.markdown( "- Density calculation" )
        st.markdown( "- Movement patterns" )
        st.markdown( "- Direction variance" )

    with col2 :
        st.markdown( "**Detection Process**" )
        st.markdown( "- Zone-based segmentation" )
        st.markdown( "- ML model inference" )
        st.markdown( "- Risk score calculation" )
        st.markdown( "- Alert generation" )

    st.markdown( "---" )
    st.markdown( "### Research & Development" )
    st.markdown( """
    **S.A.F.E** is designed for:
    - Academic research in crowd dynamics
    - Public safety decision support systems
    - Smart city infrastructure planning
    - Emergency response optimization

    **Not intended for:**
    - Individual surveillance
    - Behavior profiling
    - Automated enforcement
    - Privacy-invasive applications
    """ )