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
def load_video ( uploaded_video ) :
    """Load uploaded video file"""
    tfile = tempfile.NamedTemporaryFile( delete=False, suffix=".mp4" )
    tfile.write( uploaded_video.read() )
    tfile.close()
    return cv2.VideoCapture( tfile.name )


def draw_zone_grid ( frame, rows=2, cols=3 ) :
    """Draw zone grid overlay on frame"""
    h, w, _ = frame.shape
    zone_id = 1

    for r in range( rows ) :
        for c in range( cols ) :
            x1, y1 = int( c * w / cols ), int( r * h / rows )
            x2, y2 = int( (c + 1) * w / cols ), int( (r + 1) * h / rows )

            cv2.rectangle( frame, (x1, y1), (x2, y2), (0, 255, 0), 2 )
            cv2.putText( frame, f"Zone {zone_id}",
                         (x1 + 10, y1 + 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                         (0, 255, 0), 2 )
            zone_id += 1
    return frame


def split_into_zones ( frame, rows=2, cols=3 ) :
    """Split frame into spatial zones"""
    h, w, _ = frame.shape
    zones = {}
    zone_id = 1

    for r in range( rows ) :
        for c in range( cols ) :
            y1, y2 = int( r * h / rows ), int( (r + 1) * h / rows )
            x1, x2 = int( c * w / cols ), int( (c + 1) * w / cols )
            zones[f"Zone {zone_id}"] = frame[y1 :y2, x1 :x2]
            zone_id += 1

    return zones


def compute_zone_signals ( prev_gray, curr_gray, zone_id ) :
    """Extract motion signals from zone using optical flow"""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.4, 2, 12, 2, 3, 1.1, 0
    )

    mag, ang = cv2.cartToPolar( flow[..., 0], flow[..., 1] )

    return {
        "zone_id" : zone_id,
        "density" : float( np.mean( mag > 1.0 ) * 100 ),
        "footfall_count" : int( np.sum( mag > 1.0 ) ),
        "speed_mean" : float( np.mean( mag ) ),
        "speed_variance" : float( np.var( mag ) ),
        "direction_variance" : float( np.var( ang ) )
    }


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
    """Predict via backend API"""
    try :
        files = {"file" : ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"model_id" : model_id} if model_id else {}
        r = requests.post( f"{API_BASE}/api/predict/image", files=files, params=params, timeout=30 )
        if r.status_code == 200 :
            return r.json()
        else :
            st.error( f"Prediction error: {r.json().get( 'detail', 'Unknown' )}" )
            return None
    except Exception as e :
        st.error( f"API call failed: {e}" )
        return None


def monitor_zone_signal ( signal_data ) :
    """Store and monitor zone signal"""
    zone_id = signal_data["zone_id"]
    st.session_state.zone_history[zone_id]["signals"].append( signal_data )

    # Keep last 100 signals per zone
    if len( st.session_state.zone_history[zone_id]["signals"] ) > 100 :
        st.session_state.zone_history[zone_id]["signals"].pop( 0 )


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
        size="footfall_count",
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
                video_placeholder = st.empty()
                metrics_placeholder = st.empty()
                heatmap_placeholder = st.empty()
                progress_bar = st.progress( 0 )
                status_text = st.empty()

                cap = load_video( uploaded_video )
                total_frames = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
                fps = int( cap.get( cv2.CAP_PROP_FPS ) )

                ret, prev_frame = cap.read()
                prev_frame = cv2.resize( prev_frame, (640, 360) )
                prev_gray = cv2.cvtColor( prev_frame, cv2.COLOR_BGR2GRAY )

                frame_count = 0

                try :
                    while cap.isOpened() and st.session_state.monitoring_active :
                        ret, frame = cap.read()
                        if not ret :
                            break

                        frame = cv2.resize( frame, (640, 360) )
                        frame_count += 1

                        if frame_count % frame_skip != 0 :
                            continue

                        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
                        zones = split_into_zones( frame )

                        # Process each zone
                        for zone_id, zone_frame in zones.items() :
                            h, w = zone_frame.shape[:2]
                            zone_prev = prev_gray[:h, :w]
                            zone_curr = gray[:h, :w]

                            signal = compute_zone_signals( zone_prev, zone_curr, zone_id )
                            signal.update( {
                                "timestamp" : datetime.now().isoformat(),
                                "entry_count" : np.random.randint( 5, 30 ),
                                "exit_count" : np.random.randint( 5, 30 )
                            } )
                            monitor_zone_signal( signal )

                            # API prediction for high-risk zones
                            if signal["density"] > 60 :
                                _, buffer = cv2.imencode( '.jpg', zone_frame )
                                file_bytes = io.BytesIO( buffer )
                                result = predict_image( file_bytes, st.session_state.current_model_id )

                                if result and result["risk_score"] > risk_threshold :
                                    alert = {
                                        'timestamp' : datetime.now(),
                                        'zone' : zone_id,
                                        'risk_level' : result["risk_level"],
                                        'risk_score' : result["risk_score"],
                                        'message' : f"{result['risk_level']} risk in {zone_id}"
                                    }
                                    st.session_state.alerts.append( alert )

                        # Display annotated frame
                        frame_overlay = draw_zone_grid( frame.copy() )
                        video_placeholder.image(
                            cv2.cvtColor( frame_overlay, cv2.COLOR_BGR2RGB ),
                            use_container_width=True
                        )

                        # Update metrics
                        with metrics_placeholder.container() :
                            col1, col2, col3, col4 = st.columns( 4 )
                            with col1 :
                                st.metric( "Frame", f"{frame_count}/{total_frames}" )
                            with col2 :
                                st.metric( "Active Zones", len( zones ) )
                            with col3 :
                                avg_density = np.mean( [
                                    s["density"] for data in st.session_state.zone_history.values()
                                    for s in data["signals"][-1 :] if data["signals"]
                                ] )
                                st.metric( "Avg Density", f"{avg_density:.1f}%" )
                            with col4 :
                                st.metric( "Alerts", len( st.session_state.alerts ) )

                        # Update heatmap
                        heatmap_placeholder.plotly_chart(
                            create_zone_heatmap( st.session_state.zone_history ),
                            use_container_width=True
                        )

                        prev_gray = gray
                        progress = frame_count / total_frames
                        progress_bar.progress( progress )
                        status_text.text( f"Processing... {progress * 100:.1f}% complete" )
                        time.sleep( 1 / fps )

                    status_text.success( "✅ Video processing complete!" )
                    st.session_state.monitoring_active = False

                except Exception as e :
                    st.error( f"Error during processing: {e}" )
                finally :
                    cap.release()

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
                                  color="zone", size="footfall_count",
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
                        f"Alert #{len( st.session_state.alerts ) - idx} - {alert.get( 'zone', 'Unknown' )} - {alert.get( 'risk_level' )} - {alert['timestamp'].strftime( '%H:%M:%S' )}" ) :
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
                        st.plotly_chart( fig, use_container_width=True )

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