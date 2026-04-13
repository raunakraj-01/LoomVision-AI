import streamlit as st
import cv2
import sqlite3
import pandas as pd
from PIL import Image

# Import our custom modules
from src.camera import CameraController
from src.preprocessing import apply_preprocessing
from src.detection import DefectDetector
from src.database import DatabaseLogger
from src.ml_detection import MLDetector

# Set up page configurations
st.set_page_config(page_title="LoomVision AI", page_icon="👁️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #00FFCC;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #00FFCC;
        color: #1E1E1E;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00E6B8;
        box-shadow: 0 4px 15px rgba(0, 255, 204, 0.4);
    }
    hr {
        border-color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("👁️ LoomVision AI Dashboard")
st.markdown("**Real-Time Fabric Defect Detection using Computer Vision** | Team TriNetra Vision")
st.divider()

# Initialize classes if not in session state
if 'detector' not in st.session_state:
    st.session_state.detector = DefectDetector(contour_area_threshold=800)
    st.session_state.db_logger = DatabaseLogger()
    try:
        st.session_state.ml_detector = MLDetector()
    except Exception as e:
        st.error(f"Failed to load ML Model: {e}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")
    
    # AI Engine Toggle
    mode = st.radio("⚙️ Select Processing Engine:", ("OpenCV (Mathematical)", "YOLOv8 (Deep Learning)"), horizontal=True)
    
    # Template Capture System
    st.markdown("---")
    st.markdown("**Setup Phase:** Focus camera on a PERFECT section of cloth.")
    col_btn1, col_btn2 = st.columns([1,1])
    with col_btn1:
        if st.button("📸 Capture Golden Pattern"):
            cam = CameraController(0)
            if cam.initialize():
                ret, frame = cam.get_frame()
                if ret:
                    # Crop a center piece to use as the "Perfect Template"
                    h, w = frame.shape[:2]
                    crop_size = 100
                    center_y, center_x = h // 2, w // 2
                    patch = frame[center_y-crop_size:center_y+crop_size, center_x-crop_size:center_x+crop_size]
                    
                    # Store grayscale version in session state for the runtime to use
                    st.session_state.golden_template = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    st.success("Golden Pattern Memorized!")
                cam.release()
                
    with col_btn2:
        if 'golden_template' in st.session_state:
            st.image(st.session_state.golden_template, caption="Memorized Perfect Pattern", width=100)
        else:
            st.info("No pattern memorized yet.")
            
    st.markdown("---")
    
    run_camera = st.checkbox("▶️ Start Production Line Feed", value=False)
    FRAME_WINDOW = st.image([])

with col2:
    st.subheader("System Status")
    status_placeholder = st.empty()
    audio_placeholder = st.empty()
    st.divider()
    st.subheader("Defect Logs")
    logs_placeholder = st.empty()

def load_logs():
    """Helper to fetch recent defects from SQLite."""
    try:
        conn = sqlite3.connect(st.session_state.db_logger.db_path)
        df = pd.read_sql_query("SELECT timestamp, defect_type, image_path FROM defects ORDER BY id DESC LIMIT 10", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# Loop for Live Camera
if run_camera:
    status_placeholder.success("✅ System Active. Monitoring for defects...")
    
    # Initialize Camera
    cam = CameraController(0)
    if cam.initialize():
        while run_camera:
            ret, frame = cam.get_frame()
            if not ret:
                st.error("Failed to fetch camera feed.")
                break
            
            has_defect = False
            defect_info = None
            annotated_frame = frame.copy()
            
            # --- 1. Processing Pipeline ---
            if mode == "OpenCV (Mathematical)":
                preprocessed = apply_preprocessing(frame)
                
                # Check A: Massive Structural Holes (Edge Detection)
                has_struc_defect, struc_info, ann_frame_1 = st.session_state.detector.detect_structural_defect(preprocessed, frame)
                
                # Check B: Pattern Mismatch (Template Matching) 
                has_pat_defect = False
                if 'golden_template' in st.session_state:
                     has_pat_defect, pat_info, ann_frame_2 = st.session_state.detector.detect_defect_via_template(
                         preprocessed, st.session_state.golden_template, frame, threshold=0.6)
                
                # Consolidate results
                if has_struc_defect:
                    has_defect = True
                    defect_info = struc_info
                    annotated_frame = ann_frame_1
                elif 'golden_template' in st.session_state and has_pat_defect:
                    has_defect = True
                    defect_info = pat_info
                    annotated_frame = ann_frame_2

            else:
                # Deep Learning YOLO Pipeline
                has_defect, defect_info, annotated_frame = st.session_state.ml_detector.detect_defects(frame)
            
            # --- 2. Logging & Alarm ---
            if has_defect:
                st.session_state.db_logger.log_defect(defect_info, annotated_frame)
                status_placeholder.error(f"🚨 ALERT: {defect_info} detected!")
                # Javascript Audio Alarm for Weaver (Beep sound!)
                audio_placeholder.markdown(
                    """
                    <audio autoplay>
                      <source src="https://www.soundjay.com/buttons/sounds/beep-01a.mp3" type="audio/mpeg">
                    </audio>
                    """, unsafe_allow_html=True
                )
            else:
                audio_placeholder.empty()
                
            # --- 4. Display ---
            # Convert BGR (OpenCV) to RGB (Streamlit/PIL)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, use_container_width=True)
            
            # Refresh logs table live
            logs_df = load_logs()
            if not logs_df.empty:
                logs_placeholder.dataframe(logs_df, use_container_width=True, hide_index=True)
                
        cam.release()
    else:
        st.error("Could not access Webcam.")
        run_camera = False
else:
    status_placeholder.warning("⏸️ System Paused. Check 'Start' to begin.")
    # Show history even when camera is off
    logs_df = load_logs()
    if not logs_df.empty:
        logs_placeholder.dataframe(logs_df, use_container_width=True, hide_index=True)
    else:
        logs_placeholder.info("No defects logged yet.")
