"""
flask_server.py
---------------
Flask & WebSocket server for LoomVision AI.
Serves the web dashboard, provides REST API endpoints, and streams 
live camera feed + anomaly heatmaps + metrics over WebSockets.
"""

# Ensure eventlet is used for async/websocket
import eventlet
eventlet.monkey_patch()

import os
import cv2
import time
import json
import base64
import numpy as np
import threading
from datetime import datetime
from urllib.parse import quote

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from src.camera import CameraController
from src.prediction_engine import PredictionEngine
from src.database import DatabaseLogger
from src.design_suggestions import get_design_suggestions
from src.report_generator import generate_inspection_report
from src.notifications import send_whatsapp_alert
from evaluate_classifier import evaluate

# ── Global State ────────────────────────────────────────────────────────
MOBILE_CAMERA_ACTIVE = False
INSPECTION_ACTIVE = False
app = Flask(__name__, static_folder="frontend")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

CAM_CONTROLLER = None
PREDICTION_ENGINE = None
DB_LOGGER = None

GLOBAL_SCANS = 0
DEFECTS_FOUND = 0
START_TIME = time.time()

LATEST_FRAME = None
LATEST_HEATMAP = None

# Background thread control
PROCESSING_THREAD = None
STOP_EVENT = threading.Event()

# Rate limiting for WhatsApp
LAST_ALERT_TIME = 0
ALERT_COOLDOWN = 10  # seconds (reduced for testing)

# Rate limiting for defect logging (prevents duplicate photos for same defect)
LAST_DEFECT_LOG_TIME = 0
DEFECT_LOG_COOLDOWN = 5  # seconds — same defect won't be saved again within this window


# ── Initialization ────────────────────────────────────────────────────
def init_system():
    global CAM_CONTROLLER, PREDICTION_ENGINE, DB_LOGGER
    print("[Server] Initialising camera controller...")
    CAM_CONTROLLER = CameraController()
    if not CAM_CONTROLLER.initialize():
        print("[Server] Warning: Camera failed to initialize.")

    print("[Server] Initialising Prediction Engine...")
    PREDICTION_ENGINE = PredictionEngine(engine_type="auto")

    print("[Server] Initialising Database Logger...")
    DB_LOGGER = DatabaseLogger()


def process_and_emit_frame(frame):
    global GLOBAL_SCANS, DEFECTS_FOUND, LAST_DEFECT_LOG_TIME, LATEST_FRAME, LATEST_HEATMAP
    
    # Save a clean copy BEFORE any ML processing touches the frame
    raw_frame = frame.copy()

    try:
        GLOBAL_SCANS += 1
        
        # Run prediction
        result = PREDICTION_ENGINE.process_frame(frame)
        
        # Force contiguous numpy arrays — eventlet's monkey-patching can
        # wrap objects so that OpenCV's C extension refuses them.
        annotated = np.ascontiguousarray(result.annotated_frame)
        LATEST_FRAME = annotated
        LATEST_HEATMAP = result.heatmap
        safe_raw = np.ascontiguousarray(raw_frame)

        # Log defect and alert
        if result.has_defect:
            DEFECTS_FOUND += 1
            
            # Only log to DB once per cooldown window
            # This prevents 30+ duplicate entries for a single real-world defect
            now = time.time()
            if now - LAST_DEFECT_LOG_TIME > DEFECT_LOG_COOLDOWN:
                LAST_DEFECT_LOG_TIME = now
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Log to DB (this also saves the image and triggers WhatsApp internally)
                DB_LOGGER.log_defect(
                    defect_type=result.defect_type or "Unknown Anomaly",
                    annotated_frame=annotated,
                    confidence=result.confidence,
                    anomaly_score=float(result.anomaly_score)
                )
                
                # Notify clients via WebSocket so UI updates instantly
                # Generate the same filename format that database.py uses for the websocket message
                filename_safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                expected_img_path = f"/output/defects/defect_{filename_safe_time}.jpg"
                
                socketio.emit("defect_alert", {
                    "defect_type": result.defect_type,
                    "confidence": result.confidence,
                    "anomaly_score": float(result.anomaly_score),
                    "timestamp": timestamp,
                    "image_path": expected_img_path
                })

        # Emit live frames (JPEG encoded base64)
        # AI-annotated frame (with heatmap overlay, bounding boxes, etc.)
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64_frame = base64.b64encode(buffer).decode('utf-8')

        # Raw clean camera frame (no AI overlay)
        _, raw_buffer = cv2.imencode('.jpg', safe_raw, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64_raw_frame = base64.b64encode(raw_buffer).decode('utf-8')
        
        socketio.emit("calibration_update", {
            "is_calibrated": getattr(result, "is_calibrated", True),
            "progress": getattr(result, "calibration_progress", 1.0)
        })
        
        socketio.emit("frame_update", {
            "image": f"data:image/jpeg;base64,{b64_frame}",
            "raw_image": f"data:image/jpeg;base64,{b64_raw_frame}"
        })

    except Exception as e:
        print(f"[Background] Error processing frame: {e}")


# ── Background Processing Thread ──────────────────────────────────────
def process_camera_feed():
    """Background task to read from local camera and run predictions."""
    global GLOBAL_SCANS, DEFECTS_FOUND, LAST_DEFECT_LOG_TIME, LATEST_FRAME, LATEST_HEATMAP
    
    print("[Background] Started background camera worker.")
    while not STOP_EVENT.is_set():
        if MOBILE_CAMERA_ACTIVE:
            # If mobile camera is active, we don't pull from the local CAM_CONTROLLER.
            # Instead, we just sleep and let the websocket handler process incoming frames.
            eventlet.sleep(0.1)
            continue
            
        if not INSPECTION_ACTIVE:
            eventlet.sleep(0.1)
            continue
            
        ret, frame = CAM_CONTROLLER.get_frame()
        if not ret or frame is None:
            eventlet.sleep(0.05)
            continue

        process_and_emit_frame(frame)

        # Sleep to yield to other greenlets (Eventlet requirement)
        eventlet.sleep(0.01)


def metrics_publisher():
    """Periodically pushes KPI metrics to clients."""
    while not STOP_EVENT.is_set():
        uptime = time.time() - START_TIME
        accuracy = 100.0 - ((DEFECTS_FOUND / max(GLOBAL_SCANS, 1)) * 100.0)
        
        metrics = {
            "total_scans": GLOBAL_SCANS,
            "defects_found": DEFECTS_FOUND,
            "uptime_seconds": uptime,
            "accuracy_rate": f"{accuracy:.1f}%",
            "inspection_active": INSPECTION_ACTIVE
        }
        socketio.emit("metrics_update", metrics)
        eventlet.sleep(2.0)


# ── REST Endpoints ────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory("frontend", "index.html")

@app.route("/dashboard")
def serve_dashboard():
    return send_from_directory("frontend", "flask_dashboard.html")

@app.route("/analytics")
def serve_analytics():
    return send_from_directory("frontend", "flask_analytics.html")

@app.route("/frontend/<path:filename>")
def serve_static_frontend(filename):
    return send_from_directory("frontend", filename)

@app.route("/tailwind_output.css")
def serve_tailwind():
    return send_from_directory("frontend", "tailwind_output.css")

@app.route("/output/defects/<path:filename>")
def serve_defect_images(filename):
    return send_from_directory("output/defects", filename)

@app.route("/api/v1/metrics", methods=["GET"])
def get_metrics():
    accuracy = 100.0 - ((DEFECTS_FOUND / max(GLOBAL_SCANS, 1)) * 100.0)
    return jsonify({
        "total_scans": GLOBAL_SCANS,
        "defects_found": DEFECTS_FOUND,
        "uptime": time.time() - START_TIME,
        "accuracy_rate": f"{accuracy:.1f}%",
        "inspection_active": INSPECTION_ACTIVE
    })

@app.route("/api/v1/defects", methods=["GET"])
def get_defects():
    defects = DB_LOGGER.get_recent_defects(limit=50)
    # Prefix image path with / to serve correctly
    for d in defects:
        if d.get("image_path") and not d["image_path"].startswith("/"):
            d["image_path"] = "/" + d["image_path"]
    return jsonify(defects)

@app.route("/api/v1/defects", methods=["DELETE"])
def clear_defects():
    global DEFECTS_FOUND
    DEFECTS_FOUND = 0
    DB_LOGGER.clear_all_defects()
    return jsonify({"status": "success", "message": "All defects cleared"})

@app.route("/api/v1/toggle_inspection", methods=["POST"])
def toggle_inspection():
    global INSPECTION_ACTIVE
    INSPECTION_ACTIVE = not INSPECTION_ACTIVE
    socketio.emit("status_update", {"inspection_active": INSPECTION_ACTIVE})
    return jsonify({"inspection_active": INSPECTION_ACTIVE})

@app.route("/api/v1/set_fabric_type", methods=["POST"])
def set_fabric_type():
    global PREDICTION_ENGINE
    data = request.json or {}
    fabric_type = data.get("fabric_type")
    if fabric_type in ["plain", "embroidered"]:
        if PREDICTION_ENGINE:
            PREDICTION_ENGINE.fabric_type = fabric_type
            # Restart the warmup phase so PatchCore learns the new fabric
            PREDICTION_ENGINE.patchcore._frame_count = 0
            PREDICTION_ENGINE.patchcore._memory_deque.clear()
        return jsonify({"status": "success", "fabric_type": fabric_type})
    return jsonify({"status": "error", "message": "Invalid fabric type"}), 400

@app.route("/api/v1/switch_camera", methods=["POST"])
def switch_camera():
    if CAM_CONTROLLER:
        success = CAM_CONTROLLER.switch_camera()
        socketio.emit("status_update", {"camera_switched": success})
        return jsonify({"success": success})
    return jsonify({"success": False, "error": "Camera controller not initialized"}), 500

@app.route("/api/v1/camera_info", methods=["GET"])
def camera_info():
    if CAM_CONTROLLER:
        return jsonify(CAM_CONTROLLER.get_source_info())
    return jsonify({"status": "disconnected", "type": "none"})

@app.route("/api/v1/design_suggestions", methods=["POST"])
def design_suggestions():
    if LATEST_FRAME is None:
        return jsonify({"error": "No frame available yet"}), 400
    try:
        # get_design_suggestions expects BGR frame
        suggestions = get_design_suggestions(LATEST_FRAME)
        return jsonify(suggestions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/generate_report", methods=["POST"])
def generate_report():
    try:
        pdf_path = generate_inspection_report(total_scans=GLOBAL_SCANS)
        return jsonify({
            "status": "success",
            "report_path": pdf_path,
            "download_url": f"/api/v1/download_report?path={quote(pdf_path)}",
            "filename": os.path.basename(pdf_path)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/download_report", methods=["GET"])
def download_report():
    report_path = request.args.get("path")
    if report_path and os.path.exists(report_path) and report_path.endswith(".pdf"):
        # Security check
        abs_req = os.path.abspath(report_path)
        abs_rep = os.path.abspath("output/reports")
        if abs_req.startswith(abs_rep):
            return send_file(abs_req, as_attachment=True, download_name=os.path.basename(report_path))
    return jsonify({"error": "Report not found or access denied"}), 404

@app.route("/api/v1/shutdown", methods=["POST"])
def shutdown():
    def do_shutdown():
        STOP_EVENT.set()
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=do_shutdown).start()
    return jsonify({"status": "shutting down"})

@app.route("/api/v1/run_evaluation", methods=["POST"])
def run_evaluation():
    try:
        results = evaluate()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ── WebSocket Events ──────────────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    print("[WebSocket] Client connected")
    emit("status_update", {"inspection_active": INSPECTION_ACTIVE})

@socketio.on("disconnect")
def handle_disconnect():
    print("[WebSocket] Client disconnected")

@socketio.on("set_mobile_camera_mode")
def handle_mobile_mode(data):
    global MOBILE_CAMERA_ACTIVE
    MOBILE_CAMERA_ACTIVE = data.get("active", False)
    print(f"[WebSocket] Mobile camera mode set to: {MOBILE_CAMERA_ACTIVE}")

@socketio.on("client_frame")
def handle_client_frame(data):
    if not INSPECTION_ACTIVE or not MOBILE_CAMERA_ACTIVE:
        return
        
    try:
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
            
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            process_and_emit_frame(frame)
    except Exception as e:
        print(f"[WebSocket] Error handling client frame: {e}")


# ── Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    init_system()
    
    # Start background tasks via Eventlet
    eventlet.spawn(process_camera_feed)
    eventlet.spawn(metrics_publisher)
    
    print("[Server] Starting Flask+WebSocket Server on http://0.0.0.0:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, log_output=False)
