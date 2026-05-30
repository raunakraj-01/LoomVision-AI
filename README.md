# LoomVision AI: Real-Time Fabric Defect Detection using Deep Learning & Computer Vision



## Project Overview
LoomVision AI is a real-time system that captures live fabric images directly from a mobile phone camera and detects manufacturing defects such as broken threads, loose weaves, and stains. Built primarily for handloom workers, it features a split-cloud architecture with a lightweight **Vercel Frontend** and a heavy deep-learning **Flask/Render Backend**.

The AI combines a state-of-the-art **Dynamic PatchCore** deep learning anomaly detector with a robust temporal sequence model and OpenCV heuristic classification to accurately identify and classify defects while ignoring expected complexities like embroidery, shadows, and wrinkles.

---

## AI Architecture

### 1. Dynamic PatchCore AI (Primary Deep Learning Engine)
- **Backbone**: Pretrained ResNet-18 Deep Residual Network (PyTorch).
- **Multi-Scale Feature Extraction**: Hooks into both Layer 2 (128-ch) and Layer 3 (256-ch) to produce rich 384-dimensional patch descriptors representing fabric textures.
- **Sliding-Window Memory Bank**: Continuously updates its definition of "normal" fabric using a rolling window. This allows seamless adaptation to saree transitions (body → border → pallu) without false positives.
- **Adaptive Threshold**: Uses a running mean + standard deviation dynamic threshold, strictly capped to prevent hypersensitivity on highly uniform plain cloth.

### 2. Heuristic Classifier (OpenCV)
- **Defect Categorization**: When PatchCore flags an anomaly, this module analyzes structural density and color variance to assign a specific class (Stain, Loose Weave, Broken Thread).
- **Intelligent Suppression**: Identifies and actively suppresses non-defects like **Wrinkles/Shadows** (via extreme low edge density) and **Embroidery** (via extreme high edge density and color variance).

### 3. Temporal Sequence Debouncing (LSTM/Logic)
- Eliminates camera-shake false positives by requiring a defect to persist in the same spatial location across **3 consecutive frames** before triggering an alert.

---

## Highlights
- **Mobile-First Cloud Architecture:** No laptop required. Workers simply open the web app on their phones to stream the live camera feed to the cloud AI engine via WebSockets.
- **Real-Time WhatsApp Alerts:** Automatically pings a supervisor or worker with the exact time and type of defect via the Twilio API.
- **Unsupervised Learning:** PatchCore requires zero labelled defect images. It learns what "normal" looks like live and flags anything anomalous.
- **Premium Glassmorphism UI:** Stunning web dashboard with live camera feeds, live log panels, and dynamic analytics.

---

## Project Structure
```text
LoomVisionAI/
├── src/
│   ├── camera.py              # Camera controller
│   ├── preprocessing.py       # LAB/HSV preprocessing pipeline
│   ├── dynamic_patchcore.py   # ★ Deep Learning PatchCore anomaly detector
│   ├── heuristic_classifier.py# Post-detection categorizer & noise suppressor
│   ├── sequence_model.py      # Temporal debouncing logic
│   ├── prediction_engine.py   # Orchestrator binding all AI modules together
│   ├── database.py            # SQLite defect logger
│   └── notifications.py       # Twilio WhatsApp Integration
├── vercel_frontend/           # The static HTML/JS App (Deployed to Vercel)
├── output/defects/            # Captured frames of detected defects
├── render.yaml                # Automated Cloud Backend deployment config
├── launch.py                  # Local Unified Launcher (Server + USB Watcher)
├── usb_watcher.py             # Daemon to auto-open dashboard on USB connect
├── flask_server.py            # Main Cloud API and WebSocket server
└── requirements.txt           # Project dependencies
```

## Tech Stack
`PyTorch` · `torchvision (ResNet-18)` · `OpenCV` · `Flask-SocketIO` · `Twilio` · `SQLite` · `Vanilla HTML/JS/CSS`

---

## Setup & Running the Project (Cloud Deployment)

### 1. The Backend (Render.com)
1. Fork/Clone the repository.
2. Link the repository to a new web service on **Render.com**. It will automatically detect `render.yaml`.
3. Add your Twilio API credentials to the environment variables on the Render dashboard.

### 2. The Frontend (Vercel)
1. Open `vercel_frontend/index.html` and `vercel_frontend/analytics.html`.
2. Find `const BACKEND_URL` and replace it with your live Render backend URL.
3. Link the repository to a new project on **Vercel.com**.
4. Set the "Root Directory" to `vercel_frontend` and deploy!

## Setup & Running (Local Development)
1. **Create a virtual environment:** `python3 -m venv venv`
2. **Activate the environment:**
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
3. **Install Requirements:** `pip install -r requirements.txt`
4. **Run the System:** `python3 launch.py`
   - This will start the Flask server and USB watcher. If an Android phone is connected via USB, it will automatically open the dashboard in your browser.
