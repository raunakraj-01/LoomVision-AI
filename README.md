# LoomVision AI: Real-Time Fabric Defect Detection using Deep Learning & Computer Vision

**Team TriNetra Vision**

## Project Overview
LoomVision AI is a real-time system that captures live fabric images using a camera and detects manufacturing defects such as broken threads, pattern mismatches, weaving errors, and color inconsistencies. Built primarily for college experiential learning coursework, it features a dual-engine architecture combining classical OpenCV processing with a state-of-the-art **Dynamic PatchCore** deep learning anomaly detector, SQLite for defect logging, and a premium Streamlit dashboard for live monitoring.

## AI Architecture

### Engine 1: OpenCV (Classical Computer Vision)
- **Structural Detection**: Canny edge detection + contour analysis with geometric crease-rejection filters (solidity, aspect ratio, extent).
- **Colour Anomaly Detection**: HSV tile-based analysis detecting hue-shift stains, saturation drops, and dye bleeding.
- **Adaptive Pattern Matching**: Real-time SSIM (Structural Similarity Index) on luminance with exponential moving average background learning.

### Engine 2: Dynamic PatchCore AI (Deep Learning) — **Primary Engine**
- **Backbone**: Pretrained ResNet-18 Deep Residual Network (PyTorch).
- **Multi-Scale Feature Extraction**: Hooks into both Layer 2 (128-ch, thread textures) and Layer 3 (256-ch, structural patterns) producing 384-dimensional patch descriptors.
- **Sliding-Window Memory Bank**: Continuously updates its definition of "normal" fabric using a rolling window of the last ~30 clean frames. This allows seamless adaptation to saree transitions (body → border → pallu) without false positives.
- **Adaptive Threshold**: Uses running mean + σ-based dynamic threshold instead of a static cutoff, minimising false alarms across varying fabric types.
- **Spatial Anomaly Heatmap**: Maps deep feature anomaly scores back to a 14×14 spatial grid, overlaying a JET-colourmap heatmap on detected defects for precise localisation.

## Highlights
- **Real-time Pipeline:** Multi-stage image processing optimized for webcam and continuous frame capture.
- **All-Colour Support:** Works on any saree colour — silk, cotton, embroidery, zari — via LAB/HSV preprocessing and colour-agnostic deep features.
- **Unsupervised Learning:** PatchCore requires zero labelled defect images. It learns what "normal" looks like from the live stream and flags anything anomalous.
- **Local Logging:** Fully tracks detection alerts, saving frames and updating the database history table.
- **Premium UI:** Glassmorphism Streamlit dashboard with animated status indicators, expandable log panels, and real-time metrics.

## Project Structure
```text
LoomVisionAI/
├── src/
│   ├── camera.py              # Camera controller (auto-detect / USB)
│   ├── preprocessing.py       # LAB/HSV preprocessing pipeline
│   ├── detection.py           # Classical CV engines (structural + colour + SSIM)
│   ├── dynamic_patchcore.py   # ★ Deep Learning PatchCore anomaly detector
│   ├── database.py            # SQLite defect logger
│   └── ml_detection.py        # Legacy YOLO wrapper (deprecated)
├── data/                      # Sample defect images and baseline references
├── output/defects/            # Captured frames of detected defects
├── models/                    # Model weights
├── app.py                     # Main Streamlit dashboard application
└── requirements.txt           # Project dependencies
```

## Tech Stack
`PyTorch` · `torchvision (ResNet-18)` · `OpenCV` · `scikit-image` · `NumPy` · `Pandas` · `Streamlit` · `SQLite`

## Setup & Running the Project
1. **Clone the repository.**
2. **Create a virtual environment:** `python -m venv venv`
3. **Activate the environment:**
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. **Install Requirements:** `pip install -r requirements.txt`
5. **Run the Dashboard:** `streamlit run app.py`
