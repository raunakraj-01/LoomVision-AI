# LoomVision AI: Real-Time Fabric Defect Detection using Computer Vision

**Team TriNetra Vision**

## Project Overview
LoomVision AI is a real-time system that captures live fabric images using a camera and detects manufacturing defects such as broken threads, pattern mismatches, weaving errors, and color inconsistencies. Built primarily for college experiential learning coursework, it features an OpenCV-based processing pipeline, SQLite for defect logging, and a Streamlit dashboard built for live monitoring.

## Highlights
- **Real-time Pipeline:** Multi-stage image processing optimized for webcam and continuous frame capture.
- **Edge & Contour Analysis:** Detects structural mismatches like broken weaving threads.
- **SSIM/Template Matching:** Verifies complex pattern structures against a golden reference.
- **Local Logging:** Fully tracks detection alerts, saving frames and updating the database history table.

## Project Structure
```text
LoomVisionAI/
├── src/                  # Core modules (camera, detection, preprocessing, db)
├── data/                 # Sample defect images and baseline references
├── output/defects/       # Captured frames of detected defects
├── loomvision.db         # Auto-generated SQLite logs
├── app.py                # Main Streamlit dashboard application
└── requirements.txt      # Project dependencies
```

## Setup & Running the Project
1. **Clone the repository.**
2. **Create a virtual environment:** `python -m venv venv`
3. **Activate the environment:**
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. **Install Requirements:** `pip install -r requirements.txt`
5. **Run the Dashboard:** `streamlit run app.py`
