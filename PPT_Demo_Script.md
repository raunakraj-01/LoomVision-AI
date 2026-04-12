# LoomVision AI: PPT & Demo Script Outline

Use this template to create your final presentation slides. It is structured perfectly for a 10-15 minute college or placement defense.

---

## Slide 1: Title Screen
**Title:** LoomVision AI: Real-Time Fabric Defect Detection using Computer Vision
**Subtitle:** Automating Quality Assurance in Textile Manufacturing
**Team:** TriNetra Vision (List 5 members)
**Guide/Mentor:** [Insert Name]

## Slide 2: Problem Statement
*   **Manual Inspection is Flawed:** Human visual inspection of fabric is slow, prone to fatigue, and only ~60-70% accurate.
*   **Cost of Defects:** Undetected weaving errors, color mismatches, and broken threads result in massive material waste and financial loss.
*   **The Goal:** Build an automated, real-time Computer Vision system to instantly flag defects during the manufacturing process.

## Slide 3: Proposed Solution
*   A **low-cost Python-based system** utilizing webcam/Raspberry Pi.
*   **Real-time monitoring** via a Streamlit Dashboard.
*   **Auto-Logging:** Any defect detected is automatically saved to a database with a timestamp and visual proof.

## Slide 4: Project Architecture (Include a diagram here)
1.  **Hardware Input:** Video capture module (OpenCV).
2.  **Preprocessing:** Grayscale, Gaussian Blur (noise reduction), Histogram Equalization (lighting correction).
3.  **Algorithmic Core:**
    *   *Edge & Contour Maps (Canny/Dilation)* for structural faults (holes).
    *   *Structural Similarity Index / Absolute Difference* for pattern/color mismatches.
4.  **Logging & UI:** SQLite Database → Streamlit Frontend.

## Slide 5: Technologies Used
*   **Language:** Python 3
*   **Computer Vision:** OpenCV (`cv2`), NumPy (Matrix Operations)
*   **Frontend/Dashboard:** Streamlit (Real-time reactivity)
*   **Database:** SQLite (Lightweight, local logging)
*   **Version Control/Testing:** Git & Python `unittest`

## Slide 6: The Detection Logic (The "How It Works")
*Show a 3-step image workflow on this slide:*
1.  *Image 1:* The Raw Camera Feed.
2.  *Image 2:* The Edge Detection Mask (Black background with white edges).
3.  *Image 3:* The bounding box drawing over the anomaly.

*   **Explain:** "We calculate the contours of anomalies in the thread pattern. If an anomaly's pixel area exceeds our tuning threshold (e.g., 500px), it triggers positive for a defect."

## Slide 7: Live Demo / Screenshots
*(If you do a live demo, skip this. If not, include 3 screenshots:)*
*   Screenshot 1: The Streamlit UI with a perfect fabric.
*   Screenshot 2: A defective fabric showing the red bounding box.
*   Screenshot 3: The database logs table filling up.

## Slide 8: Challenges Faced & Solutions
*   *Challenge:* Inconsistent lighting causing false positives.
    *   *Solution:* Implemented Histogram Equalization to normalize frame lighting.
*   *Challenge:* High video latency.
    *   *Solution:* Bypassed heavy Deep Learning models (like YOLO) initially and utilized optimized classical computer vision mathematics via NumPy arrays to maintain 30+ FPS on a standard laptop.

## Slide 9: Future Scope
*   **Hardware Integration:** Mount the camera on an actual Raspberry Pi 4 above a moving conveyor belt.
*   **Deep Learning Pipeline:** Train a MobileNet / YOLOv8 classification model on thousands of fabric images to categorize specific fabric types automatically.
*   **Cloud Logging:** Push the SQLite logs to an AWS/Firebase cloud dashboard for factory managers to see.

## Slide 10: Conclusion & Q/A
*   Summarize: We successfully built an automated QA pipeline that proves computer vision is a viable, low-cost replacement for manual fabric inspection.
*   "Thank you. We are open to questions!"
