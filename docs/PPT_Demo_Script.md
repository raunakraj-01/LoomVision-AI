# LoomVision AI: PPT & Demo Script Outline

Use this template to create your final presentation slides. It is structured perfectly for a 10-15 minute college or placement defense.

---

## Slide 1: Title Screen
**Title:** LoomVision AI: Real-Time Fabric Defect Detection using Computer Vision
**Subtitle:** Automating Quality Assurance in Textile Manufacturing
**Developer:** Raunak Raj
**Guide/Mentor:** Dr. Prapulla S B

## Slide 2: Problem Statement
*   **Manual Inspection is Flawed:** Human visual inspection of fabric is slow, prone to fatigue, and only ~60-70% accurate.
*   **Cost of Defects:** Undetected weaving errors, color mismatches, and broken threads result in massive material waste and financial loss.
*   **The Goal:** Build an automated, real-time Computer Vision system to instantly flag defects during the manufacturing process, specifically tailored for handloom workers using just a mobile phone.

## Slide 3: Proposed Solution
*   **Mobile-First Cloud Architecture:** Workers stream live video directly from their phone browser to a powerful AI cloud server.
*   **Unsupervised Deep Learning:** The system learns "normal" fabric patterns dynamically on the fly, without needing a pre-trained dataset.
*   **Instant Alerts:** Defects trigger real-time WhatsApp notifications via Twilio, alongside a live premium web dashboard for supervisors.

## Slide 4: Project Architecture (Include a diagram here)
1.  **Frontend (Vercel):** Lightweight HTML/JS UI capturing mobile camera frames via WebSockets.
2.  **Backend (Flask/Render):** Orchestrates the AI pipeline and handles API requests.
3.  **Algorithmic Core:**
    *   *Dynamic PatchCore (ResNet-18):* Deep learning anomaly detector using a rolling memory bank of live frames.
    *   *OpenCV Heuristic Classifier:* Post-processes anomalies to categorize defects (Stains, Loose Weaves) and suppress noise (Embroidery, Shadows).
    *   *Temporal LSTM Logic:* Debounces the video feed to eliminate false positives from camera shake.
4.  **Logging & Alerts:** SQLite Database + Twilio WhatsApp API.

## Slide 5: Technologies Used
*   **Languages:** Python 3, JavaScript, HTML, CSS
*   **Deep Learning:** PyTorch, torchvision (ResNet-18)
*   **Computer Vision:** OpenCV (`cv2`), NumPy, scikit-image
*   **Web Framework:** Flask, Flask-SocketIO (Eventlet)
*   **APIs & Integrations:** Twilio (WhatsApp), Vercel, Render

## Slide 6: The AI Innovation (Dynamic PatchCore)
*   **Why not YOLO?** Traditional object detection requires thousands of manually labeled images for every new fabric type.
*   **The PatchCore Advantage:** It extracts deep features from a pretrained ResNet-18 and saves them into a "Memory Bank". 
*   **Dynamic Adaptation:** As the saree changes (e.g., body to border), the memory bank updates dynamically, allowing the system to work on *any* fabric instantly without retraining.

## Slide 7: Live Demonstration
*(Switch to the live Vercel web app)*
1.  **Connect Camera:** Show the mobile phone streaming to the dashboard.
2.  **Normal Fabric:** Show the system smoothly adapting to the fabric with a low anomaly score.
3.  **Defect Introduction:** Introduce a broken thread or stain.
4.  **Detection & Heatmap:** Show the bounding box and Jet-colormap heatmap isolating the defect in real-time.
5.  **WhatsApp Alert:** Show the automatic Twilio notification arriving on your phone.

## Slide 8: Future Enhancements
*   Deploying on Edge Devices (e.g., NVIDIA Jetson Nano) for zero-latency offline processing.
*   Expanding the dataset to include specialized defects like Zari tarnishing.
*   Integration with physical loom machinery to auto-stop the machine upon defect detection.

## Slide 9: Q&A
**Questions?**
