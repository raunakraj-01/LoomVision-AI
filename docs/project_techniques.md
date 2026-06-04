# LoomVisionAI Project Techniques

This document outlines the core computer vision and deep learning techniques implemented in the LoomVisionAI pipeline.

## 1. Deep Learning Anomaly Detection (`DynamicPatchCore`)
*Used for detecting complex structural and pattern defects, specifically built to adapt to changing fabric textures (like saree body to border).*
- **Pre-trained CNN Feature Extraction**: Uses **ResNet-18** (Layer 2 and Layer 3) to extract both fine thread-level textures and broader structural patterns.
- **Sliding-Window Memory Bank**: Maintains a continuous, dynamically updating memory bank of "normal" fabric patches, discarding older references to adapt to new weave patterns.
- **Adaptive Thresholding**: Dynamically computes an anomaly threshold using the running mean and standard deviation of recent anomaly scores, avoiding false positives during pattern transitions.
- **Spatial Heatmap Mapping**: Calculates similarity using cosine distance to identify anomalies and maps these scores back to the original frame to generate precise defect bounding boxes.

## 2. Temporal Sequence Analysis (`TemporalAnomalyDetector`)
*Used to catch temporal anomalies that might not be obvious in a single static frame.*
- **Unsupervised Predictive LSTM**: A Long Short-Term Memory neural network trained continuously on the stream of global feature embeddings.
- **Online Learning**: The model learns "normal" motion sequences in real-time and flags sequences that deviate from the expected flow.

## 3. Classical Computer Vision Detectors (`OpenCV`)
*Lightweight, highly sensitive rule-based models that run when the conveyor belt is stopped to catch specific, known defect types.*

### Structural Detection
*Finds physical anomalies like holes, large weaving gaps, and broken threads.*
- **Canny Edge Detection**: Used on the preprocessed L-channel to find structural boundaries.
- **Morphological Operations**: Dilation is applied to close small gaps in natural weave lines.
- **Contour Filtering & Shape Geometry**: Filters out creases by checking geometric properties:
  - **Solidity** (contour area / convex hull area)
  - **Aspect Ratio** (width vs. height)
  - **Extent** (contour area / bounding box area)

### Color Anomaly Detection
*Identifies hue-shift stains, dye bleeding, and bleached patches.*
- **HSV Color Space Analysis**: Converts the BGR image to HSV for illumination-invariant color comparisons.
- **Running Background Signature**: Builds a running median HSV signature from clean frames.
- **Tiled Comparison**: Compares the live feed to the reference in tiles, checking for **Saturation Drops** and **Angular Hue Shifts**.

### Adaptive Pattern Detection
*A structure-agnostic fallback detector for pattern consistency.*
- **SSIM (Structural Similarity Index)**: Runs tiled SSIM on the luminance channel.
- **Exponential Moving Average (EMA)**: Maintains an EMA running background of the fabric to compare against.

## 4. Conveyor Belt State Detection (`ConveyorStateDetector`)
*Real-time tracking of the physical loom state to adjust AI sensitivity without relying on Arduino hardware signals.*
- **Laplacian Variance**: Used to measure image sharpness/blur. Low variance indicates motion blur.
- **Frame Differencing (Absolute Difference)**: Used to detect actual pixel-level movement between consecutive frames.
- **Temporal State Machine**: Employs a history buffer and majority voting system to transition between `moving`, `settling`, and `stopped` states.

## 5. Preprocessing
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Used to enhance the local contrast of the fabric, specifically on the L-channel (Luminance) to make structural defects more prominent regardless of fabric color.
