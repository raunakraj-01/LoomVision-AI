# LoomVisionAI Project Parameters

This document outlines the key parameters and configuration values used across the different engines in the LoomVisionAI pipeline. It specifically highlights the values configured in the main `prediction_engine.py` which override module defaults.

## 1. Global Pipeline Settings (`PredictionEngine`)
*Location: [src/prediction_engine.py](../src/prediction_engine.py)*

- **Engine Type**: `auto` (Uses PatchCore Deep Learning with classical OpenCV as a fallback).
- **Fabric Type**: `plain` (Alternative is `embroidered`).
- **Calibration Frames**: `10` (Frames needed to calibrate the overall system).
- **Continuous Motion Adaptability**: Belt motion is considered a "normal continuous state" after **`>30 frames`** of movement, allowing the AI to learn while the belt is moving.

## 2. Deep Learning Engine (`DynamicPatchCore`)
*This engine uses a ResNet-18 backbone with a sliding-window memory bank.*
*Module Location: [src/dynamic_patchcore.py](../src/dynamic_patchcore.py)*
*Initialization Location: [src/prediction_engine.py](../src/prediction_engine.py)*

- **Warmup Frames**: **`10`** *(Configured in `prediction_engine.py`; module default is 8)*. Minimum frames to build the initial memory bank.
- **Memory Window**: **`30`** frames *(Defined in `dynamic_patchcore.py`)*. It remembers the last 30 clean frames to adapt to fabric transitions (like saree body to border).
- **Base Threshold**: **`0.55`** *(Defined in `dynamic_patchcore.py`)*. The initial static anomaly score threshold.
- **Adaptive Sigma**: **`3.0`** *(Configured in `prediction_engine.py`; module default is 4.0)*. The number of standard deviations above the running mean to set the dynamic threshold. 
- **Motion Scaling**: During continuous motion, the threshold is scaled by **`0.7x`** to increase sensitivity. During transition states, it is scaled by **`1.5x`** to avoid false positives. *(Logic implemented in `prediction_engine.py`)*.

## 3. Temporal Sequence Model (`TemporalAnomalyDetector`)
*Uses an LSTM to predict feature vectors and catch temporal anomalies.*
*Module Location: [src/sequence_model.py](../src/sequence_model.py)*
*Initialization Location: [src/prediction_engine.py](../src/prediction_engine.py)*

- **LSTM Architecture**: Input dimension **`384`**, Hidden dimension **`128`**, **`1`** layer *(Defined in `sequence_model.py`)*.
- **Sequence Length**: **`10`** frames used as context to predict the next frame *(Defined in `sequence_model.py`)*.
- **Warmup Frames**: **`50`** frames required to establish a baseline loss *(Defined in `sequence_model.py`)*.
- **Learning Rate**: **`0.001`** (Uses Adam optimizer) *(Defined in `sequence_model.py`)*.

## 4. Classical CV Detectors (`OpenCV`)
*Only runs when the conveyor belt is stopped to prevent motion blur false positives.*

### Structural / Color Detector (`DefectDetector`)
*Module Location: [src/detection.py](../src/detection.py)*
*Initialization Location: [src/prediction_engine.py](../src/prediction_engine.py)*

- **Reference Warmup**: **`10`** frames *(Configured in `prediction_engine.py`; module default is 20)*.
- **Contour Area Threshold**: **`500 px²`** (Minimum blob size to flag a structural defect) *(Defined in `detection.py`)*.
- **Color Saturation Drop Threshold**: **`0.30`** (30% drop flags faded/bleached patches) *(Defined in `detection.py`)*.
- **Color Hue Shift Threshold**: **`0.20`** (Flags dye bleeding or embroidery breaks) *(Defined in `detection.py`)*.

### Pattern Detector (`AdaptivePatternDetector`)
*Module Location: [src/detection.py](../src/detection.py)*
*Initialization Location: [src/prediction_engine.py](../src/prediction_engine.py)*

- **Warmup Frames**: **`10`** frames *(Configured in `prediction_engine.py`; module default is 30)*.
- **SSIM Threshold**: **`0.65`** (Minimum Structural Similarity score per tile) *(Defined in `detection.py`)*.
- **EMA Learning Rate ($\alpha$)**: **`0.05`** (How fast the background model adapts) *(Defined in `detection.py`)*.
- **Tile Size**: **`64 px`** *(Defined in `detection.py`)*.

## 5. Conveyor Belt State Detection (`ConveyorStateDetector`)
*Determines whether the belt is moving, settling, or stopped.*
*Module Location: [src/conveyor_state.py](../src/conveyor_state.py)*
*Initialization Location: [src/prediction_engine.py](../src/prediction_engine.py)*

- **Blur Threshold (Laplacian Variance)**: **`30.0`** *(Configured in `prediction_engine.py`; module default is 100.0)*. Below this is considered motion blur.
- **Motion Threshold (Pixel Diff)**: **`12.0`** *(Configured in `prediction_engine.py`; module default is 5.0)*. Mean absolute difference above this flags movement.
- **Settling Time**: **`0.3 seconds`** *(Configured in `prediction_engine.py`; module default is 0.5s)*. Delay to let mechanical vibrations die out.
- **History Length**: **`5 frames`** (Number of recent frames used for smoothing the motion decision) *(Defined in `conveyor_state.py`)*.
