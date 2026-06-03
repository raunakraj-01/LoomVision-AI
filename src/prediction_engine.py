"""
prediction_engine.py
--------------------
Unified prediction engine that wraps both the Deep Learning (PatchCore)
and Classical CV (OpenCV Structural/Color/Pattern) engines.
"""

from dataclasses import dataclass, field
import numpy as np
import time
import cv2

from src.preprocessing import apply_preprocessing, apply_color_preprocessing
from src.dynamic_patchcore import DynamicPatchCore
from src.detection import DefectDetector, AdaptivePatternDetector
from src.sequence_model import TemporalAnomalyDetector
from src.heuristic_classifier import HeuristicClassifier
from src.conveyor_state import ConveyorStateDetector

@dataclass
class PredictionResult:
    has_defect: bool
    defect_type: str | None
    confidence: float
    anomaly_score: float
    temporal_score: float
    heatmap: np.ndarray | None
    annotated_frame: np.ndarray
    bounding_boxes: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    is_calibrated: bool = True
    calibration_progress: float = 1.0

class PredictionEngine:
    def __init__(self, engine_type="auto"):
        """
        engine_type: "patchcore", "opencv", or "auto" (PatchCore with OpenCV fallback)
        """
        self.engine_type = engine_type
        self.fabric_type = "plain"  # "plain" or "embroidered"
        
        print("[PredictionEngine] Initializing DynamicPatchCore...")
        self.patchcore = DynamicPatchCore(warmup_frames=10)
        
        print("[PredictionEngine] Initializing OpenCV Detectors...")
        self.defect_detector = DefectDetector()
        self.defect_detector._ref_warmup = 10
        self.pattern_detector = AdaptivePatternDetector(warmup_frames=10)
        
        print("[PredictionEngine] Initializing LSTM Sequence Model...")
        self.temporal_detector = TemporalAnomalyDetector()
        
        print("[PredictionEngine] Initializing Heuristic Classifier...")
        self.classifier = HeuristicClassifier()

        print("[PredictionEngine] Initializing Conveyor Belt State Detector...")
        self.belt_detector = ConveyorStateDetector(
            blur_threshold=100.0,
            motion_threshold=5.0,
            settling_time=0.5,
        )

        self.calibration_frames = 10
        self.frames_processed = 0
        self.consecutive_defect_frames = 0
        self._belt_state = "stopped"  # cache for metadata

    def process_frame(self, frame: np.ndarray) -> PredictionResult:
        start_time = time.time()

        # ── Belt state detection ──
        self._belt_state = self.belt_detector.update(frame)
        is_stopped = self.belt_detector.is_stopped

        # If the belt is moving or settling, skip heavy AI processing.
        # Return a lightweight result with the raw frame.
        if not is_stopped:
            proc_time_ms = (time.time() - start_time) * 1000
            annotated = frame.copy()
            label = "Belt Moving — Skipping AI" if self._belt_state == "moving" else "Belt Settling..."
            color = (0, 165, 255) if self._belt_state == "moving" else (255, 200, 0)
            cv2.putText(
                annotated, label,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            # Reset debounce counter — belt movement invalidates the streak
            self.consecutive_defect_frames = 0
            return PredictionResult(
                has_defect=False,
                defect_type=None,
                confidence=0.0,
                anomaly_score=0.0,
                temporal_score=0.0,
                heatmap=None,
                annotated_frame=annotated,
                bounding_boxes=[],
                metadata={
                    "processing_ms": proc_time_ms,
                    "engine_used": "skipped",
                    "belt_state": self._belt_state,
                },
                is_calibrated=self.frames_processed >= self.calibration_frames,
                calibration_progress=min(1.0, self.frames_processed / self.calibration_frames),
            )

        # ── Belt is stopped — run full AI pipeline ──
        
        # Preprocessing
        preprocessed_gray = apply_preprocessing(frame)
        color_norm_frame = apply_color_preprocessing(frame)
        
        has_defect = False
        defect_type = None
        confidence = 0.0
        anomaly_score = 0.0
        temporal_score = 0.0
        heatmap = None
        annotated_frame = frame.copy()
        engine_used = "none"

        self.frames_processed += 1
        is_calibrated = self.frames_processed >= self.calibration_frames
        calibration_progress = min(1.0, self.frames_processed / self.calibration_frames)

        # 1. PatchCore Deep Learning Engine
        if self.engine_type in ["patchcore", "auto"]:
            engine_used = "patchcore"
            global_features = None
            pc_has_defect, pc_defect_info, pc_annotated, global_embedding, pc_heatmap_overlay = False, None, frame.copy(), None, frame.copy()
            try:
                pc_has_defect, pc_defect_info, pc_annotated, global_embedding, pc_heatmap_overlay = self.patchcore.detect_defects(frame, is_belt_stopped=True)
            except Exception as e:
                print(f"[PredictionEngine] PatchCore error: {e}")
            
            # Process with temporal detector
            t_score, is_t_warmed = self.temporal_detector.process_frame_embedding(global_embedding, pc_has_defect)
            temporal_score = t_score
            
            # If warmed up, trust PatchCore
            if self.patchcore.is_warmed_up:
                annotated_frame = pc_annotated
                
                # Check for temporal anomaly
                t_thresh = self.temporal_detector.get_baseline_threshold()
                has_temporal_defect = False
                if is_t_warmed and t_score > t_thresh:
                    has_temporal_defect = True
                
                has_defect = pc_has_defect or has_temporal_defect
                
                if has_defect:
                    if has_temporal_defect and not pc_has_defect:
                        defect_type = "Temporal Sequence Anomaly"
                        confidence = min(1.0, t_score / (t_thresh * 1.5))
                        anomaly_score = t_score / t_thresh
                        
                        # Draw temporal anomaly warning on the frame
                        cv2.putText(
                            annotated_frame,
                            f"[!] Temporal Sequence Anomaly (score: {t_score:.3f} > {t_thresh:.3f})",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255), # Orange
                            2
                        )
                        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 165, 255), 4)
                    else:
                        defect_type = "Deep Anomaly"
                    # Extract score and threshold from info string (e.g. "Deep Anomaly (score: 0.450, thr: 0.350)")
                    try:
                        parts = pc_defect_info.split("score: ")[1].split(",")
                        anomaly_score = float(parts[0])
                        threshold = float(parts[1].split("thr: ")[1].replace(")", ""))
                        # Simple confidence calculation
                        confidence = min(1.0, anomaly_score / (threshold * 1.5))
                    except:
                        confidence = 0.85
                        anomaly_score = 1.0
            
        # 2. Classical OpenCV Engine (used if auto + patchcore missed, or if explicitly chosen)
        if (self.engine_type == "opencv") or (self.engine_type == "auto" and not has_defect):
            # Try Structural (Skip if embroidered to avoid false positives on complex edges)
            struct_has, struct_info, struct_ann = False, None, frame.copy()
            if self.fabric_type != "embroidered":
                struct_has, struct_info, struct_ann = self.defect_detector.detect_structural_defect(
                    preprocessed_gray, annotated_frame if self.engine_type == "opencv" else frame.copy()
                )
            
            if struct_has:
                has_defect = True
                defect_type = "Structural Defect"
                annotated_frame = struct_ann
                confidence = 0.9
                anomaly_score = 1.0
                engine_used = "opencv_structural"
            else:
                # Try Color
                col_has, col_info, col_ann = self.defect_detector.detect_color_anomaly(
                    color_norm_frame if color_norm_frame is not None else frame
                )
                if col_has:
                    has_defect = True
                    defect_type = col_info or "Color Anomaly"
                    annotated_frame = col_ann
                    confidence = 0.85
                    anomaly_score = 0.8
                    engine_used = "opencv_color"
                else:
                    # Try Pattern
                    pat_has, pat_info, pat_ann = self.pattern_detector.update_and_detect(
                        preprocessed_gray, frame.copy()
                    )
                    if pat_has:
                        has_defect = True
                        defect_type = "Pattern Anomaly"
                        annotated_frame = pat_ann
                        confidence = 0.8
                        # Extract SSIM score
                        try:
                            anomaly_score = 1.0 - float(pat_info.split("SSIM: ")[1].replace(")", ""))
                        except:
                            anomaly_score = 0.5
                        engine_used = "opencv_pattern"

        # --- Rule-Based Defect Classification & Suppression ---
        if has_defect and defect_type != "Temporal Sequence Anomaly":
            # Run heuristic classifier to get specific defect type
            specific_defect = self.classifier.classify_defect(frame)
            defect_type = specific_defect
            
            # User request: Ignore minor defects (Stains), Wrinkles, and Embroidery (normal pattern variance)
            if defect_type in ["Stain", "Wrinkle", "Embroidery"]:
                has_defect = False
                defect_type = None
                heatmap = None
                annotated_frame = frame.copy() # Strip away bounding boxes and heatmap overlay
                
        # --- Temporal Debouncing ---
        # Belt is stopped so frames are stable; require only 2 consecutive
        # defect frames (reduced from 3) to avoid missing real defects on
        # a stop-and-go conveyor.
        if has_defect:
            self.consecutive_defect_frames += 1
            if self.consecutive_defect_frames < 2:
                has_defect = False
                defect_type = None
                heatmap = None
                annotated_frame = frame.copy() # Hide bounding box
        else:
            self.consecutive_defect_frames = 0
                
        proc_time_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            has_defect=has_defect,
            defect_type=defect_type,
            confidence=confidence,
            anomaly_score=anomaly_score,
            temporal_score=temporal_score,
            heatmap=heatmap,
            annotated_frame=annotated_frame,
            bounding_boxes=[], # Could parse from annotated frame or pass from engines
            metadata={
                "processing_ms": proc_time_ms,
                "engine_used": engine_used,
                "belt_state": self._belt_state,
            },
            is_calibrated=is_calibrated,
            calibration_progress=calibration_progress
        )
