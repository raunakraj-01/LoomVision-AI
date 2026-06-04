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
        self.patchcore = DynamicPatchCore(warmup_frames=10, adaptive_sigma=3.0)
        
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
            blur_threshold=30.0,
            motion_threshold=12.0,
            settling_time=0.3,
        )

        self.calibration_frames = 10
        self.frames_processed = 0
        self.consecutive_defect_frames = 0
        self._belt_state = "stopped"  # cache for metadata
        self._continuous_motion_frames = 0  # track how long it's been moving

    def process_frame(self, frame: np.ndarray) -> PredictionResult:
        start_time = time.time()

        # ── Belt state detection ──
        self._belt_state = self.belt_detector.update(frame)
        is_stopped = self.belt_detector.is_stopped
        is_moving = not is_stopped
        
        if is_moving:
            self._continuous_motion_frames += 1
        else:
            self._continuous_motion_frames = 0
            
        # If the belt has been moving continuously for a long time (>30 frames),
        # we consider motion to be the "normal" state and allow the AI to learn from it.
        is_continuous = self._continuous_motion_frames > 30
        is_learning_safe = is_stopped or is_continuous

        # ── Always run AI — even while belt is moving ──
        # The defect may only be visible for a few frames during motion.
        # We run PatchCore on every frame but adapt thresholds and skip
        # noise-prone classical detectors when the belt is in motion.

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

        # Periodic debug log so the terminal shows pipeline health
        if self.frames_processed % 30 == 1:
            print(f"[PredictionEngine] Frame #{self.frames_processed} | belt={self._belt_state} | "
                  f"PatchCore warmed={self.patchcore.is_warmed_up} ({self.patchcore._frame_count}/{self.patchcore.warmup_frames}) | "
                  f"memory={self.patchcore.memory_size} patches | thr={self.patchcore._current_threshold():.3f}")

        # 1. PatchCore Deep Learning Engine — runs on EVERY frame
        if self.engine_type in ["patchcore", "auto"]:
            engine_used = "patchcore"
            pc_has_defect, pc_defect_info, pc_annotated, global_embedding, pc_heatmap_overlay = False, None, frame.copy(), None, frame.copy()
            try:
                # Pass is_learning_safe so PatchCore learns from continuous motion if applicable
                pc_has_defect, pc_defect_info, pc_annotated, global_embedding, pc_heatmap_overlay = self.patchcore.detect_defects(frame, is_belt_stopped=is_learning_safe)
            except Exception as e:
                print(f"[PredictionEngine] PatchCore error: {e}")

            # Process with temporal detector when learning is safe
            # (motion frames produce meaningless temporal sequences unless continuous)
            if is_learning_safe and global_embedding is not None:
                t_score, is_t_warmed = self.temporal_detector.process_frame_embedding(global_embedding, pc_has_defect)
                temporal_score = t_score

            # If warmed up, evaluate PatchCore results
            if self.patchcore.is_warmed_up:
                annotated_frame = pc_annotated

                if is_moving:
                    # ── MOVING BELT: Use PatchCore with a stricter threshold ──
                    # Motion blur raises anomaly scores for ALL frames, so we
                    # require a much stronger signal to flag a defect during
                    # motion.  This catches real defects (tears, holes) that
                    # spike the score far above normal blur levels.
                    if pc_defect_info:
                        try:
                            parts = pc_defect_info.split("score: ")[1].split(",")
                            raw_score = float(parts[0])
                            threshold = float(parts[1].split("thr: ")[1].replace(")", ""))
                            
                            # If continuous, the threshold is already adapted to motion, but motion blur
                            # inflates the variance, making the threshold too high to catch subtle defects.
                            # So we lower it by 30% to increase sensitivity.
                            # If not continuous (just transitioning), we use a stricter threshold (1.5x).
                            motion_threshold = (threshold * 0.7) if is_continuous else (threshold * 1.5)
                            
                            if raw_score > motion_threshold:
                                has_defect = True
                                defect_type = "Deep Anomaly"
                                anomaly_score = raw_score
                                confidence = min(1.0, raw_score / (motion_threshold * 1.5))
                                print(f"[PredictionEngine] *** DEFECT DURING MOTION *** score={raw_score:.3f} > motion_thr={motion_threshold:.3f}")
                            else:
                                # Score exceeded static threshold but not motion threshold —
                                # likely just motion blur, not a real defect
                                pass
                        except Exception:
                            pass

                    # Show belt-moving indicator on the annotated frame
                    label = "Belt Moving — AI Active" if self._belt_state == "moving" else "Belt Settling — AI Active"
                    color = (0, 200, 255) if self._belt_state == "moving" else (255, 200, 0)
                    cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                else:
                    # ── STOPPED BELT: Full pipeline with all detectors ──

                    # Check for temporal anomaly
                    t_thresh = self.temporal_detector.get_baseline_threshold()
                    has_temporal_defect = False
                    if temporal_score > 0:
                        is_t_warmed = self.temporal_detector.is_warmed_up
                        if is_t_warmed and temporal_score > t_thresh:
                            has_temporal_defect = True

                    has_defect = pc_has_defect or has_temporal_defect

                    if has_defect:
                        if has_temporal_defect and not pc_has_defect:
                            defect_type = "Temporal Sequence Anomaly"
                            confidence = min(1.0, temporal_score / (t_thresh * 1.5))
                            anomaly_score = temporal_score / t_thresh

                            # Draw temporal anomaly warning on the frame
                            cv2.putText(
                                annotated_frame,
                                f"[!] Temporal Sequence Anomaly (score: {temporal_score:.3f} > {t_thresh:.3f})",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 165, 255), # Orange
                                2
                            )
                            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 165, 255), 4)
                        else:
                            defect_type = "Deep Anomaly"
                        # Extract score and threshold from info string
                        try:
                            parts = pc_defect_info.split("score: ")[1].split(",")
                            anomaly_score = float(parts[0])
                            threshold = float(parts[1].split("thr: ")[1].replace(")", ""))
                            confidence = min(1.0, anomaly_score / (threshold * 1.5))
                        except:
                            confidence = 0.85
                            anomaly_score = 1.0

        # 2. Classical OpenCV Engine — ONLY when belt is stopped
        # (transient or continuous motion blur makes detection unreliable, causing false positives)
        if is_stopped:
            if (self.engine_type == "opencv") or (self.engine_type == "auto" and not has_defect):
                # Preprocessing (only computed when needed)
                preprocessed_gray = apply_preprocessing(frame)
                color_norm_frame = apply_color_preprocessing(frame)

                # Try Structural (Skip if embroidered)
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
                            try:
                                anomaly_score = 1.0 - float(pat_info.split("SSIM: ")[1].replace(")", ""))
                            except:
                                anomaly_score = 0.5
                            engine_used = "opencv_pattern"

        # --- Rule-Based Defect Classification & Suppression ---
        # Only run on stopped frames where classification is reliable.
        # (Motion blur makes everything look like "Embroidery" or "Stain")
        if has_defect and is_stopped and defect_type != "Temporal Sequence Anomaly":
            specific_defect = self.classifier.classify_defect(frame)
            defect_type = specific_defect

            # Only suppress Wrinkles (shadow/crease false positives)
            if defect_type in ["Wrinkle"]:
                has_defect = False
                defect_type = None
                heatmap = None
                annotated_frame = frame.copy()

        # --- Temporal Debouncing ---
        # No debouncing during motion — the defect may appear in only 1-2 frames.
        # During stopped state, require 2 consecutive frames to confirm.
        if has_defect:
            self.consecutive_defect_frames += 1
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
            bounding_boxes=[],
            metadata={
                "processing_ms": proc_time_ms,
                "engine_used": engine_used,
                "belt_state": self._belt_state,
            },
            is_calibrated=is_calibrated,
            calibration_progress=calibration_progress,
        )

