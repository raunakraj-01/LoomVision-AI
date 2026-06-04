"""
conveyor_state.py
-----------------
Detects the conveyor belt's state (moving / stopped / settling) using
pure computer-vision analysis — no Arduino serial communication needed.

Two complementary signals are fused:
1.  **Laplacian Variance (blur detection)**: Moving fabric produces motion
    blur, which dramatically lowers the Laplacian variance of the frame.
2.  **Frame Differencing (motion detection)**: Consecutive frames during
    belt motion have high pixel-level differences; when the belt is
    stopped the difference drops to near-zero (camera noise only).

The module also implements a short *settling* period after the belt
transitions from moving → stopped, to let mechanical vibrations die out
before the AI processes the frame.

Usage:
    detector = ConveyorStateDetector()
    state = detector.update(frame)   # "moving" | "settling" | "stopped"
"""

import cv2
import numpy as np
import time


class ConveyorStateDetector:
    """
    Real-time conveyor belt state detector.

    Parameters
    ----------
    blur_threshold : float
        Laplacian variance below this value is considered "blurry" (belt
        moving).  Typical range: 50–200 depending on camera and fabric.
        Lower = less sensitive.  Start with 100 and tune.
    motion_threshold : float
        Mean absolute frame difference above this value indicates motion.
        Typical range: 3–15.  Start with 5.
    settling_time : float
        Seconds to wait after belt stops before declaring "stopped".
        Allows mechanical vibration to dissipate.  0.3–0.8s typical.
    history_len : int
        Number of recent frames to consider for smoothing the decision.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        motion_threshold: float = 5.0,
        settling_time: float = 0.5,
        history_len: int = 5,
    ):
        self.blur_threshold = blur_threshold
        self.motion_threshold = motion_threshold
        self.settling_time = settling_time
        self.history_len = history_len

        self._prev_gray: np.ndarray | None = None
        self._state: str = "stopped"  # "moving" | "settling" | "stopped"
        self._stop_timestamp: float | None = None  # when belt first appeared to stop
        self._motion_history: list[bool] = []  # recent per-frame motion decisions

    @property
    def state(self) -> str:
        """Current belt state: 'moving', 'settling', or 'stopped'."""
        return self._state

    @property
    def is_stopped(self) -> bool:
        """True only when the belt is fully stopped (settling complete)."""
        return self._state == "stopped"

    @property
    def is_moving(self) -> bool:
        """True when the belt is in motion."""
        return self._state == "moving"

    @property
    def is_settling(self) -> bool:
        """True during the brief settling window after belt stops."""
        return self._state == "settling"

    # ------------------------------------------------------------------ #

    def _compute_blur_score(self, gray: np.ndarray) -> float:
        """Laplacian variance — higher = sharper image, lower = blurry."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_motion_score(self, gray: np.ndarray) -> float:
        """Mean absolute difference from previous frame."""
        if self._prev_gray is None:
            return 0.0
        diff = cv2.absdiff(gray, self._prev_gray)
        return float(np.mean(diff))

    # ------------------------------------------------------------------ #

    def update(self, frame: np.ndarray) -> str:
        """
        Analyse one frame and return the current belt state.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from the camera.

        Returns
        -------
        str
            One of: "moving", "settling", "stopped"
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_score = self._compute_blur_score(gray)
        motion_score = self._compute_motion_score(gray)

        # A frame is "in motion" only when BOTH signals agree:
        # low sharpness (motion blur) AND high pixel diff (actual movement).
        # Using OR was too aggressive — fabric textures can have inherently
        # low Laplacian variance, causing false "moving" detection.
        is_frame_moving = (
            blur_score < self.blur_threshold and motion_score > self.motion_threshold
        )

        # Store previous frame for next diff
        self._prev_gray = gray.copy()

        # Maintain a short history for smoothing
        self._motion_history.append(is_frame_moving)
        if len(self._motion_history) > self.history_len:
            self._motion_history.pop(0)

        # Majority vote: belt is moving if most recent frames agree
        moving_votes = sum(self._motion_history)
        majority_moving = moving_votes > len(self._motion_history) / 2

        # ── State machine ──
        now = time.time()

        if majority_moving:
            # Belt is moving
            self._state = "moving"
            self._stop_timestamp = None

        elif self._state == "moving":
            # Just transitioned from moving → settling
            self._state = "settling"
            self._stop_timestamp = now

        elif self._state == "settling":
            # Check if settling period has elapsed
            elapsed = now - (self._stop_timestamp or now)
            if elapsed >= self.settling_time:
                self._state = "stopped"

        # "stopped" stays "stopped" until motion is detected again

        return self._state

    def reset(self):
        """Reset state machine (e.g., on system restart)."""
        self._prev_gray = None
        self._state = "stopped"
        self._stop_timestamp = None
        self._motion_history.clear()

    def get_debug_info(self) -> dict:
        """Return internal scores for debugging / dashboard display."""
        return {
            "state": self._state,
            "motion_history": self._motion_history.copy(),
            "stop_timestamp": self._stop_timestamp,
        }
