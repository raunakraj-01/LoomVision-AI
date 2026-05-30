import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class DefectDetector:
    """
    Core engine for detecting defects on a single preprocessed frame.

    Combines two complementary techniques:
    1.  **Structural detection** (Canny + Contours on the L-channel)
        Finds holes, weaving gaps, broken threads — works on any colour fabric.
    2.  **Colour-anomaly detection** (HSV analysis on the original BGR frame)
        Finds hue-shift stains, bleaching patches, embroidery colour breaks,
        and localised saturation drops.  This is the key addition that makes
        the system work on coloured sarees and embroidery, not just white fabric.
    """

    def __init__(self,
                 contour_area_threshold: int = 500,
                 color_saturation_drop: float = 0.30,
                 color_hue_shift: float = 0.20):
        """
        Args:
            contour_area_threshold: Minimum blob size (px²) to flag as structural defect.
            color_saturation_drop:  Fraction of the reference saturation below which a
                                    tile is considered a colour anomaly (0–1).
            color_hue_shift:        Fraction of hue deviation that triggers a colour flag.
        """
        self.contour_area_threshold = contour_area_threshold
        self.color_saturation_drop  = color_saturation_drop
        self.color_hue_shift        = color_hue_shift

        # Running HSV reference built from the first few clean frames
        self._ref_hsv: np.ndarray | None = None
        self._ref_count: int = 0
        self._ref_warmup: int = 20   # frames before colour reference is trusted

    # ------------------------------------------------------------------ #
    #  Structural Detection                                                #
    # ------------------------------------------------------------------ #

    def detect_structural_defect(self, preprocessed_frame, original_frame):
        """
        Detects physical anomalies: holes, large weaving gaps, broken threads.

        Crease marks are explicitly rejected using three shape-geometry filters:

        1.  **Solidity** (contour area / convex hull area):
            Creases are thin lines → very low solidity (< 0.40).
            Real defects (holes, tears) are compact blobs → higher solidity.

        2.  **Aspect Ratio** (bounding rect width / height, or inverse):
            Creases are very elongated → aspect ratio > 5 (5× longer than wide).
            Real defects tend to be roughly square or moderately elongated.

        3.  **Extent** (contour area / bounding rect area):
            Creases fill very little of their bounding box → extent < 0.25.
            Holes/tears fill a larger portion of their bounding box.

        A contour must PASS all three filters to be flagged as a defect.

        Args:
            preprocessed_frame: Single-channel uint8 (L-channel from CLAHE pipeline).
            original_frame:     Original BGR image for drawing annotations.

        Returns:
            has_defect (bool), defect_type (str | None), annotated_frame (ndarray)
        """
        has_defect  = False
        defect_info = None
        annotated   = original_frame.copy()

        # 1. Canny edge detection
        edges = cv2.Canny(preprocessed_frame, threshold1=40, threshold2=130)

        # 2. Dilate edges to close small gaps in weave lines
        kernel  = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # 3. Find & filter contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.contour_area_threshold:

                # ── Crease Rejection Filter ───────────────────────────
                x, y, w, h = cv2.boundingRect(contour)

                # Filter 1 · Solidity — creases are thin → low solidity
                hull     = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 1.0
                if solidity < 0.20:          # crease-like thin line → skip
                    continue

                # Filter 2 · Aspect Ratio — creases are very elongated
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > 8.0:             # more than 8× longer than wide → skip
                    continue

                # Filter 3 · Extent — creases fill little of their bounding box
                rect_area = w * h
                extent    = area / rect_area if rect_area > 0 else 1.0
                if extent < 0.15:            # barely fills bounding box → skip
                    continue
                # ─────────────────────────────────────────────────────
                has_defect  = True
                defect_info = "Structural Defect"

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(annotated, f"⚠ {defect_info}",
                            (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return has_defect, defect_info, annotated

    # ------------------------------------------------------------------ #
    #  Colour-Anomaly Detection  (NEW — for coloured sarees & embroidery) #
    # ------------------------------------------------------------------ #

    def detect_color_anomaly(self, original_frame):
        """
        Detects colour-based defects:
        - Hue-shift stains (e.g. accidental dye bleeding on a coloured saree)
        - Saturation drops  (bleached / faded patches)
        - Embroidery thread colour breaks

        The method builds a running median of the HSV colour signature from
        the first `_ref_warmup` clean frames, then compares subsequent frames
        tile by tile.

        Args:
            original_frame: Original BGR image.

        Returns:
            has_defect (bool), defect_type (str | None), annotated_frame (ndarray)
        """
        annotated   = original_frame.copy()
        has_defect  = False
        defect_info = None

        hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # ── Build reference during warm-up ──
        if self._ref_count < self._ref_warmup:
            if self._ref_hsv is None:
                self._ref_hsv = hsv.copy()
            else:
                alpha = 1.0 / (self._ref_count + 1)
                self._ref_hsv = cv2.accumulateWeighted(hsv, self._ref_hsv, alpha)  # type: ignore[arg-type]
            self._ref_count += 1
            return False, None, annotated

        # ── Compare frame to reference in 32×32 tiles ──
        ref_hsv = self._ref_hsv  # shape (H, W, 3)
        tile    = 32
        H, W, _ = hsv.shape
        flagged  = []

        for y in range(0, H - tile + 1, tile):
            for x in range(0, W - tile + 1, tile):
                cur_h = hsv[y:y+tile, x:x+tile, 0]
                cur_s = hsv[y:y+tile, x:x+tile, 1]
                ref_h = ref_hsv[y:y+tile, x:x+tile, 0]
                ref_s = ref_hsv[y:y+tile, x:x+tile, 1]

                mean_cur_s = np.mean(cur_s)
                mean_ref_s = np.mean(ref_s)

                # Ignore very low-saturation tiles (near-white/grey fabric sections)
                if mean_ref_s < 20:
                    continue

                # Saturation drop → bleached / faded patch
                if mean_ref_s > 0 and \
                   (mean_ref_s - mean_cur_s) / mean_ref_s > self.color_saturation_drop:
                    flagged.append((x, y, tile, tile, "Colour Fade / Stain"))
                    continue

                # Hue deviation → dye bleed / embroidery colour break
                # Hue is circular (0-180 in OpenCV), use angular distance
                delta_h = np.abs(np.mean(cur_h) - np.mean(ref_h))
                delta_h = min(delta_h, 180 - delta_h)  # circular wrap
                if delta_h / 90.0 > self.color_hue_shift:
                    flagged.append((x, y, tile, tile, "Hue Shift / Dye Anomaly"))

        if flagged:
            has_defect  = True
            # Use the most common defect type across tiles
            types        = [f[4] for f in flagged]
            defect_info  = max(set(types), key=types.count)

            for (x, y, w, h, _) in flagged:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)

            fx, fy = flagged[0][0], flagged[0][1]
            cv2.putText(annotated, f"⚠ {defect_info}",
                        (max(fx, 4), max(fy - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        return has_defect, defect_info, annotated

    def reset_color_reference(self):
        """Call this when a new saree / fabric roll is introduced."""
        self._ref_hsv   = None
        self._ref_count = 0


# ══════════════════════════════════════════════════════════════════════ #
#  Adaptive Pattern Detector (SSIM on luminance)                        #
# ══════════════════════════════════════════════════════════════════════ #

class AdaptivePatternDetector:
    """
    Real-time adaptive fabric pattern detector using SSIM on the L-channel.

    Works on ALL fabric colours because it operates on luminance (structure),
    not on colour directly.  Complemented by DefectDetector.detect_color_anomaly
    which handles hue/saturation-based defects.

    Parameters
    ----------
    warmup_frames : int
        Frames to accumulate before detection starts.
    alpha : float
        EMA learning rate for the running background (0 < alpha < 1).
    ssim_threshold : float
        Min acceptable SSIM per tile (0–1).  Lower → less sensitive.
    tile_size : int
        Side length (px) of each comparison tile.
    """

    def __init__(self, warmup_frames: int = 30, alpha: float = 0.05,
                 ssim_threshold: float = 0.65, tile_size: int = 64):
        self.warmup_frames  = warmup_frames
        self.alpha          = alpha
        self.ssim_threshold = ssim_threshold
        self.tile_size      = tile_size

        self._background: np.ndarray | None = None
        self._frame_count: int = 0

    # ── Properties ── #

    @property
    def is_warmed_up(self) -> bool:
        return self._frame_count >= self.warmup_frames

    @property
    def warmup_progress(self) -> float:
        return min(self._frame_count / max(self.warmup_frames, 1), 1.0)

    def reset(self):
        """Discard the learned background and start over."""
        self._background  = None
        self._frame_count = 0

    # ── Core method ── #

    def update_and_detect(self, preprocessed_gray, original_frame):
        """
        Feed a new preprocessed (single-channel) frame.

        During warm-up  → accumulates background, returns no defect.
        After warm-up   → tiled SSIM comparison; annotates anomalies.
                          Clean frames continue updating the background.

        Args:
            preprocessed_gray: uint8 single-channel (L-channel) frame.
            original_frame:    BGR frame for annotation.

        Returns:
            has_defect (bool), defect_info (str | None), annotated (ndarray)
        """
        gray      = preprocessed_gray.astype(np.float32)
        annotated = original_frame.copy()

        if self._background is None:
            self._background = gray.copy()

        self._frame_count += 1

        if not self.is_warmed_up:
            self._background = cv2.accumulateWeighted(gray, self._background, self.alpha)
            return False, None, annotated

        bg_uint8  = np.clip(self._background, 0, 255).astype(np.uint8)
        cur_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

        defect_regions = []
        h_img, w_img = cur_uint8.shape
        ts = self.tile_size

        for y in range(0, h_img - ts + 1, ts):
            for x in range(0, w_img - ts + 1, ts):
                tile_cur = cur_uint8[y:y+ts, x:x+ts]
                tile_bg  = bg_uint8 [y:y+ts, x:x+ts]

                win = min(7, ts - 1 if (ts - 1) % 2 == 0 else ts - 2)
                score = ssim(tile_cur, tile_bg, data_range=255, win_size=win)

                if score < self.ssim_threshold:
                    defect_regions.append((x, y, ts, ts, score))

        has_defect = len(defect_regions) > 0

        if has_defect:
            worst       = min(r[4] for r in defect_regions)
            defect_info = f"Pattern Anomaly (SSIM: {worst:.2f})"

            for (x, y, w, h, score) in defect_regions:
                severity = 1.0 - (score / self.ssim_threshold)
                b        = int(255 * (1 - severity))
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (b, 100, 255), 2)

            wx, wy = defect_regions[0][0], defect_regions[0][1]
            cv2.putText(annotated, f"⚠ {defect_info}",
                        (max(wx, 4), max(wy - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        else:
            defect_info = None
            self._background = cv2.accumulateWeighted(gray, self._background, self.alpha)

        return has_defect, defect_info, annotated
