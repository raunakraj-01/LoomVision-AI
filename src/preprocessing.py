import cv2
import numpy as np

def apply_preprocessing(frame):
    """
    Colour-aware preprocessing pipeline for fabric frames.
    Works on ALL saree colours — silk, cotton, embroidery, zari —
    not just white/light fabrics.

    Steps:
    1.  Denoise in BGR space (preserves colour edges).
    2.  Convert to LAB colour space; extract the L (luminance) channel
        for structural/edge detection.
    3.  Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
        on L to enhance local contrast without blowing out colours.
    4.  Smooth L with a Gaussian blur to reduce camera noise before
        edge/contour analysis.

    Returns the processed single-channel (grayscale-equivalent) image
    so it is a drop-in replacement for the old function while now
    correctly handling coloured fabric.
    """
    if frame is None:
        return None

    # ── 1. Light denoising in BGR (keeps colour-edge sharpness) ──────
    denoised = cv2.GaussianBlur(frame, (3, 3), 0)

    # ── 2. LAB: L channel carries luminance, robust to colour bias ────
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    # ── 3. CLAHE: adaptive contrast on L only ─────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    # ── 4. Final denoise on the enhanced L channel ────────────────────
    blurred = cv2.GaussianBlur(l_eq, (9, 9), 0)

    return blurred


def apply_color_preprocessing(frame):
    """
    Returns a colour-normalised BGR frame suitable for colour-aware
    defect checks (e.g. hue-shift stains on coloured sarees).

    Steps:
    1.  Denoise.
    2.  Convert to HSV; apply CLAHE to V channel to normalise lighting.
    3.  Convert back to BGR.
    """
    if frame is None:
        return None

    denoised = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)

    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
