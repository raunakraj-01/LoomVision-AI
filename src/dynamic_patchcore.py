"""
dynamic_patchcore.py
--------------------
Dynamic Sliding-Window PatchCore for Real-Time Saree Defect Detection.

Architecture
============
This module implements a **Dynamic PatchCore** anomaly detection engine
designed specifically for real-time loom monitoring where the fabric
pattern, colour, and texture change continuously (e.g. body → border →
pallu transitions on a saree).

Key innovations over standard PatchCore:
1.  **Sliding-Window Memory Bank**: Only remembers the last N seconds
    of "normal" fabric, so it naturally adapts when the weave pattern
    or colour palette changes (e.g. transitioning into the pallu).
2.  **Multi-Scale Feature Extraction**: Hooks into both Layer 2 and
    Layer 3 of a pretrained ResNet-18 to capture thread-level textures
    AND broader structural patterns simultaneously.
3.  **Adaptive Threshold**: Uses a running mean + standard deviation of
    recent anomaly scores to set a dynamic detection boundary. This
    avoids false positives when switching between saree sections.
4.  **Spatial Heatmap**: Maps anomaly scores back to the original frame
    to draw precise bounding boxes around the defect region.

Dependencies: torch, torchvision, opencv-python, numpy
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import time
from collections import deque


class DynamicPatchCore:
    """
    Real-time, adaptive PatchCore anomaly detector for multi-pattern
    saree defect detection on a running loom.

    Parameters
    ----------
    memory_window : int
        Number of clean frames whose features are kept in the sliding
        memory bank.  30 frames ≈ 2-3 seconds at typical webcam FPS.
    warmup_frames : int
        Minimum frames to accumulate before detection begins.
    base_threshold : float
        Initial anomaly score threshold (used before enough history
        exists to compute an adaptive one).
    adaptive_sigma : float
        Number of standard deviations above the running mean score
        to set the adaptive threshold.  Higher = less sensitive.
    """

    def __init__(
        self,
        memory_window: int = 30,
        warmup_frames: int = 8,
        base_threshold: float = 0.55,
        adaptive_sigma: float = 4.0,
    ):
        # ── Device selection ──
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load pretrained ResNet-18 (backbone) ──
        print("[PatchCore] Loading ResNet-18 backbone for Deep Feature Extraction…")
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        # ── Register forward hooks on Layer2 and Layer3 ──
        # Layer2 → 128-channel, 28×28 feature map (fine textures, threads)
        # Layer3 → 256-channel, 14×14 feature map (broader structures)
        self._hook_features: dict[str, torch.Tensor] = {}

        def _make_hook(name):
            def hook(module, inp, out):
                self._hook_features[name] = out
            return hook

        self.backbone.layer2.register_forward_hook(_make_hook("layer2"))
        self.backbone.layer3.register_forward_hook(_make_hook("layer3"))

        # ── Image preprocessing (ImageNet normalisation) ──
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # ── Memory bank config ──
        self.memory_window = memory_window
        self.warmup_frames = warmup_frames

        # The memory bank is a deque of per-frame patch feature tensors
        # Each entry is shape [N_patches, D]
        self._memory_deque: deque[torch.Tensor] = deque(maxlen=memory_window)
        self._frame_count: int = 0

        # ── Adaptive threshold state ──
        self.base_threshold = base_threshold
        self.adaptive_sigma = adaptive_sigma
        # Rolling history of per-frame max anomaly scores (for clean frames)
        self._score_history: deque[float] = deque(maxlen=120)

        # ── Grid size for spatial heatmap (Layer3: 14×14 after pooling) ──
        self._grid_h = 14
        self._grid_w = 14

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_warmed_up(self) -> bool:
        return self._frame_count >= self.warmup_frames

    @property
    def warmup_progress(self) -> float:
        return min(self._frame_count / max(self.warmup_frames, 1), 1.0)

    @property
    def memory_size(self) -> int:
        """Total number of patch vectors in the memory bank."""
        return sum(t.size(0) for t in self._memory_deque)

    def reset(self):
        """Discard the memory bank and restart learning."""
        self._memory_deque.clear()
        self._score_history.clear()
        self._frame_count = 0
        print("[PatchCore] Memory bank reset — relearning from live stream.")

    # ------------------------------------------------------------------ #
    #  Feature Extraction                                                  #
    # ------------------------------------------------------------------ #

    def _extract_patch_features(self, frame: np.ndarray) -> torch.Tensor:
        """
        Run the frame through ResNet-18 and extract a combined multi-scale
        feature vector for every spatial patch.

        Returns
        -------
        patches : Tensor of shape [N_patches, D]
            N_patches = grid_h × grid_w (14 × 14 = 196)
            D = 128 + 256 = 384 (concatenated Layer2 + Layer3 features)
        """
        self._hook_features.clear()

        # Preprocess: BGR → RGB → Resize → Normalise
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.backbone(img_tensor)

        # Layer2: [1, 128, 28, 28] → pool to [1, 128, 14, 14]
        feat2 = self._hook_features["layer2"]
        feat2 = F.adaptive_avg_pool2d(feat2, (self._grid_h, self._grid_w))

        # Layer3: [1, 256, 14, 14] — already 14×14
        feat3 = self._hook_features["layer3"]
        feat3 = F.adaptive_avg_pool2d(feat3, (self._grid_h, self._grid_w))

        # Concatenate along channel dim → [1, 384, 14, 14]
        combined = torch.cat([feat2, feat3], dim=1)

        # Reshape to [196, 384] — one vector per spatial patch
        patches = combined.squeeze(0)  # [384, 14, 14]
        patches = patches.view(patches.size(0), -1).t()  # [196, 384]

        # L2-normalise for cosine similarity
        patches = F.normalize(patches, p=2, dim=1)
        return patches

    # ------------------------------------------------------------------ #
    #  Adaptive Threshold                                                  #
    # ------------------------------------------------------------------ #

    def _current_threshold(self) -> float:
        """
        Compute the current anomaly threshold adaptively.

        If we have enough score history from clean frames, use
        mean + adaptive_sigma × std.  Otherwise fall back to the
        static base_threshold.
        """
        if len(self._score_history) < 15:
            return self.base_threshold

        scores = np.array(self._score_history)
        adaptive_thresh = float(np.mean(scores) + self.adaptive_sigma * np.std(scores))
        
        # Enforce that the adaptive threshold never drops below the base threshold
        # to prevent it from becoming overly sensitive if the fabric happens to be very uniform
        return max(self.base_threshold, adaptive_thresh)

    # ------------------------------------------------------------------ #
    #  Core Detection                                                      #
    # ------------------------------------------------------------------ #

    def detect_defects(self, frame: np.ndarray):
        """
        Process a single camera frame in real time.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV / webcam.

        Returns
        -------
        has_defect : bool
        defect_info : str | None
        annotated_frame : np.ndarray
            The original frame with annotations drawn on it.
        """
        start_time = time.time()
        annotated = frame.copy()
        h_img, w_img = frame.shape[:2]

        # ── Extract deep features ──
        patch_features = self._extract_patch_features(frame)  # [196, 384]
        global_embedding = patch_features.mean(dim=0)  # [384] global frame representation

        self._frame_count += 1

        # ── Warmup phase: accumulate memory bank ──
        if not self.is_warmed_up:
            self._memory_deque.append(patch_features)
            pct = int(self.warmup_progress * 100)
            cv2.putText(
                annotated,
                f"AI Learning Saree Pattern... {pct}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 165, 0),
                2,
            )
            # Draw progress bar
            bar_w = int(w_img * 0.6)
            bar_x = int(w_img * 0.2)
            bar_y = 50
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14),
                          (40, 40, 40), -1)
            fill_w = int(bar_w * self.warmup_progress)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + 14),
                          (0, 200, 255), -1)
            return False, None, annotated, global_embedding, annotated

        # ── Build the flat memory bank tensor from the deque ──
        memory_bank = torch.cat(list(self._memory_deque), dim=0)  # [M, 384]

        # ── Compute per-patch anomaly scores (1 − max cosine similarity) ──
        # Efficient batched matrix multiply
        similarities = torch.mm(patch_features, memory_bank.t())  # [196, M]
        max_sims, _ = torch.max(similarities, dim=1)              # [196]
        anomaly_scores = (1.0 - max_sims).cpu().numpy()           # [196]

        # Reshape to spatial grid
        score_map = anomaly_scores.reshape(self._grid_h, self._grid_w)

        # ── Adaptive threshold ──
        threshold = self._current_threshold()

        # ── Draw spatial heatmap overlay (ALWAYS) ──
        heatmap = cv2.resize(
            score_map, (w_img, h_img), interpolation=cv2.INTER_LINEAR
        )
        # Normalise to threshold so anomaly > threshold becomes red
        hm_norm = np.clip(heatmap / max(threshold * 1.5, 1e-6), 0, 1)
        hm_uint8 = (hm_norm * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

        # Base overlay frame with just the heatmap (no bounding boxes)
        heatmap_overlay = cv2.addWeighted(frame.copy(), 0.6, hm_color, 0.4, 0)
        
        # We will draw annotations on top of the RAW frame directly (No heatmap overlay)
        annotated = frame.copy()

        # Frame-level score
        max_score = float(np.max(anomaly_scores))
        mean_score = float(np.mean(anomaly_scores))

        # ── Adaptive threshold ──
        threshold = self._current_threshold()

        # ── Decision ──
        has_defect = max_score > threshold

        if has_defect:
            defect_info = f"Deep Anomaly (score: {max_score:.3f}, thr: {threshold:.3f})"

            # ── Draw bounding box around worst region ──
            worst_idx = int(np.argmax(anomaly_scores))
            gy = worst_idx // self._grid_w
            gx = worst_idx % self._grid_w
            cell_w = w_img / self._grid_w
            cell_h = h_img / self._grid_h

            # Expand box slightly for visibility
            x1 = max(0, int(gx * cell_w) - 5)
            y1 = max(0, int(gy * cell_h) - 5)
            x2 = min(w_img, int((gx + 2) * cell_w) + 5)
            y2 = min(h_img, int((gy + 2) * cell_h) + 5)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                f"DEFECT DETECTED",
                (x1, max(y1 - 12, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Status bar
            cv2.putText(
                annotated,
                f"[!] {defect_info}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        else:
            defect_info = None

            # ── Update memory bank with this clean frame ──
            self._memory_deque.append(patch_features)

            # ── Update adaptive threshold history ──
            self._score_history.append(max_score)

            cv2.putText(
                annotated,
                f"Deep AI Active | Memory: {self.memory_size} patches",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        # ── FPS counter ──
        elapsed = time.time() - start_time
        fps = 1.0 / max(elapsed, 1e-6)
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f} | Thr: {threshold:.3f}",
            (10, h_img - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 0),
            1,
        )

        return has_defect, defect_info, annotated, global_embedding, heatmap_overlay
