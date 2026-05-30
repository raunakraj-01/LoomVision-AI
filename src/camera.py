import cv2
import time
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

# Maximum camera index to probe when auto-detecting
_MAX_PROBE_INDEX = 5

# Keywords that indicate a camera is EXTERNAL (phone, USB webcam, etc.)
_EXTERNAL_KEYWORDS = [
    "android", "webcam", "uvc", "usb", "phone", "pixel", "samsung",
    "logitech", "brio", "capture", "external", "oneplus", "xiaomi",
    "redmi", "oppo", "vivo", "realme", "poco", "huawei", "motorola",
]
# Keywords that indicate a camera is BUILT-IN to the laptop
_BUILTIN_KEYWORDS = [
    "facetime", "macbook", "isight", "built-in", "internal", "imac",
]


class CameraController:
    """
    Interfaces with a webcam or USB-connected mobile phone camera.

    When camera_index is None (default), the controller uses macOS
    system_profiler to identify camera names, maps them to OpenCV
    indices, and prefers external cameras (phones, USB webcams) over
    the built-in laptop webcam.

    Explicit index:
        CameraController(camera_index=0)   → force specific camera
    Auto-detect:
        CameraController()    → scan all, prefer external
    IP Webcam:
        CameraController(source_type='ip_webcam') → use phone IP webcam
    """

    def __init__(self, camera_index=None, source_type="auto"):
        self.camera_index = camera_index  # None = auto-detect
        self.source_type = source_type    # 'auto', 'usb', 'ip_webcam'
        
        # Override source type from env if set
        env_source = os.getenv("CAMERA_SOURCE")
        if env_source and env_source in ["auto", "usb", "ip_webcam"]:
            self.source_type = env_source

        self.cap = None
        self._active_index = None         # resolved index or url after initialize()

    # ------------------------------------------------------------------ #

    def _open_and_warmup(self, idx: int) -> bool:
        """Opens a camera index and waits up to ~2.5s for a valid frame."""
        if not self._open(idx):
            return False
            
        # Warm up: Android/USB webcams often return False on the first few reads
        # while the hardware initializes. Give it up to 50 tries (2.5 seconds).
        for _ in range(50):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[Camera] ✅ Using camera index {idx}")
                self._active_index = idx
                return True
            time.sleep(0.05)
            
        # Failed to warm up
        self.cap.release()
        self.cap = None
        return False

    def _get_camera_names_mac(self) -> list:
        """
        Use system_profiler to get camera names in order.
        Returns a list of camera name strings, where the list index
        corresponds to the AVFoundation / OpenCV camera index.
        """
        try:
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True, text=True, timeout=10
            )
            names = []
            for line in result.stdout.splitlines():
                stripped = line.strip()
                # Camera names appear as top-level entries ending with ':'
                # but NOT lines like "Model ID:" or "Unique ID:"
                if stripped.endswith(":") and not stripped.startswith("Model ID") \
                   and not stripped.startswith("Unique ID") \
                   and not stripped.startswith("Camera"):
                    name = stripped.rstrip(":")
                    names.append(name)
                    print(f"[Camera] system_profiler: found '{name}' (index {len(names)-1})")
            return names
        except Exception as e:
            print(f"[Camera] Warning: system_profiler failed: {e}")
            return []

    def _pick_external_camera(self, names: list, working: list) -> int:
        """
        Given camera names and working indices, pick the external camera.
        Returns the chosen index.
        """
        # Score each working camera: external keywords get +1, built-in get -1
        best_idx = working[0]
        best_score = -999

        for idx in working:
            score = 0
            if idx < len(names):
                name_lower = names[idx].lower()
                for kw in _EXTERNAL_KEYWORDS:
                    if kw in name_lower:
                        score += 1
                for kw in _BUILTIN_KEYWORDS:
                    if kw in name_lower:
                        score -= 1
                print(f"[Camera] Index {idx} = '{names[idx]}' → score {score}")
            else:
                print(f"[Camera] Index {idx} = (unknown name) → score {score}")

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _get_working_indices(self) -> list:
        """Returns a list of available camera indices based on system_profiler and manual probes."""
        names = self._get_camera_names_mac()
        working = list(range(len(names))) if names else [0]
        
        # MacOS system_profiler is notoriously slow to update when a USB camera is plugged in.
        # If it only found the laptop webcam, actively probe index 1 just in case.
        if len(working) == 1:
            test_cap = cv2.VideoCapture(1)
            if test_cap.isOpened():
                working.append(1)
            test_cap.release()
            
        return working

    def initialize(self):
        """
        Opens the video device.

        Uses system_profiler to identify camera names and maps them to OpenCV
        indices. Prefers the external camera (phone / USB webcam) over the 
        built-in laptop camera. Or connects to an IP Webcam if specified.

        Returns True on success, False on failure.
        """
        if self.source_type == "ip_webcam":
            url = os.getenv("IP_WEBCAM_URL")
            if url:
                print(f"[Camera] Connecting to IP Webcam at {url}...")
                if self._open_and_warmup(url):
                    self._active_index = url
                    return True
                print(f"[Camera] ❌ Failed to connect to IP Webcam at {url}")
            else:
                print("[Camera] ❌ IP_WEBCAM_URL not found in .env. Falling back to auto.")
            
            # Fallback to auto if IP Webcam fails
            self.source_type = "auto"

        if self.camera_index is not None:
            if self._open_and_warmup(self.camera_index):
                return True
            print(f"[Camera] ❌ Explicit camera {self.camera_index} failed to warm up.")
            return False

        print("[Camera] Auto-detecting cameras via system_profiler…")
        names = self._get_camera_names_mac()
        working = list(range(len(names))) if names else [0]

        if not working:
            print("[Camera] ❌ No working camera found.")
            return False

        if len(working) == 1:
            chosen = working[0]
            label = names[chosen] if chosen < len(names) else "unknown"
            print(f"[Camera] Only one camera found: '{label}' at index {chosen}.")
        else:
            chosen = self._pick_external_camera(names, working)
            label = names[chosen] if chosen < len(names) else "unknown"
            print(f"[Camera] Selected external camera: '{label}' at index {chosen}.")

        return self._open_and_warmup(chosen)

    def switch_camera(self) -> bool:
        """Switches to the next available working camera, or toggles IP Webcam."""
        if self.source_type == "ip_webcam":
            # Switch back to USB
            print("[Camera] Switching from IP Webcam to USB")
            self.source_type = "usb"
            self.release()
            return self.initialize()
            
        working = self._get_working_indices()
        if not working or len(working) <= 1:
            print("[Camera] No other USB cameras to switch to. Toggling IP Webcam...")
            self.source_type = "ip_webcam"
            self.release()
            return self.initialize()
            
        current = self._active_index
        if current is None or isinstance(current, str):
            current = 0
            
        next_idx = None
        for i, w in enumerate(working):
            if w == current:
                next_idx = working[(i + 1) % len(working)]
                break
                
        if next_idx is None:
            next_idx = working[0]
            
        print(f"[Camera] Switching camera from index {current} to index {next_idx}")
        # Release current before opening next to avoid lock issues
        self.release()
        self._active_index = next_idx
        return self._open_and_warmup(next_idx)

    def get_source_info(self) -> dict:
        """Returns information about the current camera source."""
        return {
            "type": "ip_webcam" if isinstance(self._active_index, str) else "usb",
            "active_index": self._active_index,
            "status": "connected" if self.cap and self.cap.isOpened() else "disconnected"
        }

    def _open(self, idx) -> bool:
        """Open a single camera index or URL and apply standard settings."""
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap = cap
        return True

    # ------------------------------------------------------------------ #

    def get_frame(self):
        """
        Reads a single frame from the camera.

        Returns:
            success (bool): True if frame is read correctly.
            frame (numpy.ndarray): The captured image.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Releases the camera hardware."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[Camera] Released.")


# ──────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cam = CameraController()          # auto-detect
    if cam.initialize():
        print("Press 'q' to exit.")
        while True:
            ret, frame = cam.get_frame()
            if not ret:
                print("Failed to grab frame.")
                break
            cv2.imshow("LoomVision AI — Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cam.release()
        cv2.destroyAllWindows()

