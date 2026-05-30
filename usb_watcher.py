"""
usb_watcher.py — LoomVision AI USB Phone Detector
--------------------------------------------------
Runs in the background alongside the Streamlit server.
When a mobile phone (Android or iPhone) is detected over USB,
it automatically opens http://localhost:5001 in the default browser.

Supports:
  • Android devices (USB Webcam / ADB mode)
  • iPhones / iPads (via usbmuxd)
  • Any USB device with "phone", "android", "iphone", "ipad" in its name

Usage (standalone):
    python3 usb_watcher.py

Usually launched via launch.py which starts both this and Streamlit.
"""

import subprocess
import webbrowser
import time
import sys
import threading
import signal
import json
import os
import pathlib

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
APP_URL         = "http://localhost:5001"
POLL_INTERVAL   = 2          # seconds between USB checks
REOPEN_COOLDOWN = 30         # seconds before re-opening after a disconnect/reconnect

# Shared status file read by the Streamlit app to show toast notifications
STATUS_FILE = pathlib.Path(__file__).parent / ".usb_status.json"

# Keywords that identify a mobile device or webcam in the USB device tree
DEVICE_KEYWORDS = [
    "iphone", "ipad", "ipod",
    "android", "samsung", "oneplus", "xiaomi", "redmi",
    "oppo", "vivo", "realme", "poco", "huawei", "pixel",
    "motorola", "nokia", "asus zenfone",
    "mobile", "smartphone",
    "webcam", "camera", "uvc", "logitech", "brio", "obsensor", "video", "capture"
]

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_usb_devices_mac() -> str:
    """Returns raw text output of all connected USB devices on macOS."""
    try:
        # ioreg is much more reliable than system_profiler on modern macOS
        result = subprocess.run(
            ["ioreg", "-p", "IOUSB"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.lower()
    except Exception:
        return ""


def _get_usb_devices_linux() -> str:
    """Returns raw text output of all connected USB devices on Linux."""
    try:
        result = subprocess.run(
            ["lsusb"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.lower()
    except Exception:
        return ""


def get_usb_device_text() -> str:
    """Platform-agnostic USB device listing."""
    import platform
    system = platform.system()
    if system == "Darwin":
        return _get_usb_devices_mac()
    elif system == "Linux":
        return _get_usb_devices_linux()
    else:
        # Windows — wmic
        try:
            result = subprocess.run(
                ["wmic", "path", "Win32_USBControllerDevice", "get", "Dependent"],
                capture_output=True, text=True, timeout=8
            )
            return result.stdout.lower()
        except Exception:
            return ""


def phone_connected() -> bool:
    """Returns True if any known mobile phone or webcam keyword is found in the USB device list."""
    text = get_usb_device_text()
    return any(kw in text for kw in DEVICE_KEYWORDS)


def _write_status(event: str, device_name: str = "Mobile Device"):
    """
    Write a small JSON status file that the Streamlit app polls.
    Streamlit reads this on each rerun and shows a toast notification.
    """
    payload = {
        "event":      event,          # "connected" | "disconnected"
        "device":     device_name,
        "timestamp":  time.time(),    # epoch — used to detect new events
    }
    try:
        STATUS_FILE.write_text(json.dumps(payload))
    except Exception as e:
        print(f"[USB Watcher] Could not write status file: {e}")


def open_browser():
    """Opens the LoomVision AI dashboard in the default browser."""
    print(f"[USB Watcher] 📱 Phone detected! Opening {APP_URL} …")
    webbrowser.open(APP_URL)


# ──────────────────────────────────────────────
# Main watcher loop
# ──────────────────────────────────────────────

def watch(stop_event: threading.Event):
    """
    Poll USB device list every POLL_INTERVAL seconds.
    When a phone is newly plugged in, open the browser.
    Implements a cooldown so it doesn't spam open tabs.
    """
    was_connected = False
    last_opened   = 0.0

    print("[USB Watcher] 👁  Monitoring USB for mobile phone or webcam connection…")
    print(f"[USB Watcher]    Will open {APP_URL} automatically when device detected.")

    while not stop_event.is_set():
        connected = phone_connected()

        if connected and not was_connected:
            # Phone just plugged in
            now = time.time()
            _write_status("connected")
            if now - last_opened > REOPEN_COOLDOWN:
                open_browser()
                last_opened = now

        if not connected and was_connected:
            print("[USB Watcher] 📴 Phone disconnected.")
            _write_status("disconnected")

        was_connected = connected
        time.sleep(POLL_INTERVAL)

    print("[USB Watcher] Stopped.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    stop_event = threading.Event()

    def _handle_signal(sig, frame):
        print("\n[USB Watcher] Shutdown signal received.")
        stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    watch(stop_event)


if __name__ == "__main__":
    main()
