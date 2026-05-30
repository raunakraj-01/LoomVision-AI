"""
launch.py — LoomVision AI Unified Launcher
------------------------------------------
Starts two processes in parallel:
  1. Flask API Server (flask_server.py)
  2. USB watcher      (usb_watcher.py)

When your phone is plugged in via USB, the watcher
automatically opens http://localhost:5001 in your browser.

Run with:
    python3 launch.py
"""

import subprocess
import sys
import os
import time
import signal
import threading
import webbrowser
APP_URL    = "http://localhost:5001"
API_SERVER = [sys.executable, "-u", "flask_server.py"]
WATCHER    = [sys.executable, "usb_watcher.py"]

# ──────────────────────────────────────────────────────────────────
# Print a styled banner
# ──────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════╗
║          👁  LoomVision AI  —  Launcher                  ║
║──────────────────────────────────────────────────────────║
║  Starting API Web Server  +  USB phone watcher …         ║
║                                                          ║
║  → Open browser:  http://localhost:5001                  ║
║  → Plug your phone in via USB to open automatically      ║
║  → Press  Ctrl+C  to stop everything                     ║
╚══════════════════════════════════════════════════════════╝
"""


def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Poll until the Streamlit server is accepting connections."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def stream_output(proc: subprocess.Popen, label: str):
    """Forward a subprocess's stdout/stderr to the terminal with a label prefix."""
    for line in proc.stdout:
        print(f"[{label}] {line}", end="")


def main():
    print(BANNER)

    procs = []

    # ── Start API Web Server ──────────────────────────────────────
    print("[Launcher] Starting API Web Server…")
    st_proc = subprocess.Popen(
        API_SERVER,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    procs.append(st_proc)
    threading.Thread(target=stream_output, args=(st_proc, "API Server"), daemon=True).start()

    # ── Wait until server is ready, then open browser once ────────
    print("[Launcher] Waiting for server to be ready…")
    if wait_for_server(APP_URL):
        print(f"[Launcher] ✅ Server ready!  Opening {APP_URL} in browser…")
        webbrowser.open(APP_URL)
    else:
        print(f"[Launcher] ⚠️  Server didn't respond in time. Open {APP_URL} manually.")

    # ── Start USB watcher ─────────────────────────────────────────
    print("[Launcher] Starting USB phone watcher…")
    wb_proc = subprocess.Popen(
        WATCHER,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    procs.append(wb_proc)
    threading.Thread(target=stream_output, args=(wb_proc, "USB Watcher"), daemon=True).start()

    # ── Graceful shutdown on Ctrl+C ───────────────────────────────
    def shutdown(sig, frame):
        print("\n[Launcher] Shutting down…")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Keep alive ────────────────────────────────────────────────
    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"[Launcher] Child process exited. Shutting down entire application.")
                shutdown(None, None)
        time.sleep(1)


if __name__ == "__main__":
    main()
