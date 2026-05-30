import cv2
import time

def test_cameras():
    print("Scanning for cameras...")
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera found at index {idx} (Resolution: {frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"⚠️ Camera opened at index {idx}, but couldn't read frame.")
            cap.release()
        else:
            print(f"❌ No camera at index {idx}")

test_cameras()
