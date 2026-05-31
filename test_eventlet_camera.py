import eventlet
eventlet.monkey_patch()
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera 0")
else:
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read frame of shape {frame.shape}")
    else:
        print("Opened camera but failed to read frame")
    cap.release()
