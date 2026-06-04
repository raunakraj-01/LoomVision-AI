import eventlet
eventlet.monkey_patch()
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open")
else:
    def read_cam():
        print("Reading frame...")
        ret, frame = cap.read()
        print(f"Read success: {ret}")
    
    eventlet.spawn(read_cam)
    eventlet.sleep(2)
