import cv2
import time

class CameraController:
    """
    Interfaces with the webcam to capture video frames for LoomVision AI.
    Handles initialization, reading, and releasing of the hardware camera.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def initialize(self):
        """Attempts to start the video capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[Error] Could not open video device {self.camera_index}")
            return False
        
        # Optional: Set camera width and height to standardize processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[Info] Camera initialized. Width: 640, Height: 480")
        return True

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
            print("[Info] Camera released.")

if __name__ == "__main__":
    # Test script: Open camera, display feed, press 'q' to quit
    cam = CameraController(0)
    if cam.initialize():
        print("Press 'q' in the window to exit.")
        while True:
            ret, frame = cam.get_frame()
            if not ret:
                print("Failed to grab frame.")
                break
            
            # Show live feed
            cv2.imshow('LoomVision AI - Camera Test', frame)
            
            # Press 'q' to break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
