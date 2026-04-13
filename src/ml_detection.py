from ultralytics import YOLO
import cv2
import os

class MLDetector:
    """
    Core engine for detecting defects using Deep Learning.
    Wraps the Ultralytics YOLOv8 architecture for real-time inference.
    """
    def __init__(self, model_path="models/best.pt"):
        """
        Loads the Custom Trained YOLOv8 model for Fabric Defect Detection.
        """
        # Ensure the models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[Info] Loading Custom AI Model from {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"[Warning] Could not load {model_path}. Falling back to default.")
            self.model = YOLO("models/yolov8n.pt")
    
    def detect_defects(self, frame):
        """
        Runs YOLOv8 neural network inference on a single frame.
        
        Args:
            frame: The original BGR frame from the camera.
            
        Returns:
            has_defect (bool)
            defect_info (str)
            annotated_frame (numpy.ndarray) - Frame with YOLO drawn boxes
        """
        has_defect = False
        defect_type = None
        
        # Run inference (verbose=False keeps the terminal clean from frame-by-frame spam)
        results = self.model(frame, verbose=False)
        
        # YOLOv8 provides an incredibly easy way to plot the predicted bounding boxes
        annotated_frame = results[0].plot() 
        
        # results[0].boxes contains the list of all detected objects in this frame.
        # In a real deployed custom model, we look for classes like "Hole", "Color Mismatch".
        # Since the base model detects random objects (e.g. cup, cell phone), 
        # any detection is treated as an "anomaly" for prototype testing purposes.
        if len(results[0].boxes) > 0:
            has_defect = True
            
            # Fetch the name of the highest confidence class found
            class_id = int(results[0].boxes[0].cls[0].item())
            class_name = self.model.names[class_id]
            defect_type = f"ML Anomaly ({class_name})"
            
        return has_defect, defect_type, annotated_frame
