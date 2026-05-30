import cv2
import numpy as np

class HeuristicClassifier:
    """
    Rule-based classifier to categorize detected anomalies into specific types:
    - Stain
    - Broken thread
    - Loose weave
    - Wrinkle (Suppressed)
    
    This acts as a second stage after the initial anomaly detection (PatchCore/OpenCV).
    """
    
    def __init__(self):
        self.classes = ["Broken thread", "Loose weave", "Stain", "Wrinkle", "Embroidery"]
        
    def classify_defect(self, frame, defect_mask=None, bounding_box=None):
        """
        Classify the defect given the original frame and a mask or bounding box 
        highlighting the anomalous region.
        
        Args:
            frame: Original BGR frame
            defect_mask: Optional binary mask of the defect
            bounding_box: Optional (x, y, w, h) of the defect
            
        Returns:
            str: One of the 4 classes.
        """
        if frame is None:
            return "Unknown"
            
        h, w = frame.shape[:2]
        
        # Determine ROI
        if bounding_box is not None:
            x, y, bw, bh = bounding_box
            # Ensure within bounds
            x = max(0, x)
            y = max(0, y)
            bw = min(w - x, bw)
            bh = min(h - y, bh)
            roi = frame[y:y+bh, x:x+bw]
        else:
            # If no mask/bbox, use center crop as heuristic approximation
            cy, cx = h // 2, w // 2
            roi = frame[max(0, cy-50):min(h, cy+50), max(0, cx-50):min(w, cx+50)]
            
        if roi.size == 0:
            return "Loose weave" # Default fallback
            
        # 1. Stain Detection (Color variance in HSV)
        # Convert ROI and full frame to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate mean color difference
        mean_hsv_roi = cv2.mean(hsv_roi)[:3]
        mean_hsv_frame = cv2.mean(hsv_frame)[:3]
        
        color_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(mean_hsv_roi, mean_hsv_frame)))
        if color_diff > 35:  # Significant color shift threshold
            return "Stain"
            
        # 2. Convert ROI to grayscale for structure analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Embroidery / Complex Pattern Detection
        # Embroidery introduces either high color variance OR extreme structural density (many stitches).
        # Plain cloth (even with defects) has a low color std (e.g. 5-10).
        std_color = np.std(roi, axis=(0, 1))
        
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / max(1, (roi.shape[0] * roi.shape[1]))
        
        if np.mean(std_color) > 15 or edge_density > 0.14:
            return "Embroidery"
        
        # 4. Broken Thread Detection (Sharp edges, high local contrast)
        if edge_density > 0.08: # Moderate-high concentration of sharp edges (single thread)
            return "Broken thread"
            
        # 4. Wrinkle / Shadow Detection (Low edge density)
        # Wrinkles and shadows do not produce sharp Canny edges. 
        # If the edge density is very low, it's a structural false-positive (wrinkle).
        if edge_density < 0.02:
            return "Wrinkle"
            
        # 6. Default Fallback (Moderate edge density, low color variance)
        return "Loose weave"
