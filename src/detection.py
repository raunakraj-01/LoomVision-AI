import cv2
import numpy as np

class DefectDetector:
    """
    Core engine for detecting defects on a single preprocessed frame.
    Uses classical Computer Vision techniques (Canny Edge Detection + Contouring).
    """
    def __init__(self, contour_area_threshold=500):
        # Minimum size (in pixels) for an anomaly to be flagged as a defect
        self.contour_area_threshold = contour_area_threshold

    def detect_structural_defect(self, preprocessed_frame, original_frame):
        """
        Detects physical anomalies like holes, large weaving gaps, or broken threads.
        
        Args:
            preprocessed_frame: The grayscaled/blurred image
            original_frame: The original BGR image for drawing annotations.
            
        Returns:
            has_defect (bool): Flag indicating if an issue was found
            defect_type (str): Name of the defect
            annotated_frame (numpy.ndarray): Frame drawn with warning boxes
        """
        has_defect = False
        defect_info = None
        annotated = original_frame.copy()
        
        # 1. Edge Detection: Find sharp transitions in the image
        edges = cv2.Canny(preprocessed_frame, threshold1=50, threshold2=150)
        
        # 2. Dilation: Connect nearby edges to form solid blobs for easier contouring
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 3. Find Contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # If the blob is large enough, it's flagged as a defect
            if area > self.contour_area_threshold:
                has_defect = True
                defect_info = "Structural Defect"
                
                # Draw a bright red bounding rectangle around the defect
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 3) # Red box
                cv2.putText(annotated, f"WARNING: {defect_info}", (x, max(y-10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        return has_defect, defect_info, annotated

    def detect_pattern_defect(self, preprocessed_frame, reference_gray_frame, original_frame):
        """
        Detects color or pattern mismatches by comparing the current frame to a 'golden' reference.
        
        Args:
            preprocessed_frame: The current grayscaled/blurred image
            reference_gray_frame: The perfect grayscaled reference image
            original_frame: The original BGR image for annotations.
            
        Returns:
            has_defect (bool)
            defect_info (str)
            annotated_frame (numpy.ndarray)
        """
        has_defect = False
        defect_info = None
        annotated = original_frame.copy()
        
        # Ensure sizes match
        if preprocessed_frame.shape != reference_gray_frame.shape:
            # Resize preprocessed to match reference if cameras differ
            preprocessed_frame = cv2.resize(preprocessed_frame, (reference_gray_frame.shape[1], reference_gray_frame.shape[0]))
            
        # 1. Compute absolute difference between current frame and perfect reference
        diff = cv2.absdiff(reference_gray_frame, preprocessed_frame)
        
        # 2. Threshold the difference (pixels that differ by more than 30 intensity)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # 3. Dilate and find contours of the differences
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.contour_area_threshold * 2: # Require a slightly larger area for pattern mismatch
                has_defect = True
                defect_info = "Pattern/Color Mismatch"
                
                # Draw yellow bounding rectangle around the discrepancy
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 255), 3) # Yellow box
                cv2.putText(annotated, f"WARNING: {defect_info}", (x, max(y-10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
        return has_defect, defect_info, annotated
