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

    def detect_defect_via_template(self, preprocessed_frame, template_gray_patch, original_frame, threshold=0.7):
        """
        Uses OpenCV's cv2.matchTemplate Algorithm.
        
        It slides a small 'perfect' weaving template over the frame. 
        If the highest match score drops below the threshold, it means the 
        standard fabric pattern is broken, missing, or mismatched.
        
        Args:
            preprocessed_frame: The current grayscaled camera image.
            template_gray_patch: A small cropped template of "perfect" pattern.
            original_frame: The BGR frame for annotations.
            threshold: Minimum acceptable match percentage (0.0 to 1.0).
            
        Returns:
            has_defect (bool)
            defect_info (str)
            annotated_frame (numpy.ndarray)
        """
        has_defect = False
        defect_info = None
        annotated = original_frame.copy()
        
        # 1. Run the Template Matching algorithm
        # TM_CCOEFF_NORMED returns a grid of scores from -1.0 to 1.0
        res = cv2.matchTemplate(preprocessed_frame, template_gray_patch, cv2.TM_CCOEFF_NORMED)
        
        # 2. Extract the highest matching score and its location (and lowest)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # 3. Defect Logic:
        # If the BEST match on the screen is still lower than the required threshold,
        # it means the camera cannot find the perfect pattern anywhere (e.g. an anomaly took over)
        if max_val < threshold:
            has_defect = True
            defect_info = f"Template Mismatch (Max Match: {max_val*100:.1f}%)"
            
            # Since the pattern is broken, let's draw a box around the WORST matching area
            # where the defect most likely is physically located.
            h, w = template_gray_patch.shape
            top_left = min_loc  # min_loc is the coordinate of the worst match!
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            # Draw an Orange bounding box indicating a Template Failure
            cv2.rectangle(annotated, top_left, bottom_right, (0, 165, 255), 3) 
            cv2.putText(annotated, f"WARNING: {defect_info}", (max(top_left[0], 0), max(top_left[1]-10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            
        return has_defect, defect_info, annotated
