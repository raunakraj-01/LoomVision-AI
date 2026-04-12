import unittest
import cv2
import numpy as np
from src.detection import DefectDetector
from src.preprocessing import apply_preprocessing

class TestFabricDefectDetection(unittest.TestCase):
    
    def setUp(self):
        """Initialize the detector before every test."""
        self.detector = DefectDetector(contour_area_threshold=200)
        
        # Create a synthetic "perfect" fabric frame (white/gray uniform background)
        self.perfect_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Create a synthetic "defective" frame with a black hole/tear
        self.defective_frame = self.perfect_frame.copy()
        cv2.circle(self.defective_frame, (320, 240), 50, (0, 0, 0), -1) # Black hole in the center

    def test_preprocessing(self):
        """Ensure preprocessing returns a valid grayscaled image."""
        preprocessed = apply_preprocessing(self.perfect_frame)
        self.assertIsNotNone(preprocessed)
        # Check if it was converted to a single channel (grayscale)
        self.assertEqual(len(preprocessed.shape), 2)

    def test_no_defect_detected(self):
        """Ensure a perfect fabric doesn't trigger false positives."""
        preprocessed = apply_preprocessing(self.perfect_frame)
        has_defect, defect_info, _ = self.detector.detect_structural_defect(preprocessed, self.perfect_frame)
        
        self.assertFalse(has_defect)
        self.assertIsNone(defect_info)

    def test_structural_defect_detected(self):
        """Ensure a torn/holed fabric correctly triggers the detector."""
        preprocessed = apply_preprocessing(self.defective_frame)
        has_defect, defect_info, _ = self.detector.detect_structural_defect(preprocessed, self.defective_frame)
        
        self.assertTrue(has_defect)
        self.assertEqual(defect_info, "Structural Defect")

if __name__ == '__main__':
    unittest.main()
