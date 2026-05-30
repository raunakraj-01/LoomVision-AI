import unittest
from unittest.mock import patch
import numpy as np
import cv2
from src.prediction_engine import PredictionEngine

class TestPredictionEngine(unittest.TestCase):
    def setUp(self):
        # Use opencv engine for faster testing without waiting for PatchCore/ResNet warmup
        self.engine = PredictionEngine(engine_type="opencv")
        self.frame_shape = (480, 640, 3)
        self.clean_frame = np.ones(self.frame_shape, dtype=np.uint8) * 200
        
        self.defective_frame = self.clean_frame.copy()
        # Draw a big black circle to trigger structural defect in OpenCV
        cv2.circle(self.defective_frame, (320, 240), 50, (0, 0, 0), -1)

    def test_clean_frame_processing(self):
        """Test that a clean frame passes without defect."""
        result = self.engine.process_frame(self.clean_frame)
        self.assertFalse(result.has_defect)
        self.assertIsNone(result.defect_type)
        self.assertEqual(result.metadata["engine_used"], "none")

    @patch('src.heuristic_classifier.HeuristicClassifier.classify_defect')
    def test_defective_frame_processing(self, mock_classify):
        """Test that a structural defect is caught."""
        # Force classifier to return Broken Thread so it is NOT suppressed
        mock_classify.return_value = "Broken Thread"
        
        # Feed the defective frame 3 times to satisfy the SequenceModel debouncing
        self.engine.process_frame(self.defective_frame)
        self.engine.process_frame(self.defective_frame)
        result = self.engine.process_frame(self.defective_frame)
        
        self.assertTrue(result.has_defect)
        # Assuming the structural detector or heuristic classifier will flag it
        self.assertIsNotNone(result.defect_type)
        self.assertNotEqual(result.defect_type, "Wrinkle")

    @patch('src.heuristic_classifier.HeuristicClassifier.classify_defect')
    def test_wrinkle_suppression(self, mock_classify):
        """Test that anomalies classified as 'Wrinkle' are suppressed."""
        # Force the classifier to return Wrinkle
        mock_classify.return_value = "Wrinkle"
        
        # Process a defective frame 3 times (would normally trigger a defect)
        self.engine.process_frame(self.defective_frame)
        self.engine.process_frame(self.defective_frame)
        result = self.engine.process_frame(self.defective_frame)
        
        # Because it's classified as a Wrinkle, it should be suppressed
        self.assertFalse(result.has_defect)
        self.assertIsNone(result.defect_type)
        mock_classify.assert_called()

if __name__ == '__main__':
    unittest.main()
