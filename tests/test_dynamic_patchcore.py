import unittest
import cv2
import numpy as np
from src.dynamic_patchcore import DynamicPatchCore

class TestDynamicPatchCore(unittest.TestCase):
    def setUp(self):
        # Initialize patchcore with small warmup so tests run faster
        self.patchcore = DynamicPatchCore(memory_window=10, warmup_frames=3)
        self.frame_shape = (224, 224, 3)
        self.clean_frame = np.ones(self.frame_shape, dtype=np.uint8) * 200
        
        # Draw some arbitrary lines to simulate a clean fabric texture
        for i in range(0, 224, 20):
            cv2.line(self.clean_frame, (i, 0), (i, 224), (180, 180, 180), 1)
            cv2.line(self.clean_frame, (0, i), (224, i), (180, 180, 180), 1)

        self.defective_frame = self.clean_frame.copy()
        # Draw a big black circle to simulate a hole
        cv2.circle(self.defective_frame, (112, 112), 40, (0, 0, 0), -1)

    def test_initialization(self):
        """Test that PatchCore initializes and loads ResNet-18 properly."""
        self.assertIsNotNone(self.patchcore.backbone)
        self.assertFalse(self.patchcore.is_warmed_up)

    def test_warmup_and_detection(self):
        """Test the warmup phase and detection logic."""
        # 1. Warmup phase
        for _ in range(3):
            has_defect, defect_info, annotated_frame, global_embedding, heatmap = self.patchcore.detect_defects(self.clean_frame)
            # During warmup, it should return False for defects
            self.assertFalse(has_defect)
            self.assertIsNone(defect_info)

        self.assertTrue(self.patchcore.is_warmed_up)

        # 2. Test clean frame (after warmup)
        has_defect_clean, defect_info_clean, _, _, _ = self.patchcore.detect_defects(self.clean_frame)
        self.assertFalse(has_defect_clean)
        self.assertIsNone(defect_info_clean)

        # 3. Test defective frame (after warmup)
        has_defect_bad, defect_info_bad, _, _, _ = self.patchcore.detect_defects(self.defective_frame)
        self.assertTrue(has_defect_bad)
        self.assertIn("Deep Anomaly", str(defect_info_bad))

if __name__ == '__main__':
    unittest.main()
