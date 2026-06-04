import eventlet
eventlet.monkey_patch()
import cv2
import numpy as np
from src.prediction_engine import PredictionEngine

engine = PredictionEngine(engine_type="auto")
frame = np.zeros((480, 640, 3), dtype=np.uint8)
print("Processing frame...")
try:
    engine.process_frame(frame)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
