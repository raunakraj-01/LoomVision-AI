import os
import cv2
import json
import numpy as np
from src.prediction_engine import PredictionEngine

def compute_metrics(y_true, y_pred, classes):
    # Confusion Matrix
    # classes: list of class names
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1
            
    metrics = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
    return cm.tolist(), metrics

def evaluate():
    print("[Evaluator] Initializing Prediction Engine for Evaluation...")
    engine = PredictionEngine(engine_type="auto")
    
    base_dir = "data/validation"
    # Ground truth mapping based on directory names
    class_mapping = {
        "normal": "Normal",
        "broken_thread": "Broken thread",
        "loose_weave": "Loose weave",
        "stain": "Stain"
    }
    
    eval_classes = ["Normal", "Broken thread", "Loose weave", "Stain"]
    
    y_true = []
    y_pred = []
    
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        gt_class = class_mapping.get(folder)
        if not gt_class:
            continue
            
        for file in os.listdir(folder_path):
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder_path, file)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                    
                # Run prediction
                # Warm up is required for patchcore, so we run multiple times if needed, 
                # or for this evaluation script, we just process it.
                result = engine.process_frame(frame)
                
                if not result.has_defect:
                    pred_class = "Normal"
                else:
                    pred_class = result.defect_type if result.defect_type in eval_classes else "Loose weave" # Fallback mapping
                    
                y_true.append(gt_class)
                y_pred.append(pred_class)
                
    cm, metrics = compute_metrics(y_true, y_pred, eval_classes)

    if len(y_true) == 0:
        error_msg = "No validation images found. Please add labeled images to data/validation/ subdirectories."
        print(f"[Evaluator] {error_msg}")
        return {"error": error_msg}

    results = {
        "classes": eval_classes,
        "confusion_matrix": cm,
        "metrics": metrics
    }
    
    os.makedirs("output/reports", exist_ok=True)
    with open("output/reports/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("[Evaluator] Evaluation complete. Results saved to output/reports/evaluation_results.json")
    return results

if __name__ == "__main__":
    evaluate()
