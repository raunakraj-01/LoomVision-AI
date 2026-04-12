# LoomVision AI: Machine Learning Training Guide

Currently, your system is using the **"yolov8n.pt" pre-trained model**. This is a powerful, generalized Machine Learning model that detects common objects (cellphones, cups, people). 
While this is great for proving the logic pipeline works without crashing, it has *not* been taught what "Fabric Defects" look like yet!

To achieve a true **A+ College Project / Real-World Demo**, you must fine-tune (train) your own YOLOv8 model and drop it into your project folder. Here is the step-by-step roadmap to doing that for free.

## Phase 1: Source a Dataset
You need images of fabric. Half should be perfect, half should be defective.
1. Create a free account on [Roboflow.com](https://roboflow.com/) or browse [Kaggle](https://www.kaggle.com/datasets/belkhirnacif/telecom-paristech-mri-fabric-defect-dataset).
2. Search for "Fabric Defect Dataset YOLO format". 
3. Download the dataset. It should contain a folder of images, and a folder of `.txt` label files (which contain the coordinates of where the defect is).

## Phase 2: Train the Model (Google Colab)
Since YOLO training requires a GPU, do not run this on your laptop. Use Google Colab (free).
1. Go to Google Colab and open a new Notebook.
2. Ensure you change your Runtime Type to **T4 GPU**.
3. Run the following code blocks in the notebook:

```python
# Block 1: Install Ultralytics
!pip install ultralytics

# Block 2: Run the Training Command
from ultralytics import YOLO

# Load the base model
model = YOLO('yolov8n.pt') 

# Train the model (Upload your data.yaml from Roboflow here)
# Epochs = 50 means it will review the materials 50 times to learn.
results = model.train(data='your_dataset/data.yaml', epochs=50, imgsz=640)
```

## Phase 3: Export & Inject
1. Once Colab finishes training, it will generate a file named **`best.pt`**.
2. Download `best.pt` onto your Mac.
3. Move `best.pt` into your `LoomVisionAI/models/` directory.
4. Open `/src/ml_detection.py` and change line 11:
   ```python
   # Change this:
   self.model = YOLO("models/yolov8n.pt")
   
   # To this:
   self.model = YOLO("models/best.pt")
   ```

Restart your Streamlit dashboard and you are officially running a custom-trained Fabric Defect Deep Learning application!
