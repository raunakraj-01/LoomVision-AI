# LoomVision AI: Roboflow to Colab Training Guide

Since you have found an Object Detection dataset on **Roboflow Universe**, this process is going to be incredibly easy. Roboflow formats everything perfectly into an automatic Python script.

Follow this exact step-by-step guide to train your YOLOv8 model.

## Step 1: Get your Roboflow Download Code
1. On the Roboflow website, go to the Dataset page you found.
2. Click **"Download Dataset"** (top right corner).
3. A popup will appear. For the Format, select **"YOLOv8"**.
4. Check the box that says **"show download code"** and click Continue.
5. It will give you a block of Python code (it usually starts with `!pip install roboflow` and contains your API key). **Copy this entire code block.**

## Step 2: Set up Google Colab
1. Open [Google Colab](https://colab.research.google.com/) and create a "New Notebook".
2. Click **Runtime** (top menu) -> **Change runtime type**.
3. Set Hardware Accelerator to **T4 GPU** and hit Save.

## Step 3: Run the Training Code
Copy and paste the following code blocks into your Colab notebook cells and run them one by one.

### Cell 1: Download your Dataset directly from Roboflow
*Paste the exact code block you copied from Step 1 here! It should look something like this:*

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_SECRET_KEY")
project = rf.workspace("workspace-name").project("fabric-defects")
version = project.version(1)
dataset = version.download("yolov8")
```

### Cell 2: Install YOLOv8
```python
!pip install ultralytics
```

### Cell 3: Train the Machine Learning Model!
*When Roboflow downloads the dataset in Cell 1, it automatically creates a `data.yaml` file that tells YOLO exactly where the images and bounding box text files are.*

```python
from ultralytics import YOLO

# Load the base YOLOv8 Nano model
model = YOLO('yolov8n.pt') 

# Start training! 
# We use data=dataset.location + '/data.yaml' to dynamically point to whatever folder Roboflow just created.
# epochs=30 means it runs through the data 30 times. (If it trains too fast and accuracy is low, increase to 50 or 100).
results = model.train(data=dataset.location + '/data.yaml', epochs=30, imgsz=640)
```

## Step 4: Export your Trained Model (best.pt)
1. Once Cell 3 finishes running (it might take 20-30 minutes), it will automatically save your newly trained model as a file named `best.pt`.
2. Look at the output text when it finishes. It will tell you where it saved it (usually inside the `/runs/detect/train/weights/best.pt` folder).
3. Click the **Folder Icon** 📁 on the far left side of your Colab screen. 
4. Navigate to `runs` -> `detect` -> `train` -> `weights`. 
5. Right-click on `best.pt` and click **Download**.
6. Move `best.pt` into your computer's `LoomVisionAI/models/` folder.
7. Open your `app.py` or `ml_detection.py` and replace `yolov8n.pt` with `best.pt`. 

Restart your Streamlit dashboard and switch it to Deep Learning mode. It will now draw bounding boxes perfectly!
