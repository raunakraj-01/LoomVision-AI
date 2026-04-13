# LoomVision AI: Kaggle to Colab Training Guide

Since you found a dataset on Kaggle, the fastest way to train your model is to connect Google Colab directly to Kaggle. That way, you don't even have to download the huge dataset to your laptop!

Follow this exact step-by-step guide to train your model.

## Step 1: Get your Kaggle API Token
To let Google Colab download your dataset from Kaggle automatically:
1. Go to [Kaggle.com](https://kaggle.com) and log in.
2. Click your profile picture (top right) -> **Settings**.
3. Scroll down to the **API** section and click **"Create New Token"**.
4. This will download a file named `kaggle.json`. Keep this file handy!

## Step 2: Set up Google Colab
1. Open [Google Colab](https://colab.research.google.com/) and create a "New Notebook".
2. Click **Runtime** (top menu) -> **Change runtime type**.
3. Set Hardware Accelerator to **T4 GPU** and hit Save.

## Step 3: Run the Training Code
Copy and paste the following Python code blocks into your Colab notebook cells and run them one by one.

### Cell 1: Upload your Kaggle Key
```python
from google.colab import files
# When you run this cell, it will ask you to upload a file. 
# Upload the kaggle.json file you downloaded in Step 1!
files.upload() 

# This creates a hidden kaggle folder and moves your key inside it securely
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Cell 2: Download your Dataset directly from Kaggle
*Go to your Kaggle Dataset URL. Click the three dots next to the "Download" button on the dataset page and click "Copy API Command". It will look like `kaggle datasets download -d username/dataset-name`. Paste it below!*

```python
# Replace this string with YOUR exact Kaggle command
!kaggle datasets download -d USERNAME/YOUR-DATASET-NAME

# Unzip the downloaded dataset folder
!unzip -q YOUR-DATASET-NAME.zip -d dataset/
```

### Cell 3: Install Ultralytics (YOLO)
```python
!pip install ultralytics
```

### Cell 4: Train the Machine Learning Model!
*(Note: Check inside your newly unzipped `dataset` folder in Colab. Look for a file called `data.yaml` or something similar, and update the string below!)*

```python
from ultralytics import YOLO

# Load the base "Nano" model
model = YOLO('yolov8n.pt') 

# Train the model on your Kaggle dataset!
# epochs=30 means it runs through the data 30 times. Increase to 50 if the accuracy is too low.
results = model.train(data='/content/dataset/data.yaml', epochs=30, imgsz=640)
```

## Step 4: Export your Trophy
1. Once Cell 4 finishes running (it might take 20-30 minutes), it will automatically save your trained model as `best.pt`. 
2. Colab will show you the exact folder path where it saved it (usually inside `/runs/detect/train/weights/best.pt`). 
3. Click the Folder icon on the far left side of Colab, navigate to that path, right-click `best.pt`, and click **Download**.
4. Move `best.pt` into your computer's `LoomVisionAI/models/` folder.
5. In your `app.py` or `ml_detection.py`, change `yolov8n.pt` to `best.pt`. Done!
