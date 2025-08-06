import random, glob, os
from ultralytics import YOLO

# 1. Grab all your modd2 images
img_paths = glob.glob("data/images/train/modd2/*.jpg")

# 2. Sample 20 of them
sample = random.sample(img_paths, 30)

# 3. Load your pretrained model
#model = YOLO("yolov8n.pt")  # or later: your finetuned weights
model = YOLO("yolov8s.pt")
# 4. Run inference & save results
for img in sample:
    print("Processing", img)
    results = model.predict(source=img, conf=0.1, save=True, project="runs/sample_predict", name="batch1", exist_ok=True)
    # prints out whatever detections it finds
