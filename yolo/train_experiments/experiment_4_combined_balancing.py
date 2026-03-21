import os
import glob
import random
import numpy as np
from collections import Counter, defaultdict
from ultralytics import YOLO

LABEL_DIR = "Data/DsDPM_YOLO_Lightweight/labels/train"

class_counts = Counter()
class_to_images = defaultdict(list)

label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))

for label_file in label_files:
    with open(label_file) as f:
        classes = set()

        for line in f:
            cls = int(line.split()[0])
            class_counts[cls] += 1
            classes.add(cls)

    img_path = label_file.replace("labels", "images").replace(".txt", ".jpg")

    for c in classes:
        class_to_images[c].append(img_path)

# Compute weights
counts = np.array([class_counts[i] for i in sorted(class_counts.keys())])
weights = 1 / np.sqrt(counts)
weights = weights / weights.sum()

# Class-aware sampling
balanced_images = []
classes = list(class_to_images.keys())

BATCH_SIZE = 16
NUM_BATCHES = 2000

for _ in range(NUM_BATCHES):
    for _ in range(BATCH_SIZE):
        c = random.choice(classes)
        img = random.choice(class_to_images[c])
        balanced_images.append(img)

balanced_file = "balanced_train.txt"

with open(balanced_file, "w") as f:
    for img in balanced_images:
        f.write(img + "\n")

print("Balanced dataset created")

model = YOLO("yolov8n.pt")

model.train(
    data="Data/DsDPM_YOLO_Lightweight/data.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    fl_gamma=2.0,
    cls=weights.tolist(),
    name="exp4_combined_balancing"
)