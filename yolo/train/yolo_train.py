import glob
import random
from collections import defaultdict
from ultralytics import YOLO

# Initialize Comet globally so YOLOv8 automatically detects it
os.environ["COMET_API_KEY"] = "BDA8b7jJyTFvOSAbOQnIOIOpf"
comet_ml.login(project_name="YOLO-Mining-Safety")

RUN_NAME = "exp4_custom_sampling"
CHECKPOINT_PATH = f"/content/drive/MyDrive/FYP_PROJECT/FYP_Checkpoints/YOLO_Runs/{RUN_NAME}/weights/last.pt"

LABEL_DIR = "/content/DsDPM_YOLO_Lightweight/labels/train"
class_to_images = defaultdict(list)

print("⚖️ Calculating data-driven class distribution...")
for label_file in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    with open(label_file) as f:
        classes = {int(line.split()[0]) for line in f if line.strip()}

    img_path = label_file.replace("labels", "images").replace(".txt", ".jpg")
    for c in classes:
        class_to_images[c].append(img_path)

# --- PURE DATA-DRIVEN OVERSAMPLING ---
# 1. Find the size of the majority class dynamically
max_class_size = max(len(imgs) for imgs in class_to_images.values())
print(f"📈 Majority class has {max_class_size} images. Balancing dataset...")

balanced_images = []
# 2. Oversample all classes to mathematically match the majority
for c, imgs in class_to_images.items():
    # random.choices allows duplicate sampling of rare images
    balanced_images.extend(random.choices(imgs, k=max_class_size))

# 3. Shuffle the giant list so YOLO doesn't train on the same class 10,000 times in a row
random.shuffle(balanced_images)

with open("/content/DsDPM_YOLO_Lightweight/balanced_train.txt", "w") as f:
    f.write("\n".join(balanced_images))

print(f"✅ Generated dynamic text dataloader with {len(balanced_images)} total images!")

# --- YAML GENERATION ---
yaml_content = """
path: /content/DsDPM_YOLO_Lightweight
train: balanced_train.txt
val: images/val
names:
  0: coal_miner
  1: drill_pipe
  2: drill_rig
  3: interaction_between_miner_and_drill_pipe
  4: mining_helmet
"""
with open("/content/DsDPM_YOLO_Lightweight/balanced_data.yaml", "w") as f:
    f.write(yaml_content.strip())

# --- HARDWARE-OPTIMIZED TRAINING LOOP ---
if os.path.exists(CHECKPOINT_PATH):
    print(f"🔄 Checkpoint found for {RUN_NAME}! Resuming training...")
    model = YOLO(CHECKPOINT_PATH)
    model.train(resume=True)
else:
    print(f"🆕 No checkpoint found. Starting {RUN_NAME} from scratch...")
    model = YOLO("yolov8n.pt")
    model.train(
      data="/content/DsDPM_YOLO_Lightweight/balanced_data.yaml",
      imgsz=640,
      epochs=100,
      batch=32,
      fraction=0.20,
      workers=8,
      cache=True,
      save_period=10,
      patience=20,
      optimizer='AdamW',
      cos_lr=True,
      project="/content/drive/MyDrive/FYP_PROJECT/FYP_Checkpoints/YOLO_Runs",
      name=RUN_NAME
    )

# 3. TELL COMET THE RUN IS OVER
print("✅ Training complete! Syncing final weights to Comet...")
comet_ml.end()