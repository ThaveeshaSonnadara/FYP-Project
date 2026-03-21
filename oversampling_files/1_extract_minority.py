import os
import json
import cv2

# --- CONFIGURATION ---
IMG_DIR = r"D:\4th year\fyp\trash_ICRA19\trash_ICRA19\dataset\train"
ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\coco_ano_files\instances_train.json"
OUTPUT_DIR = r"D:\4th year\fyp\AquaClean_Project\extracted_minorities"

# IDs based on your dictionary: {0: 'plastic', 1: 'bio', 2: 'rov'}
MINORITY_CLASSES = [1, 2] 

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ANN_FILE, 'r') as f:
    coco = json.load(f)

# Create a lookup dictionary for images
img_dict = {img['id']: img['file_name'] for img in coco['images']}

print("Extracting ROV and BIO objects...")
count = 0

for ann in coco['annotations']:
    cat_id = ann['category_id']
    if cat_id in MINORITY_CLASSES:
        img_filename = img_dict[ann['image_id']]
        img_path = os.path.join(IMG_DIR, img_filename)
        
        image = cv2.imread(img_path)
        if image is None: continue

        # Get bounding box [x, y, width, height]
        x, y, w, h = [int(v) for v in ann['bbox']]
        
        # Ensure coordinates are within image boundaries
        ih, iw, _ = image.shape
        x, y = max(0, x), max(0, y)
        w = min(w, iw - x)
        h = min(h, ih - y)

        if w > 10 and h > 10: # Only save if it's a reasonably sized crop
            crop = image[y:y+h, x:x+w]
            class_name = "bio" if cat_id == 1 else "rov"
            save_name = f"{class_name}_{count}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), crop)
            count += 1

print(f"Extraction complete! Saved {count} minority objects to {OUTPUT_DIR}")