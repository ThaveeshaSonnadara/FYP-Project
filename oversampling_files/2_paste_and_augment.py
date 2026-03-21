import os
import json
import cv2
import random
import glob

# --- CONFIGURATION ---
IMG_DIR = r"D:\4th year\fyp\trash_ICRA19\trash_ICRA19\dataset\train"
ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\coco_ano_files\instances_train.json"
CROPS_DIR = r"D:\4th year\fyp\AquaClean_Project\extracted_minorities"

# Where to save the newly generated images and the new JSON
AUG_IMG_DIR = r"D:\4th year\fyp\AquaClean_Project\augmented_train_images"
AUG_ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\coco_ano_files\augmented_instances.json"

os.makedirs(AUG_IMG_DIR, exist_ok=True)

with open(ANN_FILE, 'r') as f:
    coco = json.load(f)

# Load crop paths
crop_files = glob.glob(os.path.join(CROPS_DIR, "*.jpg"))
print(f"Found {len(crop_files)} minority crops to paste.")

# Find the highest existing annotation and image IDs so we don't overlap
next_ann_id = max([ann['id'] for ann in coco['annotations']]) + 1
next_img_id = max([img['id'] for img in coco['images']]) + 1

# We will generate 4000 new synthetic images to balance the 6000 plastic objects
IMAGES_TO_GENERATE = 4000

# --- THE FIX: Create a static list of original images to use as backgrounds ---
original_images = coco['images'].copy()

for i in range(IMAGES_TO_GENERATE):
    # 1. Pick a random background image from the ORIGINAL dataset only
    bg_info = random.choice(original_images)
    bg_path = os.path.join(IMG_DIR, bg_info['file_name'])
    bg_img = cv2.imread(bg_path)
    
    if bg_img is None: continue
    bg_h, bg_w, _ = bg_img.shape

    # 2. Pick a random crop to paste
    crop_path = random.choice(crop_files)
    crop_img = cv2.imread(crop_path)
    crop_h, crop_w, _ = crop_img.shape
    
    # Determine class based on filename (bio_123.jpg or rov_456.jpg)
    cat_id = 1 if "bio" in os.path.basename(crop_path) else 2

    # 3. Find a random (x, y) location to paste it
    if bg_w - crop_w <= 0 or bg_h - crop_h <= 0: continue
    
    paste_x = random.randint(0, bg_w - crop_w)
    paste_y = random.randint(0, bg_h - crop_h)

    # 4. Paste the crop onto the background
    bg_img[paste_y:paste_y+crop_h, paste_x:paste_x+crop_w] = crop_img

    # 5. Save the new composite image
    new_filename = f"synthetic_aug_{i}.jpg"
    cv2.imwrite(os.path.join(AUG_IMG_DIR, new_filename), bg_img)

    # 6. Add Image details to JSON
    coco['images'].append({
        "id": next_img_id,
        "file_name": new_filename,
        "width": bg_w,
        "height": bg_h
    })

    # 7. Copy existing annotations from the background image
    existing_anns = [a for a in coco['annotations'] if a['image_id'] == bg_info['id']]
    for old_ann in existing_anns:
        new_ann = old_ann.copy()       # Make a copy of the existing object
        new_ann['id'] = next_ann_id    # Give it a new unique annotation ID
        new_ann['image_id'] = next_img_id # Assign it to our new synthetic image
        coco['annotations'].append(new_ann)
        next_ann_id += 1

    # 8. Add the newly pasted object to JSON
    coco['annotations'].append({
        "id": next_ann_id,
        "image_id": next_img_id,
        "category_id": cat_id,
        "bbox": [paste_x, paste_y, crop_w, crop_h],
        "area": crop_w * crop_h,
        "iscrowd": 0
    })

    next_img_id += 1
    next_ann_id += 1

    if (i+1) % 500 == 0:
        print(f"Generated {i+1}/{IMAGES_TO_GENERATE} synthetic images...")

with open(AUG_ANN_FILE, 'w') as f:
    json.dump(coco, f)

print(f"Done! New annotation file saved to: {AUG_ANN_FILE}")