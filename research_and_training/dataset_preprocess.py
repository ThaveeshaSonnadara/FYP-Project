import cv2
import os
import glob
import random
import shutil
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_ROOT = "../Data/MiningMix_Unified"
TARGET_SIZE = (256, 256) # <--- LOCKED TO 256 (Best for Speed/RO4)

# Dataset Allocations (Total ~15,000)
COUNTS = {
    "dsdpm": 14672,  # Huge chunk from my main dataset
    "imgmine": 400   # Half of my small real dataset
}

# Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# --- DEGRADATION FUNCTIONS ---
def add_coal_black_shift(image, intensity):
    """Simulates coal dust absorption (Black Shift)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Random intensity application
    v = v.astype(np.float32)
    v = v * (1.0 - intensity) 
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    # Slight Desaturation (Mines are monochromatic)
    s = s.astype(np.float32) * 0.8
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    final_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def add_dust_and_mist(image, density):
    """Simulates atmospheric scattering"""
    row, col, ch = image.shape
    noise = np.random.normal(0, 1, (row, col))
    
    # Cloud-like structure
    blur_size = 31 
    dust_mask = cv2.GaussianBlur(noise, (blur_size, blur_size), 0)
    cv2.normalize(dust_mask, dust_mask, 0, 255, cv2.NORM_MINMAX)
    dust_mask = dust_mask.astype(np.uint8)
    dust_layer = cv2.cvtColor(dust_mask, cv2.COLOR_GRAY2BGR)
    
    return cv2.addWeighted(image, 1 - density, dust_layer, density, 0)

def add_motion_blur(image, kernel_size=15):
    """Simulates camera/equipment motion"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_random_degradation(image):
    """Enhanced with 5 degradation types"""
    roll = random.random()
    
    if roll < 0.20:   # Dark
        return add_coal_black_shift(image, random.uniform(0.1, 0.4))
    elif roll < 0.40: # Dust
        return add_dust_and_mist(image, random.uniform(0.09, 0.3))
    elif roll < 0.60: # Both (Hard Mode)
        dark = add_coal_black_shift(image, random.uniform(0.1, 0.3))
        return add_dust_and_mist(dark, random.uniform(0.05, 0.2))
    elif roll < 0.80: # NEW: Motion Blur
        return add_motion_blur(image, kernel_size=random.randint(9, 21))
    else:             # Control
        return add_coal_black_shift(image, random.uniform(0.05, 0.15))

# --- MAIN PROCESSING ---
def process_and_split():
    # 1. Setup Folders
    for split in ['train', 'val', 'test']:
        path = os.path.join(OUTPUT_ROOT, split)
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path)

    all_processed_images = []

    # 2. Gather & Process DsDPM
    print("🔵 Processing DsDPM (Recursive Scan)...")
    dsdpm_files = glob.glob(os.path.join("../Data/DsDPM_Original", "**", "*.*"), recursive=True)
    valid_dsdpm = [f for f in dsdpm_files if f.lower().endswith(('.jpg', '.png'))]
    
    if len(valid_dsdpm) > COUNTS['dsdpm']:
        random.shuffle(valid_dsdpm)
        selected_dsdpm = valid_dsdpm[:COUNTS['dsdpm']]
    else:
        selected_dsdpm = valid_dsdpm
        
    for f in tqdm(selected_dsdpm, desc="DsDPM"):
        all_processed_images.append(('dsdpm', f))

    # 3. Gather & Process ImageMine
    print("🔵 Processing ImageMine...")
    imgmine_files = glob.glob(os.path.join("../Data/ImageMine_Split/train", "**", "*.*"), recursive=True)
    valid_imgmine = [f for f in imgmine_files if f.lower().endswith(('.jpg', '.png'))]
    
    for f in tqdm(valid_imgmine, desc="ImageMine"):
        all_processed_images.append(('imgmine', f))

    # 4. Shuffle & Split
    print(f"Total Images Selected: {len(all_processed_images)}")
    random.shuffle(all_processed_images)
    
    total = len(all_processed_images)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))
    
    # 5. Generate & Save
    count = 0
    for idx, (prefix, src_path) in enumerate(tqdm(all_processed_images, desc="Generating Dataset")):
        try:
            # Determine split
            if idx < train_end: split = 'train'
            elif idx < val_end: split = 'val'
            else: split = 'test'
            
            # Read
            img = cv2.imread(src_path)
            if img is None: continue
            
            # Resize
            img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # APPLY RANDOM DEGRADATION
            final_img = apply_random_degradation(img)
            
            # Save
            filename = f"{prefix}_{count:05d}.jpg"
            save_path = os.path.join(OUTPUT_ROOT, split, filename)
            cv2.imwrite(save_path, final_img)
            count += 1
            
        except Exception as e:
            pass
            
    print(f"\n✅ DATASET COMPLETE: {OUTPUT_ROOT}")
    print(f"   Train: {train_end} | Val: {val_end - train_end} | Test: {total - val_end}")

if __name__ == "__main__":
    process_and_split()