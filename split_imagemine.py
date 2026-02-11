import os
import glob
import random
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "Data/ImageMine_Original"
DEST_ROOT = "Data/ImageMine_Split"
TRAIN_RATIO = 0.8  # 80% Train, 20% Test

def split_dataset():
    # 1. Setup Folders
    train_dir = os.path.join(DEST_ROOT, "train")
    test_dir = os.path.join(DEST_ROOT, "test")
    
    # Reset/Create folders
    if os.path.exists(DEST_ROOT): shutil.rmtree(DEST_ROOT)
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    # 2. Get Images
    print(f"🔍 Scanning {SOURCE_DIR}...")
    images = glob.glob(os.path.join(SOURCE_DIR, "**", "*.*"), recursive=True)
    images = [f for f in images if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    total_imgs = len(images)
    print(f"Found {total_imgs} images.")
    
    if total_imgs == 0:
        print("❌ No images found! Check your folder path.")
        return

    # 3. Shuffle & Split
    random.shuffle(images)
    split_idx = int(total_imgs * TRAIN_RATIO)
    
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]
    
    # 4. Copy Files
    print(f"📦 Copying {len(train_imgs)} to Train and {len(test_imgs)} to Test...")
    
    for img in tqdm(train_imgs, desc="Copying Train"):
        shutil.copy(img, os.path.join(train_dir, os.path.basename(img)))
        
    for img in tqdm(test_imgs, desc="Copying Test"):
        shutil.copy(img, os.path.join(test_dir, os.path.basename(img)))

    print("\n✅ SPLIT COMPLETE!")
    print(f"   Train Set: {train_dir} ({len(train_imgs)} images)")
    print(f"   Test Set:  {test_dir} ({len(test_imgs)} images)")
    print("   (These are the ones you keep hidden for the final report!)")

if __name__ == "__main__":
    split_dataset()