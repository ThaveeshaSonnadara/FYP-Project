import os
import glob

# Path to lightweight labels folder
LABELS_DIR = "Data/DsDPM_YOLO_Lightweight/labels"

def fix_extensions():
    print("🔍 Looking for misnamed label files...")
    # Find all files ending in .jpg inside the labels folder
    bad_files = glob.glob(os.path.join(LABELS_DIR, "**", "*.jpg"), recursive=True)
    
    if not bad_files:
        print("✅ No misnamed files found! Everything is .txt")
        return

    print(f"🛠️ Fixing {len(bad_files)} files...")
    for filepath in bad_files:
        # Replace the last 4 characters (.jpg) with .txt
        new_filepath = filepath[:-4] + ".txt"
        os.rename(filepath, new_filepath)
        
    print("🚀 All done! Your dataset is now ready for YOLO.")

if __name__ == "__main__":
    fix_extensions()