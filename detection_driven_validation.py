import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from model import AdaLOLIE_Net
import numpy as np
import glob
import random
import os

MODEL_PATH = "checkpoints/adalolie_best.pth"
TEST_FOLDER = "Data/MiningMix_Unified/test"
INFERENCE_SIZE = (640, 640)
OUTPUT_DIR = "Output"

def run_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running validation on: {device}")

    enhancer = AdaLOLIE_Net().to(device)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            enhancer.load_state_dict(checkpoint['model_state_dict'])
        else:
            enhancer.load_state_dict(checkpoint)
        print(f"✅ Loaded AdaLOLIE model: {MODEL_PATH}")
    else:
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    enhancer.eval()

    detector = YOLO("yolov8n.pt") # default model

    # Setup specific test image or random
    img_path = "Output/test_image.jpg"
    # img_path = TEST_FOLDER + "/dsdpm_13934.jpg"
    
    # If using random from folder:
    # test_images = glob.glob(os.path.join(TEST_FOLDER, "*.*"))
    # test_images = [x for x in test_images if x.lower().endswith(('.jpg', '.png'))]
    # if not test_images:
    #     print("❌ No images found in test folder!")
    #     return
    # img_path = random.choice(test_images)
    
    print(f"Testing on image: {os.path.basename(img_path)}")

    # 1. Read Image (BGR)
    original = cv2.imread(img_path)
    if original is None:
        print(f"❌ Could not read image: {img_path}")
        return

    # Convert BGR to RGB for Model
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original_resized = cv2.resize(original_rgb, INFERENCE_SIZE)

    # 2. Enhance
    img_tensor = (original_resized / 255.0).astype(np.float32)
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)

    print(f"Input Max: {img_tensor.max()}, Min: {img_tensor.min()}") 

    with torch.no_grad():
        enhanced_tensor = enhancer(img_tensor)
    
    enhanced_img = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced_img = np.clip(enhanced_img * 255, 0, 255).astype(np.uint8)

    # SAVE THE IMAGE FOR POST-PROCESSING ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    save_path = os.path.join(OUTPUT_DIR, "enhanced_output_test.jpg")
    
    # Convert RGB back to BGR before saving with OpenCV!
    enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(save_path, enhanced_img_bgr)
    print(f"💾 Saved enhanced image to: {save_path}")
    # -----------------------------------------------

    # 3. Detect
    results_original = detector(original_resized, verbose=False, conf=0.25)
    results_enhanced = detector(enhanced_img, verbose=False, conf=0.25)

    count_org = len(results_original[0].boxes)
    count_enh = len(results_enhanced[0].boxes)

    # 4. Visualize
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    res_plotted_org = results_original[0].plot() 
    ax[0].imshow(cv2.cvtColor(res_plotted_org, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Original (Dark)\nDetections: {count_org}", fontsize=14, color='red')
    ax[0].axis('off')

    res_plotted_enh = results_enhanced[0].plot()
    ax[1].imshow(cv2.cvtColor(res_plotted_enh, cv2.COLOR_BGR2RGB))
    title_color = 'green' if count_enh > count_org else 'black'
    ax[1].set_title(f"AdaLOLIE Enhanced\nDetections: {count_enh}", fontsize=14, color=title_color)
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_validation()