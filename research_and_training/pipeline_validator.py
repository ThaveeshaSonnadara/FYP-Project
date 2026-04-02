import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from model_zero_dce_based import AdaLOLIE_Net
import yaml

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ADALOLIE_WEIGHTS = "weights/adalolie_best.pth"
YOLO_WEIGHTS = "weights/best.pt"
DATASET_YAML = "dsdpm66.yaml"

# UPDATE THESE PATHS to your local val folder
VAL_IMAGES_DIR = "datasets/dsdpm66/images/val" 
ENHANCED_VAL_DIR = "datasets/dsdpm66/images/val_enhanced"

def enhance_validation_set():
    """Step 1: Sequential Enhancement of the Validation Dataset"""
    if not os.path.exists(ENHANCED_VAL_DIR):
        os.makedirs(ENHANCED_VAL_DIR)
    
    # Load AdaLOLIE (Zero-DCE + CBAM)
    model = AdaLOLIE_Net().to(DEVICE)
    checkpoint = torch.load(ADALOLIE_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"🚀 Enhancing validation set located at: {VAL_IMAGES_DIR}")
    
    image_files = [f for f in os.listdir(VAL_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with torch.no_grad():
        for img_name in tqdm(image_files):
            img_path = os.path.join(VAL_IMAGES_DIR, img_name)
            img = cv2.imread(img_path)
            # Normalization to [0, 1] as required by Zero-DCE
            img = (np.asarray(img) / 255.0)
            img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # Generate Enhanced Image
            _, enhanced_img, _ = model(img)
            
            # Convert back to CV2 format [0, 255]
            result = enhanced_img.squeeze().permute(1, 2, 0).cpu().numpy()
            result = (result * 255.0).clip(0, 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(ENHANCED_VAL_DIR, img_name), result)

def run_yolo_validation():
    """Step 2: Evaluate YOLOv8n on the Enhanced Validation Images"""
    print("\n🧐 Running YOLOv8n Validation on Enhanced Validation Imagery...")
    
    # Load original YAML to get classes and paths
    with open(DATASET_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Temporarily point the validation path to our ENHANCED folder
    data_config['val'] = ENHANCED_VAL_DIR 
    
    temp_yaml = "temp_enhanced_val.yaml"
    with open(temp_yaml, 'w') as f:
        yaml.dump(data_config, f)

    # Load and Validate YOLOv8n
    detector = YOLO(YOLO_WEIGHTS)
    # Ensure it uses the 'val' split
    metrics = detector.val(data=temp_yaml, split='val', device=DEVICE)
    
    # EXTRACTING YOUR REPORT VALUES
    print("\n" + "="*40)
    print("📈 ADA-LOLIE + YOLOv8n PIPELINE METRICS")
    print("="*40)
    # Extracting precise metrics from the Results object
    # Note: 'B' stands for Bounding Box
    p = metrics.results_dict['metrics/precision(B)']
    r = metrics.results_dict['metrics/recall(B)']
    map50 = metrics.results_dict['metrics/m_ap_50(B)']
    map50_95 = metrics.results_dict['metrics/m_ap_50_95(B)']

    print(f"Precision (P):    {p:.4f}")
    print(f"Recall (R):       {r:.4f}")
    print(f"mAP@0.5:          {map50:.4f}")
    print(f"mAP@0.5:0.95:     {map50_95:.4f}")
    print("="*40)
    
    # Clean up temp file
    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)

if __name__ == "__main__":
    enhance_validation_set()
    run_yolo_validation()