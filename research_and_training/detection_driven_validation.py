import torch
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import comet_ml

# Import local modules
from src.model import AdaLOLIE_Net

class SafetyPerformanceEvaluator:
    def __init__(self, model_path="../checkpoints/adalolie_best.pth", yolo_path="../yolov8n.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load AdaLOLIE Enhancer (Dual-Attention Aware)
        self.enhancer = AdaLOLIE_Net().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.enhancer.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.enhancer.load_state_dict(checkpoint)
        self.enhancer.eval()
        
        # 2. Load YOLO Detector
        self.detector = YOLO(yolo_path)
        
        # 3. Comet Experiment for Result Tracking
        self.experiment = comet_ml.start(project_name="AdaLOLIE-Safety-Performance")
        
        # Target classes (0 = 'person' in COCO). Adjust if using custom mining weights.
        self.target_classes = [0] 
        self.test_dir = "../Data/MiningMix_Unified/test"
        self.output_dir = "../Output/Safety Performance Report"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_image(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: return None
        
        # Standardize to RGB for consistent enhancement
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
        # Pre-process (256x256)
        input_img = cv2.resize(img_rgb, (256, 256))
        img_tensor = (input_img / 255.0).astype(np.float32)
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Enhance
        with torch.no_grad():
            enhanced_tensor = self.enhancer(img_tensor)
        
        # Post-process & Restore to original resolution for YOLO
        enhanced_img = (enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        enhanced_high_res = cv2.resize(enhanced_img, (w, h))
        
        return img_rgb, enhanced_high_res

    def run_evaluation(self, num_samples=None):
        image_files = glob.glob(os.path.join(self.test_dir, "*.*"))
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if num_samples is not None and len(image_files) > num_samples:
            image_files = np.random.choice(image_files, num_samples, replace=False)
            
        results_data = []

        for img_path in tqdm(image_files, desc="Evaluating Safety Gain"):
            processed = self.process_image(img_path)
            if processed is None: continue
            raw_rgb, enh_rgb = processed
            
            # 1. Detection on Raw (Dark/Dusty)
            results_raw = self.detector(raw_rgb, verbose=False, conf=0.25)
            # 2. Detection on AdaLOLIE Enhanced
            results_enh = self.detector(enh_rgb, verbose=False, conf=0.25)
            
            conf_raw = [box.conf.item() for box in results_raw[0].boxes if int(box.cls) in self.target_classes]
            conf_enh = [box.conf.item() for box in results_enh[0].boxes if int(box.cls) in self.target_classes]
            
            results_data.append({
                "image": os.path.basename(img_path),
                "count_raw": len(conf_raw),
                "count_enh": len(conf_enh),
                "avg_conf_raw": np.mean(conf_raw) if conf_raw else 0,
                "avg_conf_enh": np.mean(conf_enh) if conf_enh else 0
            })

        # --- GENERATE RESEARCH METRICS ---
        df = pd.DataFrame(results_data)
        total_raw = df['count_raw'].sum()
        total_enh = df['count_enh'].sum()
        safety_gain = ((total_enh - total_raw) / (total_raw + 1e-6)) * 100

        # Log to Comet
        self.experiment.log_metric("Safety Gain (%)", safety_gain)
        self.experiment.log_metric("Objects Found Raw", total_raw)
        self.experiment.log_metric("Objects Found Enhanced", total_enh)

        # Plot Visual Comparison
        plt.figure(figsize=(10, 5))
        plt.bar(['Raw (Low-Light)', 'AdaLOLIE (Enhanced)'], [total_raw, total_enh], color=['#ff4b4b', '#00d4ff'])
        plt.title(f"Mining Safety: Object Detection Count\nSafety Gain: +{safety_gain:.1f}%")
        plt.savefig(os.path.join(self.output_dir, "safety_gain_comparison.png"))
        
        df.to_csv(os.path.join(self.output_dir, "safety_results.csv"))
        self.experiment.end()
        print(f"\n✅ Done! Safety Gain: {safety_gain:.1f}% increase in reliable detections.")

if __name__ == "__main__":
    evaluator = SafetyPerformanceEvaluator()
    evaluator.run_evaluation()