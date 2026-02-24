import torch
import cv2
import os
import glob
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from model import AdaLOLIE_Net

# --- CONFIG ---
MODEL_PATH = "checkpoints/adalolie_best.pth"
TEST_DIR = "Data/MiningMix_Unified/test" # Ground Truth source
NUM_TEST_IMAGES = 100
SAVE_PTH = "Output/Evaluation Metrics/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REALISTIC DEGRADATION FUNCTIONS (Matches Training!) ---
def add_coal_black_shift(image, intensity):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = (v.astype(np.float32) * (1.0 - intensity)).clip(0, 255).astype(np.uint8)
    s = (s.astype(np.float32) * 0.8).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def add_dust_and_mist(image, density):
    noise = np.random.normal(0, 1, image.shape[:2])
    # Cloud-like structure
    dust = cv2.GaussianBlur(noise, (31, 31), 0)
    cv2.normalize(dust, dust, 0, 255, cv2.NORM_MINMAX)
    dust_layer = cv2.cvtColor(dust.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1 - density, dust_layer, density, 0)

def apply_random_degradation(image):
    """Exactly mimics the 'Hard Mode' your model learned"""
    roll = random.random()
    if roll < 0.30:   # Dark
        return add_coal_black_shift(image, random.uniform(0.1, 0.4))
    elif roll < 0.60: # Dust
        return add_dust_and_mist(image, random.uniform(0.09, 0.3))
    elif roll < 0.90: # Both
        dark = add_coal_black_shift(image, random.uniform(0.1, 0.3))
        return add_dust_and_mist(dark, random.uniform(0.05, 0.2))
    else:             # Control (Slight Dark)
        return add_coal_black_shift(image, random.uniform(0.05, 0.15))

# NEW: Test-Time Augmentation Helper
def apply_tta(model, img_tensor, device):
    """
    Applies Test-Time Augmentation (TTA) with 4 transforms:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. 90° Rotation
    
    Returns averaged prediction.
    """
    predictions = []
    
    with torch.no_grad():
        # 1. Original
        pred = model(img_tensor)
        predictions.append(pred)
        
        # 2. Horizontal Flip
        flipped_h = torch.flip(img_tensor, dims=[3])  # Flip width
        pred_h = model(flipped_h)
        pred_h = torch.flip(pred_h, dims=[3])  # Flip back
        predictions.append(pred_h)
        
        # 3. Vertical Flip
        flipped_v = torch.flip(img_tensor, dims=[2])  # Flip height
        pred_v = model(flipped_v)
        pred_v = torch.flip(pred_v, dims=[2])  # Flip back
        predictions.append(pred_v)
        
        # 4. 90° Rotation (clockwise)
        rotated = torch.rot90(img_tensor, k=1, dims=[2, 3])
        pred_rot = model(rotated)
        pred_rot = torch.rot90(pred_rot, k=-1, dims=[2, 3])  # Rotate back
        predictions.append(pred_rot)
    
    # Average all predictions
    return torch.mean(torch.stack(predictions), dim=0)

def evaluate(use_tta=True):
    print(f"🚀 Starting Evaluation on {DEVICE}...")
    print(f"   Test-Time Augmentation: {'✅ Enabled' if use_tta else '❌ Disabled'}")
    
    # 1. Load Model Safely
    model = AdaLOLIE_Net().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"✅ Loaded weights from {MODEL_PATH}")
    else:
        print(f"❌ Error: {MODEL_PATH} not found!")
        return
    model.eval()

    # 2. Prepare Images
    all_files = glob.glob(os.path.join(TEST_DIR, "**", "*.jpg"), recursive=True)
    if len(all_files) > NUM_TEST_IMAGES:
        all_files = np.random.choice(all_files, NUM_TEST_IMAGES, replace=False)

    psnr_scores, ssim_scores, inference_times = [], [], []
    psnr_improvement, ssim_improvement = [], []
    sample_vis = None

    print(f"📊 Testing {len(all_files)} images with Random Degradation...")

    for img_path in tqdm(all_files):
        try:
            # A. Prepare Data
            clean_raw = cv2.imread(img_path)
            if clean_raw is None: continue
            
            clean = cv2.resize(clean_raw, (256, 256))
            dirty = apply_random_degradation(clean.copy())
            
            img_tensor = (dirty / 255.0).astype(np.float32)
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # B. Inference with Optional TTA
            start_time = time.time()
            if use_tta:
                enhanced_tensor = apply_tta(model, img_tensor, DEVICE)
            else:
                with torch.no_grad():
                    enhanced_tensor = model(img_tensor)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)

            # C. Post-Process
            enhanced = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

            # D. Metrics
            psnr_clean = psnr(clean, enhanced)
            psnr_dirty = psnr(clean, dirty)
            
            ssim_clean = ssim(clean, enhanced, channel_axis=2, data_range=255)
            ssim_dirty = ssim(clean, dirty, channel_axis=2, data_range=255)
            
            psnr_scores.append(psnr_clean)
            ssim_scores.append(ssim_clean)
            psnr_improvement.append(psnr_clean - psnr_dirty)
            ssim_improvement.append(ssim_clean - ssim_dirty)
            
            sample_vis = (clean, dirty, enhanced)

        except Exception as e:
            pass

    # --- RESULTS ---
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_speed = np.mean(inference_times)

    # Save results with TTA suffix
    suffix = "_tta" if use_tta else ""
    
    with open(SAVE_PTH + f"metrics_summary{suffix}.txt", "w") as f:
        f.write(f"AdaLOLIE Evaluation Report\n")
        f.write(f"==========================\n")
        f.write(f"Test-Time Augmentation: {use_tta}\n")
        f.write(f"Images Tested: {len(psnr_scores)}\n")
        f.write(f"Avg PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
        f.write(f"Avg PSNR Improvement: {np.mean(psnr_improvement):.2f} dB\n")
        f.write(f"Avg SSIM Improvement: {np.mean(ssim_improvement):.4f}\n")
        f.write(f"Avg Time: {avg_speed:.2f} ms\n")

    # Generate graphs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.boxplot(psnr_scores, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.title(f"PSNR (Avg: {avg_psnr:.1f} dB)")
    plt.ylabel("dB")
    
    plt.subplot(1, 3, 2)
    plt.boxplot(ssim_scores, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title(f"SSIM (Avg: {avg_ssim:.3f})")
    plt.ylabel("Index (0-1)")

    plt.subplot(1, 3, 3)
    plt.bar([f"AdaLOLIE{suffix.upper()}"], [avg_speed], color='orange', width=0.4)
    plt.title(f"Speed ({avg_speed:.1f} ms)")
    plt.ylabel("Time (ms)")
    
    plt.tight_layout()
    plt.savefig(SAVE_PTH + f"graph_metrics_distribution{suffix}.png")
    
    if sample_vis:
        clean, dirty, enhanced = sample_vis
        combined = np.hstack((clean, dirty, enhanced))
        cv2.imwrite(SAVE_PTH + f"sample_evaluation_triplet{suffix}.png", combined)

    print(f"\n✅ Evaluation Complete!")
    print(f"   Avg PSNR: {avg_psnr:.2f} dB (↑{np.mean(psnr_improvement):.2f} dB)")
    print(f"   Avg SSIM: {avg_ssim:.4f} (↑{np.mean(ssim_improvement):.4f})")
    print(f"   Avg Speed: {avg_speed:.2f} ms")

if __name__ == "__main__":
    # Run evaluation with TTA enabled
    # evaluate(use_tta=True)
    
    # Compare with TTA disabled
    evaluate(use_tta=False)