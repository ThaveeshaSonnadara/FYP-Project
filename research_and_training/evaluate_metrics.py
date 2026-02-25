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

# --- NEW: Modern PyTorch Image Quality Library ---
try:
    from piq import brisque, niqe
    HAS_PIQ = True
except ImportError:
    print("⚠️ 'piq' library not found. Run: pip install piq")
    HAS_PIQ = False

from src.model import AdaLOLIE_Net

# --- CONFIG ---
MODEL_PATH = "../checkpoints/adalolie_best.pth"
TEST_DIR = "../Data/MiningMix_Unified/test" # Ground Truth source
NUM_TEST_IMAGES = 100
SAVE_PTH = "../Output/Evaluation Metrics/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_PTH):
    os.makedirs(SAVE_PTH)

# --- REALISTIC DEGRADATION FUNCTIONS (Matches Training!) ---
def add_coal_black_shift(image, intensity):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = (v.astype(np.float32) * (1.0 - intensity)).clip(0, 255).astype(np.uint8)
    s = (s.astype(np.float32) * 0.8).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def add_dust_and_mist(image, density):
    noise = np.random.normal(0, 1, image.shape[:2])
    dust = cv2.GaussianBlur(noise, (31, 31), 0)
    cv2.normalize(dust, dust, 0, 255, cv2.NORM_MINMAX)
    dust_layer = cv2.cvtColor(dust.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1 - density, dust_layer, density, 0)

def apply_random_degradation(image):
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

# --- TEST-TIME AUGMENTATION HELPER ---
def apply_tta(model, img_tensor, device):
    predictions = []
    with torch.no_grad():
        pred = model(img_tensor)
        predictions.append(pred)
        
        flipped_h = torch.flip(img_tensor, dims=[3])
        pred_h = torch.flip(model(flipped_h), dims=[3])
        predictions.append(pred_h)
        
        flipped_v = torch.flip(img_tensor, dims=[2])
        pred_v = torch.flip(model(flipped_v), dims=[2])
        predictions.append(pred_v)
        
        rotated = torch.rot90(img_tensor, k=1, dims=[2, 3])
        pred_rot = torch.rot90(model(rotated), k=-1, dims=[2, 3])
        predictions.append(pred_rot)
        
    return torch.mean(torch.stack(predictions), dim=0)

def evaluate(use_tta=True):
    print(f"🚀 Starting Evaluation on {DEVICE}...")
    print(f"   Test-Time Augmentation: {'✅ Enabled' if use_tta else '❌ Disabled'}")
    
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

    all_files = glob.glob(os.path.join(TEST_DIR, "**", "*.jpg"), recursive=True)
    if len(all_files) > NUM_TEST_IMAGES:
        all_files = np.random.choice(all_files, NUM_TEST_IMAGES, replace=False)

    psnr_scores, ssim_scores, inference_times = [], [], []
    psnr_imp, ssim_imp = [], []
    niqe_raw, niqe_enh = [], []
    brisque_raw, brisque_enh = [], []
    sample_vis = None

    print(f"📊 Testing {len(all_files)} images with Random Degradation & NR Metrics...")

    for img_path in tqdm(all_files):
        try:
            clean_raw = cv2.imread(img_path)
            if clean_raw is None: continue
            
            clean = cv2.resize(clean_raw, (256, 256))
            dirty = apply_random_degradation(clean.copy())
            
            # Convert to RGB and prepare Tensor
            dirty_rgb = cv2.cvtColor(dirty, cv2.COLOR_BGR2RGB)
            img_tensor = (dirty_rgb / 255.0).astype(np.float32)
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # Inference
            start_time = time.time()
            if use_tta:
                enhanced_tensor = apply_tta(model, img_tensor, DEVICE)
            else:
                with torch.no_grad():
                    enhanced_tensor = model(img_tensor)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)

            # Post-Process for Reference Metrics
            enhanced_np = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_bgr = cv2.cvtColor((np.clip(enhanced_np * 255, 0, 255)).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # --- REFERENCE METRICS (PSNR/SSIM) ---
            psnr_c = psnr(clean, enhanced_bgr)
            ssim_c = ssim(clean, enhanced_bgr, channel_axis=2, data_range=255)
            psnr_scores.append(psnr_c)
            ssim_scores.append(ssim_c)
            psnr_imp.append(psnr_c - psnr(clean, dirty))
            ssim_imp.append(ssim_c - ssim(clean, dirty, channel_axis=2, data_range=255))
            
            # --- NO-REFERENCE METRICS using PIQ ---
            if HAS_PIQ:
                with torch.no_grad():
                    # PIQ expects tensors in [0, 1] range
                    enhanced_clamped = enhanced_tensor.clamp(0, 1)
                    
                    n_raw = niqe(img_tensor, data_range=1.0).item()
                    n_enh = niqe(enhanced_clamped, data_range=1.0).item()
                    b_raw = brisque(img_tensor, data_range=1.0).item()
                    b_enh = brisque(enhanced_clamped, data_range=1.0).item()
                    
                    niqe_raw.append(n_raw)
                    niqe_enh.append(n_enh)
                    brisque_raw.append(b_raw)
                    brisque_enh.append(b_enh)
            
            sample_vis = (clean, dirty, enhanced_bgr)

        except Exception as e:
            continue

    # --- CALCULATE AVERAGES ---
    avg_psnr, avg_ssim, avg_speed = np.mean(psnr_scores), np.mean(ssim_scores), np.mean(inference_times)
    
    avg_n_raw, avg_n_enh = np.mean(niqe_raw) if niqe_raw else 0, np.mean(niqe_enh) if niqe_enh else 0
    avg_b_raw, avg_b_enh = np.mean(brisque_raw) if brisque_raw else 0, np.mean(brisque_enh) if brisque_enh else 0

    suffix = "_tta" if use_tta else ""
    
    # --- SAVE TEXT REPORT ---
    with open(SAVE_PTH + f"metrics_summary{suffix}.txt", "w") as f:
        f.write(f"AdaLOLIE Evaluation Report\n==========================\n")
        f.write(f"Test-Time Augmentation: {use_tta}\nImages Tested: {len(psnr_scores)}\n\n")
        
        f.write(f"--- Reference-Based (Higher is better) ---\n")
        f.write(f"Avg PSNR: {avg_psnr:.2f} dB (Imp: {np.mean(psnr_imp):.2f} dB)\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f} (Imp: {np.mean(ssim_imp):.4f})\n\n")
        
        f.write(f"--- No-Reference (Lower is better) ---\n")
        f.write(f"Avg NIQE: Raw={avg_n_raw:.2f} -> Enhanced={avg_n_enh:.2f} (Imp: {avg_n_raw - avg_n_enh:.2f})\n")
        f.write(f"Avg BRISQUE: Raw={avg_b_raw:.2f} -> Enhanced={avg_b_enh:.2f} (Imp: {avg_b_raw - avg_b_enh:.2f})\n\n")
        
        f.write(f"Avg Time: {avg_speed:.2f} ms\n")

    # --- GENERATE COMPREHENSIVE GRAPHS ---
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 3, 1)
    plt.boxplot(psnr_scores, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.title(f"PSNR (Avg: {avg_psnr:.1f} dB)")
    
    plt.subplot(2, 3, 2)
    plt.boxplot(ssim_scores, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title(f"SSIM (Avg: {avg_ssim:.3f})")

    plt.subplot(2, 3, 3)
    plt.bar([f"AdaLOLIE{suffix.upper()}"], [avg_speed], color='orange')
    plt.title(f"Speed ({avg_speed:.1f} ms)")
    
    if niqe_raw and niqe_enh:
        plt.subplot(2, 3, 4)
        plt.boxplot([niqe_raw, niqe_enh], labels=['Raw', 'Enhanced'], patch_artist=True, boxprops=dict(facecolor="salmon"))
        plt.title(f"NIQE (Lower=Better)\nRaw: {avg_n_raw:.1f} | Enh: {avg_n_enh:.1f}")

        plt.subplot(2, 3, 5)
        plt.boxplot([brisque_raw, brisque_enh], labels=['Raw', 'Enhanced'], patch_artist=True, boxprops=dict(facecolor="plum"))
        plt.title(f"BRISQUE (Lower=Better)\nRaw: {avg_b_raw:.1f} | Enh: {avg_b_enh:.1f}")
    
    plt.tight_layout()
    plt.savefig(SAVE_PTH + f"graph_metrics_distribution{suffix}.png")
    
    if sample_vis:
        clean, dirty, enhanced = sample_vis
        combined = np.hstack((clean, dirty, enhanced))
        cv2.imwrite(SAVE_PTH + f"sample_evaluation_triplet{suffix}.png", combined)

    print(f"\n✅ Evaluation Complete!")
    print(f"   PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    print(f"   NIQE: {avg_n_enh:.2f} (Improved by {avg_n_raw - avg_n_enh:.2f})")
    print(f"   BRISQUE: {avg_b_enh:.2f} (Improved by {avg_b_raw - avg_b_enh:.2f})")

if __name__ == "__main__":
    evaluate(use_tta=False)