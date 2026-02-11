import cv2
import numpy as np
import random
import os

def add_coal_black_shift(image, intensity=0.4):
    """Simulates coal dust absorption (Black Shift)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Non-linear darkening
    v = v.astype(np.float32)
    v = v * (1.0 - intensity)
    v = np.clip(v, 0, 255).astype(np.uint8)

    # Desaturation
    s = s.astype(np.float32) * 0.8
    s = np.clip(s, 0, 255).astype(np.uint8)

    final_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def add_dust_and_mist(image, density=0.3):
    """Simulates atmospheric scattering"""
    row, col, ch = image.shape
    noise = np.random.normal(0, 1, (row, col))

    # Create cloud-like mist structure
    blur_size = 15
    dust_mask = cv2.GaussianBlur(noise, (blur_size, blur_size), 0)
    cv2.normalize(dust_mask, dust_mask, 0, 255, cv2.NORM_MINMAX)
    dust_mask = dust_mask.astype(np.uint8)
    dust_layer = cv2.cvtColor(dust_mask, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(image, 1 - density, dust_layer, density, 0)


def process_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(input_dir)
    print(f"Processing {len(images)} images from {input_dir}...")

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # Pipeline: Black Shift -> Dust -> Glare
        step1 = add_coal_black_shift(img, intensity=0.3)
        final = add_dust_and_mist(step1, density=0.25)
        # final = add_uneven_illumination(step2)

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, final)
    print("Done generating synthetic mining data!")

if __name__ == "__main__":
    process_dataset("Data/ImageMine_Original", "Data/ImageMine_Synthetic")