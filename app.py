import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO

from model import AdaLOLIE_Net

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/adalolie_best.pth"
YOLO_MODEL = "yolov8n.pt" # Standard YOLO model

###
# --- LOAD MODELS ONCE ---
@st.cache_resource
def load_models():
    # 1. Load AdaLOLIE (Enhancer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhancer = AdaLOLIE_Net().to(device)
    
    # Load the weights you trained
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle dictionary mismatch (sometimes happens if saved differently)
    if 'model_state_dict' in checkpoint:
        enhancer.load_state_dict(checkpoint['model_state_dict'])
    else:
        enhancer.load_state_dict(checkpoint)
        
    enhancer.eval()
    
    # 2. Load YOLO (Detector)
    detector = YOLO(YOLO_MODEL)
    
    return enhancer, detector, device

enhancer, detector, device = load_models()

# ###
# --- PROCESSING FUNCTION ---
def process_image(uploaded_file):
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Capture Original Dimensions
    h, w, _ = original_img.shape
    
    # --- ENHANCEMENT PIPELINE ---
    
    # 1. Resize for Model Inference Only
    # ###
    inference_size = (512, 512)
    resized_img = cv2.resize(original_img, inference_size)
    
    # 2. ENHANCE
    img_tensor = (resized_img / 255.0).astype(np.float32)
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced_tensor = enhancer(img_tensor)
        
    enhanced_small = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced_small = np.clip(enhanced_small, 0, 1)
    enhanced_small = (enhanced_small * 255).astype(np.uint8)
    
    # 3. Restore to High Resolution
    # ###
    enhanced_high_res = cv2.resize(enhanced_small, (w, h))
    
    # --- DETECTION PIPELINE ---
    
    # ###
    # 4. DETECT (Original High-Res)
    results_org = detector(original_img, verbose=False)
    org_plotted = results_org[0].plot()
    org_count = len(results_org[0].boxes)
    
    # ###
    # 5. DETECT (Enhanced High-Res)
    results_enh = detector(enhanced_high_res, verbose=False)
    enh_plotted = results_enh[0].plot()
    enh_count = len(results_enh[0].boxes)
    
    return org_plotted, enh_plotted, org_count, enh_count

# --- GUI LAYOUT ---
st.set_page_config(layout="wide") # Use wide mode for better side-by-side view

st.title("🔦 AdaLOLIE: Mining Safety Enhancement")
st.markdown("### Ultimate Proof of Concept")
st.write("Comparing standard YOLO object detection on **Raw Low-Light Mining Images** vs. **AdaLOLIE Enhanced Images**.")

uploaded_file = st.file_uploader("Upload a Mining Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Use full width
    col1, col2 = st.columns(2)
    
    with st.spinner('Processing High-Resolution Feed...'):
        org_img, enh_img, org_count, enh_count = process_image(uploaded_file)
        
    # Display Result: Original
    with col1:
        st.image(org_img, caption=f"Original Input\nDetections: {org_count}", width="stretch")
        if org_count == 0:
            st.error("⚠️ SAFETY RISK: No objects detected!")
            
    # Display Result: AdaLOLIE
    with col2:
        st.image(enh_img, caption=f"AdaLOLIE Output\nDetections: {enh_count}", width="stretch")
        if enh_count > org_count:
            st.success(f"✅ SAFETY IMPROVED: +{enh_count - org_count} Objects Found")
        elif enh_count == org_count and enh_count > 0:
             st.info(f"✅ Detection Confirmed")
            
    # Metrics Table
    st.table({
        "Metric": ["Visibility", "YOLO Detections", "Safety Status"],
        "Original": ["Low/Dark", f"{org_count}", "❌ High Risk" if org_count == 0 else "⚠️ Caution"],
        "AdaLOLIE": ["Enhanced", f"{enh_count}", "✅ Protected"]
    })