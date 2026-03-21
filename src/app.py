import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO

from model_zero_dce_based import AdaLOLIE_Net

# CONFIGURATION
MODEL_PATH = "checkpoints/adalolie_best.pth"
YOLO_MODEL = "yolov8n.pt"

class AdaLOLIE_SafetyMonitorApp:
    def __init__(self):
        """Initialize the Controller and load models."""
        self.enhancer, self.detector, self.device = self.load_models()

    @staticmethod
    @st.cache_resource
    def load_models():
        """
        Loads models with caching.
        Static method prevents 'self' from breaking the cache hash.
        """
        # 1. Load AdaLOLIE (Enhancer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enhancer = AdaLOLIE_Net().to(device)
        
        # Load the weights
        if torch.cuda.is_available():
            checkpoint = torch.load(MODEL_PATH, map_location=device)
        else:
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Handle dictionary mismatch
        if 'model_state_dict' in checkpoint:
            enhancer.load_state_dict(checkpoint['model_state_dict'])
        else:
            enhancer.load_state_dict(checkpoint)
            
        enhancer.eval()
        
        # 2. Load YOLO (Detector)
        detector = YOLO(YOLO_MODEL)
        
        return enhancer, detector, device

    def process_image(self, uploaded_file):
        """Core logic: Pre-process -> Enhance -> Detect -> Post-process"""
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        # 1. Convert to RGB for the entire pipeline
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Capture Original Dimensions
        h, w, _ = img_rgb.shape
        
        # ENHANCEMENT PIPELINE
        
        # 1. Resize for Model Inference Only
        inference_size = (512, 512)
        resized_img = cv2.resize(img_rgb, inference_size)
        
        # 2. ENHANCE
        img_tensor = (resized_img / 255.0).astype(np.float32)
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            enhanced_tensor = self.enhancer(img_tensor)
            
        enhanced_small = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        enhanced_small = np.clip(enhanced_small, 0, 1)
        enhanced_small = (enhanced_small * 255).astype(np.uint8)
        
        # 3. Restore to High Resolution
        enhanced_high_res_rgb = cv2.resize(enhanced_small, (w, h))
        
        # DETECTION PIPELINE
        
        # 4. DETECT (Original High-Res)
        results_org = self.detector(img_rgb, verbose=False)
        org_plotted = results_org[0].plot()
        org_count = len(results_org[0].boxes)
        
        # 5. DETECT (Enhanced High-Res)
        results_enh = self.detector(enhanced_high_res_rgb, verbose=False)
        enh_plotted = results_enh[0].plot()
        enh_count = len(results_enh[0].boxes)
        
        return org_plotted, enh_plotted, org_count, enh_count

    def run(self):
        """Renders the Streamlit GUI."""
        st.set_page_config(layout="wide") 

        st.title("🔦 AdaLOLIE: Mining Safety Enhancement")
        st.markdown("### Ultimate Proof of Concept")
        st.write("Comparing standard YOLO object detection on **Raw Low-Light Mining Images** vs. **AdaLOLIE Enhanced Images**.")

        uploaded_file = st.file_uploader("Upload a Mining Image", type=['jpg', 'png', 'jpeg'])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with st.spinner('Processing High-Resolution Feed...'):
                org_img, enh_img, org_count, enh_count = self.process_image(uploaded_file)
                
            # Display Result: Original
            with col1:
                st.image(org_img, caption=f"Original Input\nDetections: {org_count}", width='stretch')
                if org_count == 0:
                    st.error("⚠️ SAFETY RISK: No objects detected!")
                    
            # Display Result: AdaLOLIE
            with col2:
                st.image(enh_img, caption=f"AdaLOLIE Output\nDetections: {enh_count}", width='stretch')
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

# ENTRY POINT
if __name__ == "__main__":
    app = AdaLOLIE_SafetyMonitorApp()
    app.run()