import streamlit as st
import torch
import cv2
import numpy as np
import os
import tempfile
import uuid
from datetime import datetime
from ultralytics import YOLO
import pymongo
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

from model_zero_dce_based import AdaLOLIE_Net

# CONFIGURATION
MODEL_PATH = "weights/adalolie_best.pth"
YOLO_MODEL = "weights/best_latest.pt"
# YOLO_MODEL = "yolov8n.pt"

username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')

# MongoDB Configuration
MONGO_URI = f"mongodb+srv://{username}:{password}@fyp-cluster0.uszdvma.mongodb.net/?appName=FYP-Cluster0"
DB_NAME = "adalolie_safety_system"
COLLECTION_NAME = "incident_logs"

class AdaLOLIE_SafetyMonitorApp:
    def __init__(self):
        """Initialize the Controller, load models, and connect to DB."""
        self.enhancer, self.detector, self.device = self.load_models()
        self.db_collection = self.connect_database()

    @staticmethod
    @st.cache_resource
    def load_models():
        """Loads AI models with Streamlit caching to prevent reloading."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enhancer = AdaLOLIE_Net().to(device)
        
        if torch.cuda.is_available():
            checkpoint = torch.load(MODEL_PATH, map_location=device)
        else:
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        if 'model_state_dict' in checkpoint:
            enhancer.load_state_dict(checkpoint['model_state_dict'])
        else:
            enhancer.load_state_dict(checkpoint)
            
        enhancer.eval()
        detector = YOLO(YOLO_MODEL)
        
        return enhancer, detector, device

    @staticmethod
    @st.cache_resource
    def connect_database():
        """Establishes a connection to the local MongoDB instance."""
        try:
            client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            # Force connection check
            client.server_info()
            db = client[DB_NAME]
            return db[COLLECTION_NAME]
        except pymongo.errors.ServerSelectionTimeoutError:
            st.sidebar.warning("⚠️ MongoDB is not running. Audit logging is disabled.")
            return None
        
    def apply_clahe(self, img_rgb):
        """Mandatory pre-processing to match training signal."""
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def process_image(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
        # 1. Pipeline Matching: Pre-lift signal with CLAHE
        lifted_img = self.apply_clahe(img_rgb)
        resized_img = cv2.resize(lifted_img, (640, 640))
        # resized_img = cv2.resize(img_rgb, (640, 640))
        
        # 2. Enhance
        img_tensor = (resized_img / 255.0).astype(np.float32)
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            enhanced_tensor, _ = self.enhancer(img_tensor)
            
        enhanced_small = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        enhanced_small = np.clip(enhanced_small, 0, 1)
        enhanced_small = (enhanced_small * 255).astype(np.uint8)
        
        # 3. Restore to High Resolution
        enhanced_high_res_rgb = cv2.resize(enhanced_small, (w, h))
        
        # 4. Detection (THE BUG FIX)
        # We must pass the BGR arrays to YOLO, and we add conf=0.15 to match your tests.
        
        # A. Raw Detection
        results_org = self.detector(img_bgr, verbose=False, conf=0.15) 
        org_plotted_bgr = results_org[0].plot()
        org_plotted_rgb = cv2.cvtColor(org_plotted_bgr, cv2.COLOR_BGR2RGB) # Convert back for Streamlit
        org_count = len(results_org[0].boxes)
        
        # B. Enhanced Detection
        enhanced_high_res_bgr = cv2.cvtColor(enhanced_high_res_rgb, cv2.COLOR_RGB2BGR)
        results_enh = self.detector(enhanced_high_res_bgr, verbose=False, conf=0.15)
        enh_plotted_bgr = results_enh[0].plot()
        enh_plotted_rgb = cv2.cvtColor(enh_plotted_bgr, cv2.COLOR_BGR2RGB) # Convert back for Streamlit
        enh_count = len(results_enh[0].boxes)
        
        return org_plotted_rgb, enh_plotted_rgb, org_count, enh_count

    def generate_pdf_report(self, incident_id, timestamp, org_count, enh_count, org_img, enh_img):
        """Generates a downloadable PDF Safety Report."""
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="AdaLOLIE: Mining Safety Incident Report", ln=True, align='C')
        pdf.ln(5)
        
        # Metadata
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 8, txt=f"Incident ID: {incident_id}", ln=True)
        pdf.cell(200, 8, txt=f"Timestamp: {timestamp}", ln=True)
        pdf.cell(200, 8, txt="Operator: Thaveesha Sonnadara", ln=True)
        pdf.ln(5)
        
        # Metrics
        safety_status = "Protected" if enh_count > 0 else "High Risk"
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 8, txt=f"Raw Detections (Low-Light): {org_count}", ln=True)
        pdf.cell(200, 8, txt=f"AdaLOLIE Detections (Enhanced): {enh_count}", ln=True)
        pdf.cell(200, 8, txt=f"System Safety Assessment: {safety_status}", ln=True)
        pdf.ln(10)
        
        # Save temporary images to embed in PDF
        with tempfile.TemporaryDirectory() as tmpdirname:
            org_path = os.path.join(tmpdirname, "org.jpg")
            enh_path = os.path.join(tmpdirname, "enh.jpg")
            
            # Convert back to BGR for OpenCV saving
            cv2.imwrite(org_path, cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(enh_path, cv2.cvtColor(enh_img, cv2.COLOR_RGB2BGR))
            
            # Embed Images
            pdf.cell(200, 10, txt="Original Feed vs. Enhanced Feed:", ln=True)
            # x, y, width
            pdf.image(org_path, x=10, y=100, w=90)
            pdf.image(enh_path, x=110, y=100, w=90)
            
        return pdf.output(dest='S').encode('latin-1')

    def run(self):
        """Renders the Streamlit GUI."""
        st.set_page_config(layout="wide", page_title="AdaLOLIE Dashboard") 

        st.title("🔦 AdaLOLIE: Mining Safety Enhancement")
        st.markdown("### Enterprise Safety & Audit Dashboard")

        uploaded_file = st.file_uploader("Upload a Mining Image", type=['jpg', 'png', 'jpeg'])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with st.spinner('Running AI Tensor Pipeline...'):
                org_img, enh_img, org_count, enh_count = self.process_image(uploaded_file)
                
            with col1:
                container1 = st.container(vertical_alignment='center', horizontal_alignment='center')
                with container1:
                    st.image(org_img, caption=f"Original Input | Detections: {org_count}", width=350)
                    if org_count == 0:
                        st.error("⚠️ SAFETY RISK: No objects detected in raw feed!")
                    
            with col2:
                container2 = st.container(vertical_alignment='center', horizontal_alignment='center')
                with container2:
                    st.image(enh_img, caption=f"AdaLOLIE Output | Detections: {enh_count}", width=350)
                    if enh_count > org_count:
                        st.success(f"✅ SAFETY IMPROVED: +{enh_count - org_count} Objects Found")
                    elif enh_count == org_count and enh_count > 0:
                        st.info(f"✅ Detection Confirmed")
                    
            # Metrics Table
            st.table({
                "Metric": ["Visibility", "YOLO Detections", "Safety Status"],
                "Original": ["Low/Dark", f"{org_count}", "❌ High Risk" if org_count == 0 else "⚠️ Caution"],
                "AdaLOLIE": ["Enhanced", f"{enh_count}", "✅ Protected" if enh_count > 0 else "❌ High Risk"]
            })

            st.divider()
            st.markdown("### 📋 Audit & Reporting")
            
            # Generate Session Metadata
            incident_id = f"MIN-{str(uuid.uuid4())[:8].upper()}"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("💾 Log Incident to Database", use_container_width=True):
                    if self.db_collection is not None:
                        log_entry = {
                            "incident_id": incident_id,
                            "timestamp": timestamp,
                            "raw_detections": org_count,
                            "enhanced_detections": enh_count,
                            "safety_gain": enh_count - org_count,
                            "status": "Protected" if enh_count > 0 else "High Risk"
                        }
                        self.db_collection.insert_one(log_entry)
                        st.success(f"Incident {incident_id} securely logged to MongoDB.")
                    else:
                        st.error("Cannot log to database. MongoDB connection failed.")
            
            with action_col2:
                pdf_bytes = self.generate_pdf_report(incident_id, timestamp, org_count, enh_count, org_img, enh_img)
                st.download_button(
                    label="📄 Download Safety Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"Safety_Report_{incident_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# ENTRY POINT
if __name__ == "__main__":
    app = AdaLOLIE_SafetyMonitorApp()
    app.run()