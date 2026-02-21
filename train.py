# 1. Comet Integration Imports
import comet_ml
from comet_ml.integration.pytorch import watch, log_model

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import os
import numpy as np
from tqdm import tqdm

# Import Local Modules
from model import AdaLOLIE_Net
from loss_functions import AdaLOLIELoss
# Import metrics for the testing part
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- DATASET CLASS ---
class MiningDataset(Dataset):
    def __init__(self, folder_path, limit=None):
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.*")))
        self.image_paths = [x for x in self.image_paths if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if limit is not None and len(self.image_paths) > limit:
            import random
            random.seed(42)
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:limit]

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.image_paths[idx])
            if img is None: return torch.zeros(3, 256, 256) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = (np.asarray(img)/255.0).astype(np.float32)
            return torch.from_numpy(img).permute(2, 0, 1)
        except:
            return torch.zeros(3, 256, 256)

# --- TRAIN SCRIPT CLASS ---
class TrainScript:
    def __init__(self, exp_obj):
        self.exp_obj = exp_obj
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 1e-4
        self.NUM_EPOCHS = 30
        self.SAVE_DIR = "checkpoints"
        self.LOG_FILE = "training_log.csv"
        self.RESUME_CHECKPOINT = os.path.join(self.SAVE_DIR, "latest_checkpoint.pth")
        
        self.TRAIN_PATH = "Data/MiningMix_Unified/train"
        self.VAL_PATH = "Data/MiningMix_Unified/val"
        self.TEST_PATH = "Data/MiningMix_Unified/test"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_test_evaluation(self, model):
        """Logs PSNR and SSIM accuracy to Comet during the test phase"""
        print("\n🧪 Running Test Evaluation...")
        model.eval()
        
        # Using a subset for quick logging during training, or full set at the end
        test_files = glob.glob(os.path.join(self.TEST_PATH, "**", "*.jpg"), recursive=True)[:50]
        psnr_scores, ssim_scores = [], []

        # Wrap in Comet test context
        # with self.exp_obj.test():
        with torch.no_grad():
            for img_path in tqdm(test_files, desc="Testing"):
                    clean_raw = cv2.imread(img_path)
                    if clean_raw is None: continue
                    clean = cv2.resize(clean_raw, (256, 256))
                    
                    # Normalizing for model
                    img_tensor = (clean / 255.0).astype(np.float32)
                    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)

                    enhanced_tensor = model(img_tensor)
                    
                    # Post-process for metrics
                    enhanced = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
                    clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

                    psnr_scores.append(psnr(clean_rgb, enhanced))
                    ssim_scores.append(ssim(clean_rgb, enhanced, channel_axis=2, data_range=255))

            # Log average metrics to Comet
            avg_psnr = np.mean(psnr_scores)
            avg_ssim = np.mean(ssim_scores)
            self.exp_obj.log_metric("Avarage PSNR", avg_psnr)
            self.exp_obj.log_metric("Avarage SSIM", avg_ssim)
            print(f"✅ Test Metrics Logged -> PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

    def train(self):
        model = AdaLOLIE_Net().to(self.device)
        loss_fn = AdaLOLIELoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        
        # Log Hyperparameters to Comet
        hyper_params = {"batch_size": self.BATCH_SIZE, "learning_rate": self.LEARNING_RATE, "epochs": self.NUM_EPOCHS}
        self.exp_obj.log_parameters(hyper_params)

        # Watch model for weights/gradients histograms
        watch(model) 

        if not os.path.exists(self.SAVE_DIR): os.makedirs(self.SAVE_DIR)
        
        start_epoch = 0
        best_val_loss = float('inf')
        
        # Load data
        train_loader = DataLoader(MiningDataset(self.TRAIN_PATH, limit=250), batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(MiningDataset(self.VAL_PATH, limit=50), batch_size=self.BATCH_SIZE, shuffle=False)

        # Comet Training Context
        # with self.exp_obj.train(): 
        for epoch in range(start_epoch, self.NUM_EPOCHS):
                model.train()
                train_loss = 0.0
                loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.NUM_EPOCHS}")
                
                for imgs in loop:
                    imgs = imgs.to(self.device)
                    enhanced = model(imgs)
                    loss = loss_fn(enhanced, imgs)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

                avg_train_loss = train_loss / len(train_loader)
                self.exp_obj.log_metric("Train Loss", avg_train_loss, step=epoch)
                self.exp_obj.log_current_epoch(epoch)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for imgs in val_loader:
                        imgs = imgs.to(self.device)
                        val_loss += loss_fn(model(imgs), imgs).item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.exp_obj.log_metric("Validation Loss", avg_val_loss, step=epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), os.path.join(self.SAVE_DIR, "adalolie_best.pth"))
            
        
        # Run final test evaluation and log to Comet
        self.run_test_evaluation(model)
        
        # Save and Log the final model to Comet Registry
        log_model(self.exp_obj, model, "AdaLOLIE-Net-Mining")

if __name__ == "__main__":
    # Initialize Comet
    experiment = comet_ml.start(project_name="AdaLOLIE-Mining-Safety")
    
    trainer = TrainScript(experiment)
    trainer.train()

    experiment.end()
