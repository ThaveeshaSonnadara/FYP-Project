import comet_ml
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import cv2
import glob
import os
import numpy as np
from tqdm import tqdm

# Import Local Modules
from model_zero_dce_based import AdaLOLIE_Net
from loss_functions import AdaLOLIELoss
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
        self.NUM_EPOCHS = 100
        self.SAVE_DIR = "/content/drive/MyDrive/FYP_PROJECT/FYP_Checkpoints/AdaLOLIE"
        
        self.TRAIN_PATH = "/content/MiningMix_Unified/train"
        self.VAL_PATH = "/content/MiningMix_Unified/val"
        self.TEST_PATH = "/content/MiningMix_Unified/test"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ACCUMULATION_STEPS = 1
        
        # Early Stopping Configuration
        self.PATIENCE = 15
        self.MIN_DELTA = 0.001  

    def run_test_evaluation(self, model):
        """Logs PSNR and SSIM accuracy to Comet during the test phase"""
        print("\n🧪 Running Test Evaluation...")
        model.eval()
        
        test_files = glob.glob(os.path.join(self.TEST_PATH, "**", "*.jpg"), recursive=True)[:50]
        psnr_scores, ssim_scores = [], []

        with torch.no_grad():
            for img_path in tqdm(test_files, desc="Testing"):
                    clean_raw = cv2.imread(img_path)
                    if clean_raw is None: continue
                    clean = cv2.resize(clean_raw, (256, 256))
                    
                    img_tensor = (clean / 255.0).astype(np.float32)
                    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)

                    # UPDATE: Ignore x_r during evaluation
                    enhanced_tensor, _ = model(img_tensor)
                    
                    enhanced = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
                    clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

                    psnr_scores.append(psnr(clean_rgb, enhanced))
                    ssim_scores.append(ssim(clean_rgb, enhanced, channel_axis=2, data_range=255))

            avg_psnr = np.mean(psnr_scores)
            avg_ssim = np.mean(ssim_scores)
            self.exp_obj.log_metric("Avarage PSNR", avg_psnr)
            self.exp_obj.log_metric("Avarage SSIM", avg_ssim)
            print(f"✅ Test Metrics Logged -> PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
    
    def train_for_hpo(self, weights, epochs=5):
        """Modified training loop for Optuna HPO trials."""
        w_exp, w_col, w_spa, w_tv, w_glare = weights
        model = AdaLOLIE_Net().to(self.device)
        loss_fn = AdaLOLIELoss().to(self.device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        
        train_loader = DataLoader(MiningDataset(self.TRAIN_PATH), batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(MiningDataset(self.VAL_PATH), batch_size=self.BATCH_SIZE, shuffle=False)

        best_trial_ssim = 0.0

        for epoch in range(epochs):
            model.train()
            for imgs in train_loader:
                imgs = imgs.to(self.device)
                
                # UPDATE: Unpack tuple, ignore x_r for HPO (unless you add it to Optuna later)
                enhanced, _ = model(imgs)
                
                L_exp = loss_fn.get_exposure_loss(enhanced)
                L_col = loss_fn.get_color_loss(enhanced)
                L_spa = loss_fn.get_spatial_loss(enhanced, imgs)
                L_tv = loss_fn.get_grayscale_loss(enhanced)
                L_glare = loss_fn.get_glare_loss(enhanced)
                
                loss = w_exp*L_exp + w_col*L_col + w_spa*L_spa + w_tv*L_tv + w_glare*L_glare
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            ssim_scores = []
            with torch.no_grad():
                for imgs in val_loader:
                    imgs = imgs.to(self.device)
                    
                    # UPDATE: Ignore x_r here too
                    enhanced, _ = model(imgs)
                    
                    enh_np = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
                    org_np = imgs.squeeze().permute(1, 2, 0).cpu().numpy()
                    ssim_scores.append(ssim(org_np, enh_np, channel_axis=2, data_range=1.0))
            
            current_ssim = np.mean(ssim_scores)
            if current_ssim > best_trial_ssim:
                best_trial_ssim = current_ssim
        
        return best_trial_ssim

    def train(self):
        model = AdaLOLIE_Net().to(self.device)
        loss_fn = AdaLOLIELoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.NUM_EPOCHS, 
            eta_min=1e-6
        )
        
        scaler = GradScaler() if torch.cuda.is_available() else None
        
        hyper_params = {
            "batch_size": self.BATCH_SIZE, 
            "learning_rate": self.LEARNING_RATE, 
            "epochs": self.NUM_EPOCHS,
            "mixed_precision": scaler is not None,
            "early_stopping_patience": self.PATIENCE
        }
        self.exp_obj.log_parameters(hyper_params)

        if not os.path.exists(self.SAVE_DIR): 
            os.makedirs(self.SAVE_DIR)
        
        start_epoch = 1
        best_val_loss = float('inf')
        patience_counter = 0
        
        checkpoint_path = os.path.join(self.SAVE_DIR, "adalolie_last.pth")
        if os.path.exists(checkpoint_path):
            print(f"🔄 Found checkpoint at {checkpoint_path}. Resuming training...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if scaler is not None and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"⏩ Fast-forwarding to Epoch {start_epoch} (Previous Best Loss: {best_val_loss:.5f})")
        else:
            print("🆕 Starting training from scratch...")
        
        train_loader = DataLoader(MiningDataset(self.TRAIN_PATH), batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(MiningDataset(self.VAL_PATH), batch_size=self.BATCH_SIZE, shuffle=False)

        for epoch in range(start_epoch, self.NUM_EPOCHS+1):
            # --- TRAINING PHASE ---
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch}/{self.NUM_EPOCHS}")
            
            optimizer.zero_grad()
            
            for batch_idx, imgs in enumerate(loop):
                imgs = imgs.to(self.device)
                
                if scaler is not None:
                    with autocast(device_type="cuda"): 
                        # UPDATE: Capture x_r and pass to new loss function
                        enhanced, x_r = model(imgs)
                        loss = loss_fn(enhanced, imgs, x_r)
                        loss = loss / self.ACCUMULATION_STEPS
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.ACCUMULATION_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # UPDATE: Capture x_r and pass to new loss function
                    enhanced, x_r = model(imgs)
                    loss = loss_fn(enhanced, imgs, x_r)
                    loss = loss / self.ACCUMULATION_STEPS
                    loss.backward()
                    
                    if (batch_idx + 1) % self.ACCUMULATION_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                train_loss += loss.item() * self.ACCUMULATION_STEPS
                loop.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)
            self.exp_obj.log_metric("Train Loss", avg_train_loss, step=epoch)
            self.exp_obj.log_current_epoch(epoch)

            # --- VALIDATION PHASE ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs in val_loader:
                    imgs = imgs.to(self.device)
                    
                    # UPDATE: Capture x_r in validation
                    enhanced, x_r = model(imgs)
                    
                    if scaler is not None:
                        with autocast(device_type="cuda"):
                            val_loss += loss_fn(enhanced, imgs, x_r).item()
                    else:
                        val_loss += loss_fn(enhanced, imgs, x_r).item()
            
            avg_val_loss = val_loss / len(val_loader)
            self.exp_obj.log_metric("Validation Loss", avg_val_loss, step=epoch)
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            self.exp_obj.log_metric("Learning Rate", current_lr, step=epoch)

            if avg_val_loss < (best_val_loss - self.MIN_DELTA):
                best_val_loss = avg_val_loss
                patience_counter = 0  
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'scaler_state_dict': scaler.state_dict() if scaler else None
                }, os.path.join(self.SAVE_DIR, "adalolie_best.pth"))
                
                print(f"✅ New Best Model Saved! Val Loss: {avg_val_loss:.5f}")
            else:
                patience_counter += 1
                print(f"⚠️ No improvement for {patience_counter}/{self.PATIENCE} epochs")
                
                if patience_counter >= self.PATIENCE:
                    print(f"🛑 Early Stopping Triggered at Epoch {epoch}")
                    self.exp_obj.log_metric("Early Stopped", 1, step=epoch)
                    break
        
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'scaler_state_dict': scaler.state_dict() if scaler else None
                }, os.path.join(self.SAVE_DIR, "adalolie_last.pth"))
        
        self.run_test_evaluation(model)
        print(f"\n✅ Training Complete! Best Val Loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    import comet_ml
    experiment = comet_ml.start(project_name="AdaLOLIE-Mining-Safety")
    
    trainer = TrainScript(experiment)
    trainer.train()

    experiment.end()