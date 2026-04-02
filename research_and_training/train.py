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
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import Local Modules
from model_zero_dce_based import AdaLOLIE_Net
from loss_functions import AdaLOLIELoss
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class MiningDataset(Dataset):
    def __init__(self, folder_path, is_train=True):
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.*")))
        self.image_paths = [x for x in self.image_paths if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.is_train = is_train
        
        # Applying augmentation to the dataset while training
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomFog(p=0.4 if is_train else 0.0),
            A.OneOf([
                # GaussNoise parameters
                A.GaussNoise(std_range=(0.1, 0.3), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.4 if is_train else 0.0),
            A.MotionBlur(blur_limit=7, p=0.2 if is_train else 0.0),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2()
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None: return torch.zeros(3, 256, 256)
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge((l, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        augmented = self.transform(image=img)
        return augmented['image']

class TrainScript:
    def __init__(self, exp_obj):
        self.exp_obj = exp_obj
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 1e-4
        self.NUM_EPOCHS = 100
        self.SAVE_DIR = "/content/drive/MyDrive/FYP_PROJECT/FYP_Checkpoints/AdaLOLIE_9"
        self.TRAIN_PATH = "/content/MiningMix_Unified/train"
        self.VAL_PATH = "/content/MiningMix_Unified/val"
        self.TEST_PATH = "/content/MiningMix_Unified/test"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ACCUMULATION_STEPS = 1
        self.PATIENCE = 15
        self.MIN_DELTA = 0.001  

    def run_test_evaluation(self, model):
        print("\n🧪 Running Test Evaluation...")
        model.eval()
        test_files = glob.glob(os.path.join(self.TEST_PATH, "**", "*.jpg"), recursive=True)
        psnr_scores, ssim_scores = [], []
        with torch.no_grad():
            for img_path in tqdm(test_files, desc="Testing"):
                clean_raw = cv2.imread(img_path)
                if clean_raw is None: continue
                clean = cv2.resize(clean_raw, (256, 256))
                
                lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                clean_clahe = cv2.merge((l, a, b))
                clean_rgb = cv2.cvtColor(clean_clahe, cv2.COLOR_LAB2RGB)

                img_tensor = (clean_rgb / 255.0).astype(np.float32)
                img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                enhanced_tensor, _ = model(img_tensor)
                enhanced = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
                
                psnr_scores.append(psnr(clean_rgb, enhanced))
                ssim_scores.append(ssim(clean_rgb, enhanced, channel_axis=2, data_range=255))
                
        avg_psnr, avg_ssim = np.mean(psnr_scores), np.mean(ssim_scores)
        self.exp_obj.log_metric("Average PSNR", avg_psnr)
        self.exp_obj.log_metric("Average SSIM", avg_ssim)
        print(f"✅ Test Metrics Logged -> PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

    def train(self):
        model = AdaLOLIE_Net().to(self.device)
        loss_fn = AdaLOLIELoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.NUM_EPOCHS, eta_min=1e-6)
        scaler = GradScaler() if torch.cuda.is_available() else None
        
        if not os.path.exists(self.SAVE_DIR): os.makedirs(self.SAVE_DIR)
        
        train_loader = DataLoader(MiningDataset(self.TRAIN_PATH, is_train=True), batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(MiningDataset(self.VAL_PATH, is_train=False), batch_size=self.BATCH_SIZE, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        start_epoch = 1

        # AUTO-RESUME LOGIC
        last_checkpoint_path = os.path.join(self.SAVE_DIR, "adalolie_last.pth")
        if os.path.exists(last_checkpoint_path):
            print(f"🔄 Interruption detected! Resuming from {last_checkpoint_path}...")
            checkpoint = torch.load(last_checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            if 'patience_counter' in checkpoint:
                patience_counter = checkpoint['patience_counter']
                
            print(f"▶️ Resuming at Epoch {start_epoch} | Current Best Val Loss: {best_val_loss:.4f}")
        
        for epoch in range(start_epoch, self.NUM_EPOCHS+1):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch}/{self.NUM_EPOCHS}")
            for imgs in loop:
                imgs = imgs.to(self.device)
                optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    enhanced, x_r = model(imgs)
                    loss = loss_fn(enhanced, imgs, x_r)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs in val_loader:
                    imgs = imgs.to(self.device)
                    enhanced, x_r = model(imgs)
                    val_loss += loss_fn(enhanced, imgs, x_r).item()
            
            avg_val_loss = val_loss / len(val_loader)
            self.exp_obj.log_metric("Validation Loss", avg_val_loss, step=epoch)
            scheduler.step()

            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }
            # Always save the latest epoch to 'adalolie_last.pth'
            torch.save(checkpoint_state, last_checkpoint_path)

            if avg_val_loss < (best_val_loss - self.MIN_DELTA):
                best_val_loss = avg_val_loss
                patience_counter = 0
                checkpoint_state['best_val_loss'] = best_val_loss
                # Save the absolute best weights to 'adalolie_best.pth'
                torch.save(checkpoint_state, os.path.join(self.SAVE_DIR, "adalolie_best.pth"))
            else:
                patience_counter += 1
                if patience_counter >= self.PATIENCE: 
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    break
        
        self.run_test_evaluation(model)

if __name__ == "__main__":
    import comet_ml
    experiment = comet_ml.start(project_name="AdaLOLIE-Mining-Safety_9")
    trainer = TrainScript(experiment)
    trainer.train()
    experiment.end()