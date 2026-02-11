import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Import Local Modules
from model import AdaLOLIE_Net
from loss_functions import AdaLOLIELoss

# CONFIGURATION
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
SAVE_DIR = "checkpoints"
LOG_FILE = "training_log.csv"
RESUME_CHECKPOINT = os.path.join(SAVE_DIR, "latest_checkpoint.pth")

# POINTS TO DATA
TRAIN_PATH = "Data/MiningMix_Unified/train"
VAL_PATH = "Data/MiningMix_Unified/val"

class MiningDataset(Dataset):
    def __init__(self, folder_path, limit=None):
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.*")))
        self.image_paths = [x for x in self.image_paths if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # If we asked for a limit (e.g., 250), shuffle and pick only that many
        if limit is not None and len(self.image_paths) > limit:
            import random
            random.seed(42) # Keep it consistent every time you run
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:limit]
            print(f"⚠️ DEBUG MODE: Truncated dataset to {len(self.image_paths)} images!")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # 1. Read Image (OpenCV reads as BGR by default)
            img = cv2.imread(self.image_paths[idx])
            if img is None: return torch.zeros(3, 256, 256) 
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 2. Resize & Normalize
            img = cv2.resize(img, (256, 256))
            img = (np.asarray(img)/255.0).astype(np.float32)
            
            # 3. Create Tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)

            return img_tensor
        except:
            return torch.zeros(3, 256, 256)

def plot_training_curves(log_file):
    try:
        df = pd.read_csv(log_file)
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange')
        plt.title('AdaLOLIE Training Progress')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    except: 
        print("Error in plot_training_curves!")


# ###
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Training on: {device}")
    
    model = AdaLOLIE_Net().to(device)
    loss_fn = AdaLOLIELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # Auto-Resume Logic
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"🔄 Checkpoint found! Resuming...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        print("🆕 Starting fresh training.")
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f: f.write("epoch,train_loss,val_loss\n")

    # limit=None means use ALL images.
    train_loader = DataLoader(
        MiningDataset(TRAIN_PATH, limit=None), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        MiningDataset(VAL_PATH, limit=None), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for imgs in loop:
            imgs = imgs.to(device)
            enhanced = model(imgs)
            loss = loss_fn(enhanced, imgs)
            
            optimizer.zero_grad()
            loss.backward()
            
            # ###
            # GRADIENT CLIPPING
            # Prevents model explosion from bad images
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val  ]")
        
        with torch.no_grad():
            for imgs in val_loop:
                imgs = imgs.to(device)
                val_loss += loss_fn(model(imgs), imgs).item()
                val_loop.set_postfix(loss=val_loss / (val_loop.n + 1))
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"--> Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "adalolie_best.pth"))
            print("    🌟 Best Model Updated!")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'loss': avg_train_loss
        }, RESUME_CHECKPOINT)
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"adalolie_epoch_{epoch+1}.pth"))
            
    plot_training_curves(LOG_FILE)

if __name__ == "__main__":
    # train()
    plot_training_curves(LOG_FILE)