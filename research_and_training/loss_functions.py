import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLOLIELoss(nn.Module):
    def __init__(self):
        super(AdaLOLIELoss, self).__init__()
        # Pre-create edge kernels
        self.register_buffer("k_left", torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("k_right", torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("k_up", torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("k_down", torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    # --- NEW: Illumination Smoothness Loss from original Zero-DCE ---
    def get_illumination_smoothness_loss(self, x_r):
        batch_size = x_r.size()[0]
        h_x = x_r.size()[2]
        w_x = x_r.size()[3]
        count_h = (x_r.size()[2]-1) * x_r.size()[3]
        count_w = x_r.size()[2] * (x_r.size()[3] - 1)
        
        # Calculates the squared difference between neighboring pixels in the curve map
        h_tv = torch.pow((x_r[:,:,1:,:] - x_r[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x_r[:,:,:,1:] - x_r[:,:,:,:w_x-1]), 2).sum()
        
        return 2 * (h_tv/count_h + w_tv/count_w) / batch_size

    def get_exposure_loss(self, enhanced, patch_size=16, mean_val=0.50): 
        pool = nn.AvgPool2d(patch_size)
        mean_curr = pool(enhanced)
        return torch.mean(torch.pow(mean_curr - mean_val, 2))

    def get_color_loss(self, enhanced):
        mean_rgb = torch.mean(enhanced, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mb - mg, 2)
        return torch.mean(torch.pow(torch.sqrt(d_rg + d_rb + d_gb), 2))

    def get_spatial_loss(self, enhanced, original):
        pool = nn.AvgPool2d(4)
        enh_p = pool(enhanced)
        org_p = pool(original)
        
        enh_g = 0.299*enh_p[:,0,:,:] + 0.587*enh_p[:,1,:,:] + 0.114*enh_p[:,2,:,:]
        org_g = 0.299*org_p[:,0,:,:] + 0.587*org_p[:,1,:,:] + 0.114*org_p[:,2,:,:]
        enh_g, org_g = enh_g.unsqueeze(1), org_g.unsqueeze(1)

        mask = (org_g > 0.02).float()

        d_left = torch.pow(F.conv2d(org_g, self.k_left, padding=1) - F.conv2d(enh_g, self.k_left, padding=1), 2)
        d_right = torch.pow(F.conv2d(org_g, self.k_right, padding=1) - F.conv2d(enh_g, self.k_right, padding=1), 2)
        d_up = torch.pow(F.conv2d(org_g, self.k_up, padding=1) - F.conv2d(enh_g, self.k_up, padding=1), 2)
        d_down = torch.pow(F.conv2d(org_g, self.k_down, padding=1) - F.conv2d(enh_g, self.k_down, padding=1), 2)
        
        masked_diff = (d_left + d_right + d_up + d_down) * mask
        return torch.sum(masked_diff) / (torch.sum(mask) + 1e-8)

    def get_grayscale_loss(self, enhanced):
        min_channel, _ = torch.min(enhanced, dim=1, keepdim=True)
        return torch.mean(min_channel)
    
    def get_glare_loss(self, enhanced, limit=0.85):
        return torch.mean(F.relu(enhanced - limit))

    # --- UPDATED: Forward pass now accepts x_r ---
    def forward(self, enhanced, original, x_r):
        L_exp = self.get_exposure_loss(enhanced)
        L_col = self.get_color_loss(enhanced)
        L_spa = self.get_spatial_loss(enhanced, original) 
        L_tv = self.get_grayscale_loss(enhanced)
        L_glare = self.get_glare_loss(enhanced)
        
        # Calculate the new smoothness loss
        L_smooth = self.get_illumination_smoothness_loss(x_r)

        # I added a baseline weight of 10.0 for L_smooth. You may want to tune this in Optuna later!
        return (5.65)*L_exp + (4.87)*L_col + (7.34)*L_spa + (4.32)*L_tv + (9.53)*L_glare + (10.0)*L_smooth