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

    def get_exposure_loss(self, enhanced, patch_size=16, mean_val=0.55): 
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
        # Downsample
        pool = nn.AvgPool2d(4)
        enh_p_rgb = pool(enhanced)
        org_p_rgb = pool(original)
        
        # RGB to Grayscale
        enh_p_gray = 0.299 * enh_p_rgb[:, 0, :, :] + 0.587 * enh_p_rgb[:, 1, :, :] + 0.114 * enh_p_rgb[:, 2, :, :]
        org_p_gray = 0.299 * org_p_rgb[:, 0, :, :] + 0.587 * org_p_rgb[:, 1, :, :] + 0.114 * org_p_rgb[:, 2, :, :]

        enh_p_gray = enh_p_gray.unsqueeze(1)
        org_p_gray = org_p_gray.unsqueeze(1)

        # Create Mask
        mask = (org_p_gray > 0.05).float()

        # Convolutions with Padding=1
        d_left = torch.pow(F.conv2d(org_p_gray, self.k_left, padding=1) - F.conv2d(enh_p_gray, self.k_left, padding=1), 2)
        d_right = torch.pow(F.conv2d(org_p_gray, self.k_right, padding=1) - F.conv2d(enh_p_gray, self.k_right, padding=1), 2)
        d_up = torch.pow(F.conv2d(org_p_gray, self.k_up, padding=1) - F.conv2d(enh_p_gray, self.k_up, padding=1), 2)
        d_down = torch.pow(F.conv2d(org_p_gray, self.k_down, padding=1) - F.conv2d(enh_p_gray, self.k_down, padding=1), 2)
        
        masked_loss = (d_left + d_right + d_up + d_down) * mask

        # Normalize by Fixed Area, NOT Mask Sum
        # This prevents explosion when mask is nearly empty (pitch black images).
        batch, channel, h, w = org_p_gray.shape
        return torch.sum(masked_loss) / (batch * channel * h * w)

    def get_grayscale_loss(self, enhanced):
        min_channel, _ = torch.min(enhanced, dim=1, keepdim=True)
        return torch.mean(min_channel)
    
    def get_glare_loss(self, enhanced, limit=0.85):
        return torch.mean(F.relu(enhanced - limit))

    def forward(self, enhanced, original):
        L_exp = self.get_exposure_loss(enhanced)
        L_col = self.get_color_loss(enhanced)
        L_spa = self.get_spatial_loss(enhanced, original) 
        L_tv = self.get_grayscale_loss(enhanced)
        L_glare = self.get_glare_loss(enhanced)

        # Custom finetuned weights for each loss function by running multiple debug training runs
        return 8*L_exp + 0.5*L_col + 5*L_spa + 2*L_tv + 10*L_glare