import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class AdaLOLIE_Net(nn.Module):
    def __init__(self):
        super(AdaLOLIE_Net, self).__init__()
        nf = 32
        
        # --- The Zero-DCE Feature Extraction ---
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # --- Your Custom Dual Attention Module ---
        self.ca = ChannelAttention(nf)
        self.sa = SpatialAttention()
        
        # --- The Zero-DCE Symmetrical Concatenation Decoder ---
        # Note: Input channels are doubled (nf * 2) because we concatenate skip connections
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Extract Features
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        
        # 2. Apply Dual Attention at the Bottleneck
        # This isolates the localized bright spots (e.g., headlamps) and scales their feature weights
        x4 = self.ca(x4) * x4
        x4 = self.sa(x4) * x4
        
        # 3. Symmetrical Skip Connections (Restoring original Zero-DCE structural preservation)
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        
        # 4. Final Curve Parameter Generation
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))
        
        # 5. Iterative Enhancement Application
        enhanced = x
        for i in range(8):
            r = x_r[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + r * (torch.pow(enhanced, 2) - enhanced)
            
        return enhanced, x_r