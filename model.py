import torch
import torch.nn as nn

# ###
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))

# ###
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
        number_f = 32
        
        # ###
        self.e_conv1 = DepthwiseSeparableConv(3, number_f) 
        self.e_conv2 = DepthwiseSeparableConv(number_f, number_f)

        # Injecting Attention here
        self.attention = SpatialAttention()

        self.e_conv3 = DepthwiseSeparableConv(number_f, number_f)
        self.e_conv4 = DepthwiseSeparableConv(number_f, number_f)
        self.e_conv5 = nn.Conv2d(number_f, 24, 3, 1, 1, bias=True)
        
        # --- REQUIREMENT 1: Weight Initialization ---
        # Forces the model to start in a "safe" state
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.e_conv1(x)
        x2 = self.e_conv2(x1)
        
        # ### SPEAK: "Here, the attention map weights the features before the final curve estimation."
        # Adaptive Attention
        attn_map = self.attention(x2)
        x2 = x2 * attn_map 
        
        x3 = self.e_conv3(x2)
        x4 = self.e_conv4(x3)
        
        # ###
        # Tanh Activation
        # Forces parameters to be between -1 and 1.
        # WITHOUT THIS, IMAGE TURNS GRAY/WHITE.
        curve_params = torch.tanh(self.e_conv5(x4))
        
        # ###
        # Curve Application
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(curve_params, 3, dim=1)
        
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_image = x + r8 * (torch.pow(x, 2) - x)
        
        return enhanced_image