import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGGLoss(nn.Module):
    def __init__(self, device='cuda', scale_factor=1/12.75):
        super(VGGLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(Device)
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.scale_factor = scale_factor


        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):

        sr = (sr + 1) / 2


        # hr = torch.clamp(hr, 0, 1)

        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std


        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)

        N, C, H, W = sr_features.size()


        loss = torch.mean((sr_features - hr_features) ** 2) * self.scale_factor / (H * W)
        return loss