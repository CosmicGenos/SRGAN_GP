import torch
import torch.nn as nn
import lpips
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

class ImageEvaluationMetrics:
    def __init__(self, device):
        self.device = device
        self.mse_criterion = nn.MSELoss().to(device)
        self.lpips_criterion = lpips.LPIPS(net='alex').to(device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_and_prepare_image(self, image_path):

        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor

    def prepare_images(self, sr, hr):

        sr_01 = torch.clamp((sr + 1) / 2, 0, 1)
        hr_01 = torch.clamp((hr + 1) / 2, 0, 1)


        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        sr_lpips = normalize(sr_01)
        hr_lpips = normalize(hr_01)

        return sr_01, hr_01, sr_lpips, hr_lpips

    def calculate_ssim(self, sr, hr, window_size=11):

        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2

        mu1 = F.avg_pool2d(sr, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(hr, window_size, stride=1, padding=window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(sr * sr, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(hr * hr, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(sr * hr, window_size, stride=1, padding=window_size//2) - mu1_mu2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()

    def evaluate_images(self, generator, lr_path, hr_path):

        generator.eval()
        metrics = {}

        with torch.no_grad():

            lr = self.load_and_prepare_image(lr_path)
            hr = self.load_and_prepare_image(hr_path)

            sr = generator(lr)

            sr_01, hr_01, sr_lpips, hr_lpips = self.prepare_images(sr, hr)

            mse = self.mse_criterion(sr_01, hr_01).item()
            psnr = -10 * torch.log10(torch.tensor(mse + 1e-8))
            ssim = self.calculate_ssim(sr_01, hr_01)
            lpips_value = self.lpips_criterion(sr_lpips, hr_lpips).mean()

            metrics = {
                'mse': mse,
                'psnr': psnr.item(),
                'ssim': ssim.item(),
                'lpips': lpips_value.item()
            }

        generator.train()
        return metrics

    def evaluate_directory(self, generator, lr_dir, hr_dir):

        total_metrics = {'psnr': 0, 'ssim': 0, 'mse': 0, 'lpips': 0}
        n_samples = 0

        lr_files = sorted([f for f in Path(lr_dir).glob('*.png')])
        hr_files = sorted([f for f in Path(hr_dir).glob('*.png')])

        for lr_path, hr_path in zip(lr_files, hr_files):
            metrics = self.evaluate_images(generator, lr_path, hr_path)

            for k, v in metrics.items():
                total_metrics[k] += v
            n_samples += 1

        avg_metrics = {k: v/n_samples for k, v in total_metrics.items()}
        return avg_metrics