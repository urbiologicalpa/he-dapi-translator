"""
losses.py - Composite losses for virtual staining
  - L1, SSIM, DISTS (perceptual, from FLIM paper), GAN, Cycle, NCE (CUT), Nuclei Dice
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """SSIM loss: 1 - SSIM (uses kornia if available, else simple MSE proxy)
    NOTE 2026-10-03: kornia NOT installed on 3.7GB box -> fallback L1*0.1+cosine is used.
    Offline PSNR/L1 are primary until kornia added on bigger machine. Do NOT install mid-train.
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self._kornia = None
        try:
            import kornia
            self._kornia = kornia.losses.SSIMLoss(window_size=window_size)
        except ImportError:
            pass

    def forward(self, pred, target):
        if self._kornia:
            return self._kornia(pred, target)
        # Fallback: 1 - normalized cross-correlation proxy
        # Simple: L1 in normalized space
        return F.l1_loss(pred, target) * 0.1 + (1 - F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1).mean())

class DISTSLoss(nn.Module):
    """
    DISTS loss placeholder - full impl needs VGG features.
    Falls back to LPIPS-like L2 on normalized features if torchvision VGG available.
    From: FLIM->H&E paper (Nature npj Imaging 2024) uses DISTS for texture.
    """
    def __init__(self):
        super().__init__()
        self._vgg = None
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights="DEFAULT").features[:16].eval()
            for p in vgg.parameters(): p.requires_grad=False
            self._vgg = vgg
        except Exception:
            pass

    def forward(self, pred, target):
        # pred/target: BxCxHxW in [-1,1] or [0,1] -> normalize to [0,1]
        pred01 = (pred+1)/2 if pred.min() < 0 else pred
        target01 = (target+1)/2 if target.min() < 0 else target
        if self._vgg is None:
            return F.l1_loss(pred01, target01)
        # VGG features
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485,0.456,0.406], device=pred.device).view(1,3,1,1)
        std = torch.tensor([0.229,0.224,0.225], device=pred.device).view(1,3,1,1)
        # Handle grayscale target (1ch) -> repeat to 3ch
        if pred01.shape[1]==1:
            pred01 = pred01.repeat(1,3,1,1)
        if target01.shape[1]==1:
            target01 = target01.repeat(1,3,1,1)
        pred_n = (pred01 - mean) / std
        target_n = (target01 - mean) / std
        f_pred = self._vgg(pred_n)
        f_target = self._vgg(target_n)
        return F.mse_loss(f_pred, f_target)

class NucleiDiceLoss(nn.Module):
    """
    Auxiliary nuclei segmentation loss: runs pretrained StarDist/Cellpose features?
    Simplified: threshold DAPI at thresh -> binary mask, compute Dice between real vs fake DAPI.
    NOTE 2026-10-03: thresh was 0.5 but HEMIT DAPI in [0,1] has mean 0.02-0.05 max ~0.3,
    so 0.5 gave empty masks -> vacuous Dice=1.0. Fixed default to 0.1 (covers nuclei peaks).
    """
    def __init__(self, thresh=0.1, smooth=1e-5):
        super().__init__()
        self.thresh = thresh
        self.smooth = smooth

    def forward(self, fake_dapi, real_dapi):
        # fake/real: Bx1xHxW in [0,1]
        pred_mask = (fake_dapi > self.thresh).float()
        true_mask = (real_dapi > self.thresh).float()
        intersection = (pred_mask * true_mask).sum(dim=(1,2,3))
        union = pred_mask.sum(dim=(1,2,3)) + true_mask.sum(dim=(1,2,3))
        dice = (2*intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class ColorStatsLoss(nn.Module):
    """Match per-channel mean/std so H&E stain colour/contrast is not washed out."""

    def forward(self, pred, target):
        mean_loss = F.l1_loss(pred.mean(dim=(2, 3)), target.mean(dim=(2, 3)))
        std_loss = F.l1_loss(pred.std(dim=(2, 3), unbiased=False),
                             target.std(dim=(2, 3), unbiased=False))
        return mean_loss + std_loss


class GradientLoss(nn.Module):
    """L1 on horizontal/vertical image gradients; discourages blurred outputs."""

    def forward(self, pred, target):
        dx = F.l1_loss(pred[:, :, :, 1:] - pred[:, :, :, :-1],
                       target[:, :, :, 1:] - target[:, :, :, :-1])
        dy = F.l1_loss(pred[:, :, 1:, :] - pred[:, :, :-1, :],
                       target[:, :, 1:, :] - target[:, :, :-1, :])
        return dx + dy
