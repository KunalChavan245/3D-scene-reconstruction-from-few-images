"""Image quality metrics: PSNR, SSIM and LPIPS (VGG), the trio every few-view paper reports."""

from __future__ import annotations

import warnings

import torch

from .losses import ssim as _ssim

_LPIPS_CACHE: dict = {}


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred.clamp(0, 1) - target) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp(min=1e-12)))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM for a single image (H, W, 3) in [0, 1]."""
    return float(_ssim(pred.clamp(0, 1)[None], target[None]))


def lpips(pred: torch.Tensor, target: torch.Tensor, net: str = "vgg") -> float | None:
    """LPIPS for a single image (H, W, 3) in [0, 1]. Returns None if the package is missing."""
    try:
        import lpips as lpips_lib
    except ImportError:
        warnings.warn("the 'lpips' package is not installed; LPIPS is reported as null")
        return None
    key = (net, str(pred.device))
    if key not in _LPIPS_CACHE:
        _LPIPS_CACHE[key] = lpips_lib.LPIPS(net=net, verbose=False).to(pred.device).eval()
    a = pred.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1
    b = target.permute(2, 0, 1)[None] * 2 - 1
    with torch.no_grad():
        return float(_LPIPS_CACHE[key](a, b))


def image_metrics(pred: torch.Tensor, target: torch.Tensor, with_lpips: bool = True) -> dict:
    out = {"psnr": psnr(pred, target), "ssim": ssim(pred, target)}
    out["lpips"] = lpips(pred, target) if with_lpips else None
    return out
