"""Photometric and depth-prior losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Photometric
# --------------------------------------------------------------------------- #
def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Mean SSIM between two image batches (B, H, W, 3) in [0, 1]. Same form as the 3DGS code."""
    x = img1.permute(0, 3, 1, 2)
    y = img2.permute(0, 3, 1, 2)
    C = x.shape[1]
    w = _gaussian_window(window_size, 1.5, x.device, x.dtype).expand(C, 1, -1, -1)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=C)
    mu_y = F.conv2d(y, w, padding=pad, groups=C)
    sxx = F.conv2d(x * x, w, padding=pad, groups=C) - mu_x**2
    syy = F.conv2d(y * y, w, padding=pad, groups=C) - mu_y**2
    sxy = F.conv2d(x * y, w, padding=pad, groups=C) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return ssim_map.mean()


def photometric_loss(render: torch.Tensor, target: torch.Tensor, lambda_dssim: float = 0.2) -> torch.Tensor:
    """The 3DGS colour loss: (1 - l) * L1 + l * (1 - SSIM)."""
    l1 = (render - target).abs().mean()
    if lambda_dssim <= 0:
        return l1
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim(render, target))


# --------------------------------------------------------------------------- #
# Depth priors
# --------------------------------------------------------------------------- #
def pearson_depth_loss(
    rendered_depth: torch.Tensor,
    prior_disparity: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """1 - Pearson correlation between rendered inverse depth and a monocular disparity map.

    Monocular models such as Depth Anything V2 predict *relative inverse depth*,
    known only up to an unknown scale and shift. The Pearson correlation is
    invariant to both, so no alignment is needed (the choice made by FSGS and
    SparseGS).
    """
    inv = 1.0 / rendered_depth.clamp(min=eps)
    x, y = inv.reshape(-1), prior_disparity.reshape(-1)
    if mask is not None:
        m = mask.reshape(-1) > 0
        x, y = x[m], y[m]
    if x.numel() < 2:
        return rendered_depth.sum() * 0.0
    x = x - x.mean()
    y = y - y.mean()
    corr = (x * y).sum() / (x.norm() * y.norm() + eps)
    return 1.0 - corr


def solve_scale_shift(
    src: torch.Tensor, dst: torch.Tensor, weight: torch.Tensor, mode: str = "scale_shift"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted least squares (a, b) minimising sum w * (a * src + b - dst)^2.

    ``mode`` is ``"none"`` (a=1, b=0), ``"scale"`` (b=0) or ``"scale_shift"``.
    """
    one, zero = src.new_tensor(1.0), src.new_tensor(0.0)
    if mode == "none":
        return one, zero
    w, s, d = weight.reshape(-1), src.reshape(-1), dst.reshape(-1)
    sw = w.sum().clamp(min=1e-12)
    if mode == "scale":
        a = (w * s * d).sum() / (w * s * s).sum().clamp(min=1e-12)
        return a, zero
    if mode != "scale_shift":
        raise ValueError(f"unknown alignment mode {mode!r}")
    ms, md = (w * s).sum() / sw, (w * d).sum() / sw
    var = (w * (s - ms) ** 2).sum()
    cov = (w * (s - ms) * (d - md)).sum()
    a = cov / var.clamp(min=1e-12)
    b = md - a * ms
    return a, b


def confidence_weights(
    conf: torch.Tensor | None, like: torch.Tensor, mode: str = "soft", lo_q: float = 0.2, hi_q: float = 0.8
) -> torch.Tensor:
    """Turn a foundation model's confidence map into per-pixel loss weights in [0, 1].

    * ``"none"``: every pixel counts equally (the no-confidence ablation).
    * ``"threshold"``: drop the least confident ``lo_q`` fraction of pixels.
    * ``"soft"``: linear ramp from 0 at the ``lo_q`` quantile to 1 at the ``hi_q`` quantile.

    Quantiles are taken per image, which makes the weights independent of the
    confidence range of a particular model (VGGT confidences are >= 1 and unbounded).
    """
    if conf is None or mode == "none":
        return torch.ones_like(like)
    flat = conf.reshape(-1).float()
    lo = torch.quantile(flat, lo_q)
    if mode == "threshold":
        return (conf > lo).to(like.dtype)
    if mode != "soft":
        raise ValueError(f"unknown confidence mode {mode!r}")
    hi = torch.quantile(flat, hi_q)
    return ((conf - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0).to(like.dtype)


def confidence_depth_loss(
    rendered_depth: torch.Tensor,
    prior_depth: torch.Tensor,
    weight: torch.Tensor,
    align: str = "none",
) -> torch.Tensor:
    """Confidence-weighted L1 between rendered depth and a (multi-view) depth prior.

    The prior is first aligned to the current render with a weighted least-squares
    scale (and shift), treated as a constant. With ``align="none"`` the prior is
    used as is, which is valid when the prior depth and the camera poses come from
    the same model and therefore share one metric frame (the honest-pose setting).
    The loss is divided by the weighted mean prior depth so that ``lambda`` has the
    same meaning across scenes of different scale.
    """
    valid = (prior_depth > 0).to(weight.dtype)
    w = weight * valid
    wsum = w.sum()
    if wsum <= 0:
        return rendered_depth.sum() * 0.0
    with torch.no_grad():
        a, b = solve_scale_shift(prior_depth, rendered_depth, w, align)
    target = a * prior_depth + b
    norm = ((w * target.abs()).sum() / wsum).clamp(min=1e-8)
    return (w * (rendered_depth - target).abs()).sum() / wsum / norm
