"""Run VGGT on the training views only and store its poses, depth and confidence.

VGGT (Wang et al., CVPR 2025) predicts cameras and dense depth for a set of
images in one forward pass. Feeding it exactly the few training views, and
nothing else, is what makes the downstream poses "honest".

Requires the ``vggt`` package: ``pip install --no-deps git+https://github.com/facebookresearch/vggt``
(plus ``einops``, ``safetensors`` and ``huggingface_hub``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import MultiViewPrior

PATCH = 14
TARGET = 518


def vggt_input_size(width: int, height: int, target: int = TARGET) -> tuple[int, int]:
    """Network input size (w, h): longest side ``target``, both sides multiples of the patch size.

    Unlike VGGT's default loader this never crops or pads, so every input pixel
    maps to an image pixel by a pure (slightly anisotropic) resize, and the
    predicted intrinsics can be rescaled exactly.
    """
    scale = target / max(width, height)
    w = max(PATCH, round(width * scale / PATCH) * PATCH)
    h = max(PATCH, round(height * scale / PATCH) * PATCH)
    return w, h


def preprocess(images: np.ndarray) -> torch.Tensor:
    """uint8 images (S, H, W, 3) to a float tensor (S, 3, h, w) in [0, 1] at the network size."""
    S, H, W, _ = images.shape
    w, h = vggt_input_size(W, H)
    x = torch.as_tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return x.clamp(0, 1)


def run_vggt(images: np.ndarray, names: list[str], model=None, device: str = "cuda",
             checkpoint: str = "facebook/VGGT-1B") -> MultiViewPrior:
    """Predict poses, intrinsics, depth and confidence for ``images`` (S, H, W, 3) uint8."""
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if model is None:
        model = VGGT.from_pretrained(checkpoint).to(device).eval()
    x = preprocess(images).to(device)
    use_amp = torch.device(device).type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=use_amp):
        pred = model(x)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"].float(), x.shape[-2:])
    return predictions_to_prior(
        names,
        extr[0].cpu().numpy(),
        intr[0].cpu().numpy(),
        pred["depth"][0, ..., 0].float().cpu().numpy(),
        pred["depth_conf"][0].float().cpu().numpy(),
    )


def predictions_to_prior(names, extrinsics_3x4, intrinsics, depth, conf) -> MultiViewPrior:
    """Pack raw model outputs (camera-from-world 3x4 extrinsics, OpenCV) into a ``MultiViewPrior``."""
    S = len(names)
    w2c = np.tile(np.eye(4), (S, 1, 1))
    w2c[:, :3, :4] = np.asarray(extrinsics_3x4, dtype=np.float64)
    if not (len(intrinsics) == len(depth) == len(conf) == S):
        raise ValueError("the model returned a different number of views than requested")
    return MultiViewPrior(list(names), w2c, np.asarray(intrinsics, dtype=np.float64),
                          np.asarray(depth, dtype=np.float32), np.asarray(conf, dtype=np.float32))
