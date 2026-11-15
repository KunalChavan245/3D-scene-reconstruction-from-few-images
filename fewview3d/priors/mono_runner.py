"""Monocular relative depth with Depth Anything V2 (Yang et al., NeurIPS 2024).

The model sees one image at a time, so it cannot leak anything across views.
Its output is relative inverse depth (larger = closer), known only up to an
unknown scale and shift per image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cuda"):
    from transformers import pipeline

    return pipeline(task="depth-estimation", model=model, device=device)


def predict_disparity(pipe, image: np.ndarray) -> np.ndarray:
    """uint8 image (H, W, 3) to a float32 relative inverse-depth map at the model's resolution."""
    from PIL import Image

    out = pipe(Image.fromarray(image))
    return out["predicted_depth"].squeeze().float().cpu().numpy().astype(np.float32)


def save_disparity(prior_dir: str | Path, name: str, disparity: np.ndarray) -> Path:
    path = Path(prior_dir) / "mono" / (Path(name).stem + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, disparity)
    return path
