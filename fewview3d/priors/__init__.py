"""Depth and pose priors: on-disk format and loaders.

Layout under a scene's prior directory::

    mono/<image stem>.npy     relative inverse depth from a monocular model (any resolution)
    vggt_<n>views.npz         poses, intrinsics, depth and confidence predicted from the
                              n training views only

The ``.npz`` stores the names of the images it was computed from. The loader
refuses to use it with a different set of training views, which is what
guarantees that no held-out view ever influenced the poses or the depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MultiViewPrior:
    names: list[str]
    w2c: np.ndarray  # (S, 4, 4) world-to-camera, OpenCV, in the model's own frame
    K: np.ndarray  # (S, 3, 3) at the processed resolution
    depth: np.ndarray  # (S, h, w) z-depth in the same frame as the poses
    conf: np.ndarray  # (S, h, w) confidence, larger is more reliable

    @property
    def proc_hw(self) -> tuple[int, int]:
        return int(self.depth.shape[1]), int(self.depth.shape[2])


def mv_prior_path(prior_dir: str | Path, n_views: int, model: str = "vggt") -> Path:
    return Path(prior_dir) / f"{model}_{n_views}views.npz"


def save_mv_prior(path: str | Path, prior: MultiViewPrior) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        names=np.array(prior.names),
        w2c=prior.w2c.astype(np.float64),
        K=prior.K.astype(np.float64),
        depth=prior.depth.astype(np.float32),
        conf=prior.conf.astype(np.float32),
    )


def load_mv_prior(path: str | Path, expected_names: list[str] | None = None) -> MultiViewPrior:
    data = np.load(path, allow_pickle=False)
    prior = MultiViewPrior(
        names=[str(n) for n in data["names"]],
        w2c=data["w2c"],
        K=data["K"],
        depth=data["depth"],
        conf=data["conf"],
    )
    if expected_names is not None and list(expected_names) != prior.names:
        raise ValueError(
            f"{path} was computed from views {prior.names}, but this run trains on {list(expected_names)}. "
            "Recompute the prior from exactly the training views."
        )
    return prior


def load_mono_prior(prior_dir: str | Path, names: list[str]) -> list[np.ndarray]:
    """Load one relative inverse-depth map per image name."""
    out = []
    for name in names:
        path = Path(prior_dir) / "mono" / (Path(name).stem + ".npy")
        if not path.exists():
            raise FileNotFoundError(f"missing monocular depth {path}; run scripts/precompute_mono_depth.py")
        out.append(np.load(path).astype(np.float32))
    return out


def resize_map(arr: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Bilinearly resize a (h, w) map to (height, width)."""
    t = torch.as_tensor(arr, dtype=torch.float32)[None, None]
    if t.shape[-2:] == (height, width):
        return t[0, 0]
    return F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
