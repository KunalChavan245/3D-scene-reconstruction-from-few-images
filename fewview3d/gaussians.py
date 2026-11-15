"""Gaussian parameter initialisation and optimisers."""

from __future__ import annotations

import numpy as np
import torch

from .render import rgb_to_sh


def knn_mean_distance(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance from each point to its k nearest neighbours."""
    from scipy.spatial import cKDTree

    if len(points) <= 1:
        return np.ones(len(points))
    k = min(k, len(points) - 1)
    dist, _ = cKDTree(points).query(points, k=k + 1)
    return dist[:, 1:].mean(axis=1)


def init_gaussians(
    xyz: np.ndarray,
    rgb: np.ndarray,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> torch.nn.ParameterDict:
    """Create isotropic Gaussians at ``xyz`` (N, 3) with colours ``rgb`` (N, 3) in [0, 1].

    Follows the 3DGS initialisation: the scale is the mean distance to the three
    nearest neighbours, opacity starts low, and only the DC colour term is set.
    """
    gen = torch.Generator().manual_seed(seed)
    xyz_t = torch.as_tensor(xyz, dtype=torch.float32)
    n = xyz_t.shape[0]
    dist = np.clip(knn_mean_distance(np.asarray(xyz, dtype=np.float64)), 1e-7, None)
    scales = torch.log(torch.as_tensor(dist * init_scale, dtype=torch.float32))[:, None].repeat(1, 3)
    quats = torch.rand(n, 4, generator=gen)
    opacities = torch.logit(torch.full((n,), init_opacity))
    sh0 = rgb_to_sh(torch.as_tensor(rgb, dtype=torch.float32))[:, None, :]
    shN = torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3)
    params = {"means": xyz_t, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh0, "shN": shN}
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in params.items()})


def create_optimizers(params: torch.nn.ParameterDict, scene_scale: float, cfg) -> dict:
    """One Adam optimiser per parameter group, as gsplat's densification strategy expects."""
    lrs = {
        "means": cfg.lr_means * scene_scale,
        "scales": cfg.lr_scales,
        "quats": cfg.lr_quats,
        "opacities": cfg.lr_opacities,
        "sh0": cfg.lr_sh0,
        "shN": cfg.lr_sh0 / 20.0,
    }
    return {
        name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in lrs.items()
    }


def random_points_in_bounds(xyz_ref: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random points and colours inside the bounding box of ``xyz_ref`` (init ablation)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile(xyz_ref, 1, axis=0), np.percentile(xyz_ref, 99, axis=0)
    return rng.uniform(lo, hi, size=(n, 3)), rng.uniform(0, 1, size=(n, 3))
