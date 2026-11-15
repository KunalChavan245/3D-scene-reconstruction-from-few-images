"""A tiny synthetic scene with exact ground truth, for CPU tests and the sandbox demo.

The scene is a textured wall with a few blobs floating in front of it, seen by
a forward-facing camera rig (the LLFF situation in miniature). Images and depth
are rendered with the reference rasteriser, so poses, depth and geometry are
known exactly. Nothing here is used for the paper's results; it exists so that
every code path can be exercised without a GPU or a dataset download.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import SceneData
from .geometry import apply_sim3_to_c2w, invert_se3, scale_intrinsics
from .priors import MultiViewPrior
from .render import render, rgb_to_sh


def _gt_gaussians(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.linspace(-5.5, 5.5, 23), np.linspace(-4.0, 4.0, 17))
    wall = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 6.0)], axis=-1)
    wall_rgb = 0.35 + 0.3 * rng.uniform(size=(len(wall), 3))
    wall_rgb[:, 0] += 0.25 * np.sin(wall[:, 0] * 1.3)
    wall_rgb[:, 2] += 0.25 * np.cos(wall[:, 1] * 1.7)
    wall_scale = np.full(len(wall), 0.3)

    blobs, blob_rgb = [], []
    for centre, colour in [((-0.9, 0.2, 3.2), (0.9, 0.2, 0.2)), ((0.8, -0.3, 3.8), (0.2, 0.8, 0.3)),
                           ((0.0, 0.5, 4.4), (0.2, 0.3, 0.9))]:
        pts = np.array(centre) + rng.normal(scale=0.22, size=(50, 3))
        blobs.append(pts)
        blob_rgb.append(np.clip(np.array(colour) + rng.normal(scale=0.08, size=(50, 3)), 0, 1))
    blobs, blob_rgb = np.concatenate(blobs), np.concatenate(blob_rgb)

    xyz = np.concatenate([wall, blobs])
    rgb = np.clip(np.concatenate([wall_rgb, blob_rgb]), 0, 1)
    scale = np.concatenate([wall_scale, np.full(len(blobs), 0.09)])
    n = len(xyz)
    quats = np.tile([1.0, 0, 0, 0], (n, 1))
    return {
        "means": torch.tensor(xyz, dtype=torch.float32),
        "scales": torch.log(torch.tensor(scale, dtype=torch.float32))[:, None].repeat(1, 3),
        "quats": torch.tensor(quats, dtype=torch.float32),
        "opacities": torch.logit(torch.full((n,), 0.97)),
        "sh0": rgb_to_sh(torch.tensor(rgb, dtype=torch.float32))[:, None, :],
        "shN": torch.zeros(n, 0, 3),
    }


def _look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera matrix for a camera at ``centre`` looking at ``target`` (OpenCV axes)."""
    fwd = target - centre
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, -1.0, 0.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, centre
    return invert_se3(c2w)


def make_synthetic_scene(n_cams: int = 17, width: int = 64, height: int = 48, seed: int = 0):
    """Return ``(SceneData, gt_depth (N, H, W), gt_gaussians)``."""
    rng = np.random.default_rng(seed)
    gt = _gt_gaussians(seed)
    centres = np.stack(
        [rng.uniform(-1.2, 1.2, n_cams), rng.uniform(-0.6, 0.6, n_cams), rng.uniform(-0.2, 0.2, n_cams)], axis=-1
    )
    centres = centres[np.argsort(centres[:, 0])]
    w2c = np.stack([_look_at(c, np.array([0.0, 0.0, 4.5])) for c in centres])
    focal = 1.1 * width
    K = np.tile(np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]]), (n_cams, 1, 1))

    with torch.no_grad():
        rgb, depth, _, _ = render(
            gt, torch.tensor(w2c, dtype=torch.float32), torch.tensor(K, dtype=torch.float32), width, height,
            backend="torch",
        )
    images = (rgb.clamp(0, 1).numpy() * 255).round().astype(np.uint8)

    # "SfM" points: Gaussian centres, each observed by the cameras whose image contains it.
    xyz = gt["means"].numpy().astype(np.float64)
    cols = (render_colours(gt) * 255).round().astype(np.uint8)
    views = []
    for p in xyz:
        cam = np.einsum("nij,j->ni", w2c[:, :3, :3], p) + w2c[:, :3, 3]
        u = focal * cam[:, 0] / cam[:, 2] + width / 2
        v = focal * cam[:, 1] / cam[:, 2] + height / 2
        views.append(np.nonzero((cam[:, 2] > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height))[0])
    scene = SceneData(
        name="synthetic",
        names=[f"view_{i:03d}.png" for i in range(n_cams)],
        images=images,
        K=K,
        w2c=w2c,
        points=xyz,
        points_rgb=cols,
        point_views=views,
    )
    return scene, depth.numpy(), gt


def render_colours(gaussians: dict) -> np.ndarray:
    from .render import sh_to_rgb

    return sh_to_rgb(gaussians["sh0"][:, 0, :]).clamp(0, 1).numpy()


def fake_multiview_prior(
    scene: SceneData,
    gt_depth: np.ndarray,
    train_idx: list[int],
    scale: float = 0.37,
    pose_noise: float = 0.0,
    depth_noise: float = 0.0,
    bad_region: bool = False,
    proc_scale: float = 1.0,
    seed: int = 0,
) -> MultiViewPrior:
    """Simulate what a multi-view model returns for the training views.

    The poses and depth live in a different world frame (random rotation, a
    scale change and a shift), as a real model's output does. ``bad_region``
    corrupts the depth in one image corner and marks it low-confidence, which is
    what the confidence weighting is supposed to cope with.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    t = rng.normal(size=3)
    c2w = apply_sim3_to_c2w(invert_se3(scene.w2c[train_idx]), scale, Q, t)
    if pose_noise > 0:
        c2w[:, :3, 3] += rng.normal(scale=pose_noise * scale, size=(len(train_idx), 3))
    depth = gt_depth[train_idx] * scale
    conf = np.full_like(depth, 5.0)
    if depth_noise > 0:
        depth = depth * (1 + rng.normal(scale=depth_noise, size=depth.shape))
    if bad_region:
        h, w = depth.shape[1:]
        depth[:, : h // 2, : w // 2] *= 0.5
        conf[:, : h // 2, : w // 2] = 1.0
    K = scene.K[train_idx]
    if proc_scale != 1.0:
        h, w = depth.shape[1:]
        nh, nw = round(h * proc_scale), round(w * proc_scale)
        resize = lambda a: torch.nn.functional.interpolate(  # noqa: E731
            torch.tensor(a)[:, None], size=(nh, nw), mode="bilinear", align_corners=False
        )[:, 0].numpy()
        depth, conf = resize(depth), resize(conf)
        K = scale_intrinsics(K, nw / w, nh / h)
    return MultiViewPrior(
        names=[scene.names[i] for i in train_idx],
        w2c=invert_se3(c2w),
        K=K,
        depth=depth.astype(np.float32),
        conf=conf.astype(np.float32),
    )


def fake_mono_disparity(gt_depth: np.ndarray, scale: float = 3.0, shift: float = 0.4) -> np.ndarray:
    """A perfect monocular prior: inverse depth up to an unknown scale and shift."""
    return (scale / np.clip(gt_depth, 1e-6, None) + shift).astype(np.float32)
