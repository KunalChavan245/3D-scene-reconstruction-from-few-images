"""Gaussian rasterisation behind one interface, with two backends.

* ``gsplat``: the CUDA rasteriser from the gsplat library. Used for all real experiments.
* ``torch``: a small dense reference rasteriser written in plain PyTorch. It is
  far too slow for real scenes, but it runs on a CPU, so the whole pipeline
  (training loop, densification, depth losses, pose refinement) can be tested
  end to end without a GPU. It renders view-independent colour only (SH degree 0).

Both return ``rgb`` (C, H, W, 3), ``depth`` (C, H, W) as alpha-normalised expected
z-depth, ``alpha`` (C, H, W) and an ``info`` dict in the format expected by
gsplat's densification strategies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * SH_C0 + 0.5


def pick_backend(backend: str, device: torch.device | str) -> str:
    if backend != "auto":
        return backend
    return "gsplat" if torch.device(device).type == "cuda" else "torch"


def render(
    params: dict,
    w2c: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 0,
    backend: str = "auto",
    near: float = 0.01,
    background: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Render Gaussians from ``C`` cameras. ``params`` holds the raw (pre-activation) tensors."""
    backend = pick_backend(backend, params["means"].device)
    if backend == "gsplat":
        return _render_gsplat(params, w2c, K, width, height, sh_degree, near, background)
    if backend == "torch":
        return _render_torch(params, w2c, K, width, height, near, background)
    raise ValueError(f"unknown backend {backend!r}")


def _render_gsplat(params, w2c, K, width, height, sh_degree, near, background):
    from gsplat import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=w2c,
        Ks=K,
        width=width,
        height=height,
        sh_degree=sh_degree,
        near_plane=near,
        packed=False,
        render_mode="RGB+ED",
        backgrounds=background,
    )
    return out[..., :3], out[..., 3], alpha[..., 0], info


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternions (N, 4) in (w, x, y, z) order to rotation matrices; normalises first."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def _render_torch(params, w2c, K, width, height, near, background, eps2d: float = 0.3):
    means = params["means"]
    N, C = means.shape[0], w2c.shape[0]
    device, dtype = means.device, means.dtype
    scales = torch.exp(params["scales"])
    opac = torch.sigmoid(params["opacities"])
    rgb_g = sh_to_rgb(params["sh0"][:, 0, :])
    Rq = quat_to_rotmat(params["quats"])
    cov3d = Rq @ torch.diag_embed(scales**2) @ Rq.transpose(1, 2)

    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    pc = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    z = pc[..., 2]
    zc = z.clamp(min=near)
    fx, fy = K[:, 0, 0, None], K[:, 1, 1, None]
    cx, cy = K[:, 0, 2, None], K[:, 1, 2, None]
    means2d = torch.stack([fx * pc[..., 0] / zc + cx, fy * pc[..., 1] / zc + cy], dim=-1)  # (C, N, 2)

    # EWA splatting: project each 3D covariance with the local affine approximation.
    J = torch.zeros(C, N, 2, 3, device=device, dtype=dtype)
    J[..., 0, 0] = fx / zc
    J[..., 0, 2] = -fx * pc[..., 0] / zc**2
    J[..., 1, 1] = fy / zc
    J[..., 1, 2] = -fy * pc[..., 1] / zc**2
    cov_cam = R[:, None] @ cov3d[None] @ R[:, None].transpose(-1, -2)
    cov2d = J @ cov_cam @ J.transpose(-1, -2)
    a = cov2d[..., 0, 0] + eps2d
    b = cov2d[..., 0, 1]
    c = cov2d[..., 1, 1] + eps2d
    det = (a * c - b * b).clamp(min=1e-12)
    radius = 3.0 * torch.sqrt(((a + c) / 2 + torch.sqrt(((a - c) / 2) ** 2 + b * b)).clamp(min=1e-12))
    visible = (
        (z > near)
        & (means2d[..., 0] + radius > 0)
        & (means2d[..., 0] - radius < width)
        & (means2d[..., 1] + radius > 0)
        & (means2d[..., 1] - radius < height)
    )
    radii = (torch.ceil(radius) * visible).int()[..., None].expand(-1, -1, 2)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (P, 2)

    rgbs, depths, alphas = [], [], []
    for ci in range(C):
        idx = torch.nonzero(visible[ci], as_tuple=False)[:, 0]
        idx = idx[torch.argsort(z[ci, idx])]
        if idx.numel() == 0:
            rgbs.append(torch.zeros(height * width, 3, device=device, dtype=dtype))
            depths.append(torch.zeros(height * width, device=device, dtype=dtype))
            alphas.append(torch.zeros(height * width, device=device, dtype=dtype))
            continue
        d = pix[None, :, :] - means2d[ci, idx][:, None, :]  # (M, P, 2)
        ia, ib, ic = (c[ci, idx] / det[ci, idx]), (-b[ci, idx] / det[ci, idx]), (a[ci, idx] / det[ci, idx])
        power = -0.5 * (ia[:, None] * d[..., 0] ** 2 + ic[:, None] * d[..., 1] ** 2) - ib[:, None] * d[..., 0] * d[..., 1]
        alpha = (opac[idx][:, None] * torch.exp(power)).clamp(max=0.999)
        alpha = torch.where(alpha < 1.0 / 255.0, torch.zeros_like(alpha), alpha)
        trans = torch.cumprod(1.0 - alpha, dim=0)
        trans = torch.cat([torch.ones_like(trans[:1]), trans[:-1]], dim=0)
        w = alpha * trans  # (M, P)
        acc = w.sum(0)
        rgbs.append(w.t() @ rgb_g[idx])
        depths.append((w * z[ci, idx][:, None]).sum(0) / acc.clamp(min=1e-10))
        alphas.append(acc)

    rgb = torch.stack(rgbs).reshape(C, height, width, 3)
    alpha_img = torch.stack(alphas).reshape(C, height, width)
    if background is not None:
        rgb = rgb + (1.0 - alpha_img[..., None]) * background.reshape(-1, 1, 1, 3)
    depth = torch.stack(depths).reshape(C, height, width)
    info = {"means2d": means2d, "radii": radii, "width": width, "height": height, "n_cameras": C, "gaussian_ids": None}
    return rgb, depth, alpha_img, info
