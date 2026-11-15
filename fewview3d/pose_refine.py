"""Test-time camera pose refinement with frozen Gaussians.

Under the honest protocol the scene lives in the coordinate frame of the
*estimated* training poses, so ground-truth test poses cannot be used directly.
They are first mapped into that frame with a Sim(3) fitted on the training
cameras, then each test camera is refined by minimising the photometric error
against its own image while the Gaussians stay fixed. This is the evaluation
used by pose-free methods such as InstantSplat and CF-3DGS. It only moves the
camera; it never changes the scene.
"""

from __future__ import annotations

import torch

from .geometry import se3_exp
from .losses import photometric_loss
from .render import render


def refine_camera_pose(
    params: dict,
    w2c_init: torch.Tensor,
    K: torch.Tensor,
    target: torch.Tensor,
    steps: int = 200,
    lr: float = 2e-3,
    sh_degree: int = 0,
    backend: str = "auto",
    lambda_dssim: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Optimise one camera (4x4 ``w2c``) against ``target`` (H, W, 3). Returns the best pose found."""
    frozen = {k: v.detach() for k, v in params.items()}
    H, W = target.shape[:2]
    xi = torch.zeros(6, device=w2c_init.device, dtype=w2c_init.dtype, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.01 ** (1.0 / max(steps, 1)))
    best_loss, best_pose, first_loss = float("inf"), w2c_init.detach().clone(), None
    for _ in range(steps + 1):
        pose = se3_exp(xi) @ w2c_init
        rgb, _, _, _ = render(frozen, pose[None], K[None], W, H, sh_degree=sh_degree, backend=backend)
        loss = photometric_loss(rgb, target[None], lambda_dssim)
        value = loss.item()
        if first_loss is None:
            first_loss = value
        if value < best_loss:
            best_loss, best_pose = value, pose.detach().clone()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return best_pose, {"loss_before": first_loss, "loss_after": best_loss}
