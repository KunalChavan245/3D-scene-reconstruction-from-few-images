import numpy as np
import pytest
import torch

from fewview3d.geometry import se3_exp
from fewview3d.render import quat_to_rotmat, render, rgb_to_sh


def one_gaussian(xyz, scale=0.2, opacity=0.9, rgb=(1.0, 0.0, 0.0)):
    return {
        "means": torch.tensor([xyz], dtype=torch.float32),
        "scales": torch.log(torch.full((1, 3), scale)),
        "quats": torch.tensor([[1.0, 0, 0, 0]]),
        "opacities": torch.logit(torch.tensor([opacity])),
        "sh0": rgb_to_sh(torch.tensor([rgb]))[:, None, :],
        "shN": torch.zeros(1, 0, 3),
    }


def camera(w=33, h=25, f=40.0):
    K = torch.tensor([[[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]])
    return torch.eye(4)[None], K, w, h


def test_single_gaussian_projects_to_the_right_pixel_with_right_depth():
    w2c, K, w, h = camera()
    g = one_gaussian([0.3, -0.2, 4.0])
    rgb, depth, alpha, info = render(g, w2c, K, w, h, backend="torch")
    assert rgb.shape == (1, h, w, 3) and depth.shape == (1, h, w) and alpha.shape == (1, h, w)
    u, v = 40 * 0.3 / 4 + w / 2, 40 * -0.2 / 4 + h / 2
    assert torch.allclose(info["means2d"][0, 0], torch.tensor([u, v]), atol=1e-4)
    peak = np.unravel_index(alpha[0].argmax().item(), (h, w))
    assert abs(peak[1] + 0.5 - u) <= 0.5 and abs(peak[0] + 0.5 - v) <= 0.5
    assert alpha.max().item() == pytest.approx(0.9, abs=0.02)
    assert depth[0][peak].item() == pytest.approx(4.0, abs=1e-4)
    assert rgb[0][peak][0] > 0.85 and rgb[0][peak][1:].abs().max() < 1e-5
    assert info["radii"].shape == (1, 1, 2) and (info["radii"] > 0).all()


def test_occlusion_order_and_behind_camera_culling():
    w2c, K, w, h = camera()
    near = one_gaussian([0.0, 0.0, 2.0], scale=0.3, opacity=0.99, rgb=(1, 0, 0))
    far = one_gaussian([0.0, 0.0, 5.0], scale=0.8, opacity=0.99, rgb=(0, 0, 1))
    both = {k: torch.cat([far[k], near[k]]) for k in near}  # far listed first on purpose
    rgb, depth, _, _ = render(both, w2c, K, w, h, backend="torch")
    centre = rgb[0, h // 2, w // 2]
    assert centre[0] > 0.95 and centre[2] < 0.05
    assert depth[0, h // 2, w // 2].item() == pytest.approx(2.0, abs=0.1)
    behind = one_gaussian([0.0, 0.0, -3.0])
    rgb, _, alpha, info = render(behind, w2c, K, w, h, backend="torch")
    assert alpha.abs().max() == 0 and (info["radii"] == 0).all()


def test_background_fills_empty_pixels():
    w2c, K, w, h = camera()
    g = one_gaussian([0.0, 0.0, 4.0], scale=0.05)
    rgb, _, _, _ = render(g, w2c, K, w, h, backend="torch", background=torch.tensor([[0.2, 0.4, 0.6]]))
    assert torch.allclose(rgb[0, 0, 0], torch.tensor([0.2, 0.4, 0.6]), atol=1e-5)


def test_gradients_reach_gaussians_and_camera_pose():
    w2c, K, w, h = camera()
    target_g = one_gaussian([0.3, 0.1, 4.0])
    with torch.no_grad():
        target, _, _, _ = render(target_g, w2c, K, w, h, backend="torch")
    # 1) move a misplaced Gaussian back by gradient descent
    g = {k: v.clone().requires_grad_(True) for k, v in one_gaussian([0.1, 0.0, 4.0]).items()}
    opt = torch.optim.Adam([g["means"]], lr=0.02)
    for _ in range(150):
        opt.zero_grad()
        out, _, _, _ = render(g, w2c, K, w, h, backend="torch")
        ((out - target) ** 2).mean().backward()
        opt.step()
    assert torch.allclose(g["means"][0, :2], torch.tensor([0.3, 0.1]), atol=0.03)
    # 2) recover a camera offset with the Gaussian fixed
    xi = torch.zeros(6, requires_grad=True)
    start = se3_exp(torch.tensor([0.15, -0.1, 0.0, 0.0, 0.0, 0.0])) @ torch.eye(4)
    opt = torch.optim.Adam([xi], lr=0.01)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        out, _, _, _ = render(target_g, (se3_exp(xi) @ start)[None], K, w, h, backend="torch")
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.02 * losses[0]


def test_quat_to_rotmat_is_orthonormal():
    R = quat_to_rotmat(torch.randn(10, 4))
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(10, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(10), atol=1e-5)
