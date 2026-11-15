import pytest
import torch

from fewview3d import losses as L
from fewview3d.metrics import image_metrics, psnr


def test_pearson_loss_is_invariant_to_scale_and_shift_of_the_prior():
    torch.manual_seed(0)
    depth = torch.rand(20, 30) * 4 + 1
    disp = 1.0 / depth
    assert L.pearson_depth_loss(depth, disp).item() == pytest.approx(0.0, abs=1e-5)
    assert L.pearson_depth_loss(depth, 7.3 * disp + 2.0).item() == pytest.approx(0.0, abs=1e-5)
    assert L.pearson_depth_loss(depth, -disp).item() == pytest.approx(2.0, abs=1e-5)
    assert L.pearson_depth_loss(depth, torch.rand(20, 30)).item() > 0.5


def test_pearson_loss_gradient_improves_depth():
    torch.manual_seed(0)
    target = torch.rand(16, 16) * 3 + 1
    depth = (target + torch.randn(16, 16) * 0.5).clamp(min=0.3).requires_grad_(True)
    opt = torch.optim.Adam([depth], lr=0.02)
    first = L.pearson_depth_loss(depth, 1 / target).item()
    for _ in range(200):
        opt.zero_grad()
        L.pearson_depth_loss(depth, 1 / target).backward()
        opt.step()
    assert L.pearson_depth_loss(depth, 1 / target).item() < 0.2 * first


@pytest.mark.parametrize("mode", ["scale", "scale_shift"])
def test_scale_shift_solver(mode):
    torch.manual_seed(1)
    src = torch.rand(500) * 5
    a_true, b_true = 2.5, (0.7 if mode == "scale_shift" else 0.0)
    a, b = L.solve_scale_shift(src, a_true * src + b_true, torch.ones(500), mode)
    assert a.item() == pytest.approx(a_true, rel=1e-4) and b.item() == pytest.approx(b_true, abs=1e-3)


def test_weighted_solver_ignores_zero_weight_outliers():
    src = torch.linspace(1, 5, 100)
    dst = 3 * src + 1
    dst[:30] = 100.0
    w = torch.ones(100)
    w[:30] = 0
    a, b = L.solve_scale_shift(src, dst, w)
    assert a.item() == pytest.approx(3.0, rel=1e-4) and b.item() == pytest.approx(1.0, abs=1e-3)


def test_confidence_weights_modes():
    conf = torch.linspace(1, 11, 101).reshape(1, 101)
    like = torch.zeros(1, 101)
    assert L.confidence_weights(conf, like, "none").sum() == 101
    assert L.confidence_weights(None, like, "soft").sum() == 101
    thr = L.confidence_weights(conf, like, "threshold", lo_q=0.2)
    assert thr.sum().item() == 80
    soft = L.confidence_weights(conf, like, "soft", 0.2, 0.8)
    assert soft.min() == 0 and soft.max() == 1
    assert soft[0, 50].item() == pytest.approx(0.5, abs=1e-5)
    assert (soft[0, 1:] >= soft[0, :-1]).all()


def test_confidence_depth_loss_alignment_and_weighting():
    torch.manual_seed(2)
    depth = torch.rand(24, 32) * 4 + 1
    ones = torch.ones_like(depth)
    assert L.confidence_depth_loss(depth, depth, ones, "none").item() == pytest.approx(0, abs=1e-6)
    # a prior in another scale is only consistent once aligned
    assert L.confidence_depth_loss(depth, 0.4 * depth, ones, "none").item() > 0.3
    assert L.confidence_depth_loss(depth, 0.4 * depth, ones, "scale").item() == pytest.approx(0, abs=1e-5)
    assert L.confidence_depth_loss(depth, 0.4 * depth - 0.2, ones, "scale_shift").item() == pytest.approx(0, abs=1e-5)
    # a corrupted region hurts unless its confidence weight is zero
    bad = depth.clone()
    bad[:12, :16] *= 0.5
    w = ones.clone()
    w[:12, :16] = 0
    assert L.confidence_depth_loss(depth, bad, ones, "none").item() > 0.05
    assert L.confidence_depth_loss(depth, bad, w, "none").item() == pytest.approx(0, abs=1e-6)
    # invalid (zero) prior depth is ignored, and an all-zero weight gives zero loss with a valid graph
    bad[:12, :16] = 0
    assert L.confidence_depth_loss(depth, bad, ones, "none").item() == pytest.approx(0, abs=1e-6)
    d = depth.clone().requires_grad_(True)
    L.confidence_depth_loss(d, depth, torch.zeros_like(depth)).backward()
    assert d.grad.abs().sum() == 0


def test_ssim_and_psnr_reference_values():
    torch.manual_seed(3)
    img = torch.rand(40, 40, 3)
    assert L.ssim(img[None], img[None]).item() == pytest.approx(1.0, abs=1e-5)
    noisy = (img + 0.1 * torch.randn_like(img)).clamp(0, 1)
    assert 0.2 < L.ssim(img[None], noisy[None]).item() < 0.99
    assert psnr(img + 0.1, img) == pytest.approx(20.0, abs=0.3)  # clamped to [0, 1]
    assert psnr(torch.full((4, 4, 3), 0.5), torch.full((4, 4, 3), 0.4)) == pytest.approx(20.0, abs=1e-3)
    m = image_metrics(noisy, img, with_lpips=False)
    assert set(m) == {"psnr", "ssim", "lpips"} and m["lpips"] is None
