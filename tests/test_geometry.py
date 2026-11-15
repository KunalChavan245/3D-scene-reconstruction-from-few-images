import numpy as np
import pytest
import torch

from fewview3d import geometry as g


def random_c2w(rng, n):
    out = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        out[i, :3, :3] = Q
        out[i, :3, 3] = rng.normal(size=3)
    return out


def random_sim3(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return float(rng.uniform(0.2, 5)), Q, rng.normal(size=3)


@pytest.mark.parametrize("method,n", [("rotation", 2), ("rotation", 3), ("rotation", 9), ("umeyama", 4)])
def test_sim3_recovers_known_transform(method, n):
    rng = np.random.default_rng(n)
    src = random_c2w(rng, n)
    s, R, t = random_sim3(rng)
    dst = g.apply_sim3_to_c2w(src, s, R, t)
    s2, R2, t2 = g.align_cameras_sim3(src, dst, method)
    assert s2 == pytest.approx(s, rel=1e-8)
    assert np.allclose(R2, R, atol=1e-8)
    assert np.allclose(t2, t, atol=1e-7)


def test_sim3_rotation_method_handles_collinear_centres():
    rng = np.random.default_rng(0)
    src = random_c2w(rng, 3)
    src[:, :3, 3] = np.outer([0.0, 1.0, 2.5], [1.0, 0.2, -0.3])  # three centres on one line
    s, R, t = random_sim3(rng)
    dst = g.apply_sim3_to_c2w(src, s, R, t)
    s2, R2, t2 = g.align_cameras_sim3(src, dst)
    assert s2 == pytest.approx(s) and np.allclose(R2, R, atol=1e-8) and np.allclose(t2, t, atol=1e-7)


def test_pose_errors_zero_for_similar_frames_and_positive_for_noise():
    rng = np.random.default_rng(1)
    gt = random_c2w(rng, 5)
    est = g.apply_sim3_to_c2w(gt, *random_sim3(rng))
    err = g.pose_errors(est, gt)
    assert err["rot_err_deg_max"] < 1e-5 and err["rel_centre_err_max"] < 1e-8
    est[0, :3, 3] += 0.3
    assert g.pose_errors(est, gt)["rel_centre_err_max"] > 1e-3


def test_se3_exp_is_rigid_and_matches_small_angle():
    xi = torch.tensor([0.1, -0.2, 0.3, 0.02, -0.01, 0.03], dtype=torch.float64)
    T = g.se3_exp(xi)
    R = T[:3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-10)
    assert torch.det(R).item() == pytest.approx(1.0)
    assert torch.allclose(g.se3_exp(torch.zeros(6)), torch.eye(4))
    assert torch.allclose(T[:3, 3], xi[:3], atol=5e-3)


def test_qvec_roundtrip_and_inverse():
    rng = np.random.default_rng(2)
    c2w = random_c2w(rng, 4)
    for M in c2w:
        assert np.allclose(g.qvec_to_rotmat(g.rotmat_to_qvec(M[:3, :3])), M[:3, :3], atol=1e-9)
    assert np.allclose(g.invert_se3(c2w) @ c2w, np.tile(np.eye(4), (4, 1, 1)), atol=1e-10)


def test_unproject_then_project_roundtrip():
    rng = np.random.default_rng(3)
    c2w = random_c2w(rng, 1)[0]
    K = np.array([[50.0, 0, 16], [0, 55.0, 12], [0, 0, 1]])
    depth = rng.uniform(1, 5, size=(24, 32))
    pts = g.unproject_depth(depth, K, c2w).reshape(-1, 3)
    w2c = g.invert_se3(c2w)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    assert np.allclose(cam[:, 2], depth.ravel())
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    assert np.allclose(u.reshape(24, 32)[0], np.arange(32) + 0.5)
