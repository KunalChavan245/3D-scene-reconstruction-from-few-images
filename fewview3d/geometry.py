"""Camera geometry helpers.

Conventions used everywhere in this repo:
  * ``w2c`` is a 4x4 world-to-camera matrix, OpenCV axes (x right, y down, z forward).
  * ``c2w`` is its inverse; the camera centre is ``c2w[:3, 3]``.
  * ``K`` is a 3x3 pinhole intrinsics matrix in pixels.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# SE(3) / rotations
# --------------------------------------------------------------------------- #
def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Exponential map from a 6-vector twist (tx, ty, tz, rx, ry, rz) to a 4x4 matrix.

    Differentiable. Used to parameterise small camera-pose corrections.
    """
    v, w = xi[..., :3], xi[..., 3:]
    zeros = torch.zeros_like(w[..., 0])
    row0 = torch.stack([zeros, -w[..., 2], w[..., 1], v[..., 0]], dim=-1)
    row1 = torch.stack([w[..., 2], zeros, -w[..., 0], v[..., 1]], dim=-1)
    row2 = torch.stack([-w[..., 1], w[..., 0], zeros, v[..., 2]], dim=-1)
    row3 = torch.zeros_like(row0)
    twist = torch.stack([row0, row1, row2, row3], dim=-2)
    return torch.linalg.matrix_exp(twist)


def invert_se3(mat: np.ndarray) -> np.ndarray:
    """Invert rigid transforms of shape (..., 4, 4)."""
    mat = np.asarray(mat, dtype=np.float64)
    out = np.zeros_like(mat)
    R = mat[..., :3, :3]
    t = mat[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) to a rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to COLMAP quaternion (w, x, y, z)."""
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ]
    ) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    q[1:] = -q[1:]
    if q[0] < 0:
        q = -q
    return q


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Closest rotation matrix (Frobenius norm) to a 3x3 matrix."""
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# --------------------------------------------------------------------------- #
# Sim(3) alignment between two camera sets
# --------------------------------------------------------------------------- #
def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``dst ~ s * R @ src + t`` from point pairs (N, 3).

    Needs at least 3 non-collinear points to determine the rotation.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / max(var_s, 1e-20))
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_cameras_sim3(
    src_c2w: np.ndarray, dst_c2w: np.ndarray, method: str = "rotation"
) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) mapping the ``src`` world frame onto the ``dst`` world frame.

    Given the same N cameras expressed in two world frames, find the transform
    such that ``dst_centre ~ s * R @ src_centre + t`` and ``dst_R ~ R @ src_R``.

    ``method="rotation"`` (default) averages the per-camera relative rotations and
    takes the scale from the spread of the camera centres. It stays well-posed for
    two cameras and for collinear camera centres, both of which occur with 3
    forward-facing input views. ``method="umeyama"`` uses the camera centres only.
    """
    src_c2w = np.asarray(src_c2w, dtype=np.float64)
    dst_c2w = np.asarray(dst_c2w, dtype=np.float64)
    if src_c2w.shape != dst_c2w.shape or src_c2w.shape[0] < 2:
        raise ValueError("need the same N >= 2 cameras in both frames")
    cs, cd = src_c2w[:, :3, 3], dst_c2w[:, :3, 3]
    if method == "umeyama":
        if len(cs) < 3:
            raise ValueError("umeyama needs at least 3 cameras")
        return umeyama(cs, cd)
    if method != "rotation":
        raise ValueError(f"unknown method {method!r}")
    M = np.zeros((3, 3))
    for Rs, Rd in zip(src_c2w[:, :3, :3], dst_c2w[:, :3, :3]):
        M += Rd @ Rs.T
    R = project_to_so3(M)
    mu_s, mu_d = cs.mean(0), cd.mean(0)
    spread_s = np.sqrt(((cs - mu_s) ** 2).sum())
    spread_d = np.sqrt(((cd - mu_d) ** 2).sum())
    if spread_s < 1e-12:
        raise ValueError("source camera centres coincide; scale is undetermined")
    s = float(spread_d / spread_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3_to_c2w(c2w: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Move camera-to-world matrices (N, 4, 4) into the frame ``x' = s R x + t``."""
    c2w = np.asarray(c2w, dtype=np.float64)
    out = np.tile(np.eye(4), (len(c2w), 1, 1))
    out[:, :3, :3] = np.einsum("ij,njk->nik", R, c2w[:, :3, :3])
    out[:, :3, 3] = s * np.einsum("ij,nj->ni", R, c2w[:, :3, 3]) + t
    return out


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between rotation matrices (..., 3, 3)."""
    rel = np.einsum("...ij,...kj->...ik", Ra, Rb)
    cos = (np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def pose_errors(est_c2w: np.ndarray, gt_c2w: np.ndarray, method: str = "rotation") -> dict:
    """Pose accuracy of ``est`` against ``gt`` after Sim(3) alignment of gt onto est.

    The translation error is reported relative to the spread of the camera
    centres, so it is unit-free and comparable across scenes.
    """
    s, R, t = align_cameras_sim3(gt_c2w, est_c2w, method)
    gt_in_est = apply_sim3_to_c2w(gt_c2w, s, R, t)
    rot = rotation_error_deg(est_c2w[:, :3, :3], gt_in_est[:, :3, :3])
    centres = est_c2w[:, :3, 3]
    extent = np.linalg.norm(centres - centres.mean(0), axis=1).max()
    trans = np.linalg.norm(centres - gt_in_est[:, :3, 3], axis=1) / max(extent, 1e-12)
    return {
        "rot_err_deg_mean": float(rot.mean()),
        "rot_err_deg_max": float(rot.max()),
        "rel_centre_err_mean": float(trans.mean()),
        "rel_centre_err_max": float(trans.max()),
        "sim3_scale": float(s),
    }


# --------------------------------------------------------------------------- #
# Depth <-> points
# --------------------------------------------------------------------------- #
def unproject_depth(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Lift a z-depth map (H, W) to world points (H, W, 3). Pixel centres at +0.5."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], axis=-1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Rescale intrinsics when an image is resized by (sx, sy)."""
    K = np.array(K, dtype=np.float64, copy=True)
    K[..., 0, :] *= sx
    K[..., 1, :] *= sy
    return K


def camera_extent(c2w: np.ndarray) -> float:
    """Largest distance of a camera centre from the mean centre."""
    c = np.asarray(c2w)[:, :3, 3]
    return float(np.linalg.norm(c - c.mean(0), axis=1).max())
