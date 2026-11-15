"""Minimal reader (and writer, for tests) for COLMAP binary sparse models.

Only what the experiments need: camera intrinsics, image poses with their
2D-3D track ids, and the 3D points with colours.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# model_id -> (name, number of parameters)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
}
MODEL_IDS = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}


@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def K(self) -> np.ndarray:
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def distortion(self) -> np.ndarray:
        """OpenCV-style (k1, k2, p1, p2) coefficients; zeros for pinhole models."""
        p = self.params
        if self.model == "SIMPLE_RADIAL":
            return np.array([p[3], 0, 0, 0], dtype=np.float64)
        if self.model == "RADIAL":
            return np.array([p[3], p[4], 0, 0], dtype=np.float64)
        if self.model == "OPENCV":
            return np.array(p[4:8], dtype=np.float64)
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return np.zeros(4)
        raise NotImplementedError(f"camera model {self.model} is not supported")


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray  # (w, x, y, z), world-to-camera
    tvec: np.ndarray
    camera_id: int
    name: str
    point3D_ids: np.ndarray  # one per 2D keypoint, -1 if not triangulated


def _read(f, fmt: str):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path: str | Path) -> dict[int, ColmapCamera]:
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cam_id, model_id, w, h = _read(f, "iiQQ")
            name, n_params = CAMERA_MODELS[model_id]
            params = np.array(_read(f, "d" * n_params))
            cams[cam_id] = ColmapCamera(cam_id, name, int(w), int(h), params)
    return cams


def read_images_bin(path: str | Path) -> dict[int, ColmapImage]:
    images = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            vals = _read(f, "idddddddi")
            image_id, qvec, tvec, cam_id = vals[0], np.array(vals[1:5]), np.array(vals[5:8]), vals[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n_pts,) = _read(f, "Q")
            raw = np.frombuffer(f.read(24 * n_pts), dtype=np.dtype([("xy", "<f8", 2), ("id", "<i8")]))
            images[image_id] = ColmapImage(
                image_id, qvec, tvec, cam_id, name.decode("utf-8"), raw["id"].astype(np.int64)
            )
    return images


def read_points3D_bin(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids [M], xyz [M,3], rgb uint8 [M,3], reprojection error [M])."""
    ids, xyz, rgb, err = [], [], [], []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            vals = _read(f, "QdddBBBd")
            ids.append(vals[0])
            xyz.append(vals[1:4])
            rgb.append(vals[4:7])
            err.append(vals[7])
            (track_len,) = _read(f, "Q")
            f.seek(8 * track_len, 1)
    return (
        np.array(ids, dtype=np.int64),
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        np.array(err, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Writers (used by the tests and the synthetic-scene exporter)
# --------------------------------------------------------------------------- #
def write_cameras_bin(path: str | Path, cams: dict[int, ColmapCamera]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cam in cams.values():
            f.write(struct.pack("<iiQQ", cam.id, MODEL_IDS[cam.model], cam.width, cam.height))
            f.write(struct.pack("<" + "d" * len(cam.params), *cam.params))


def write_images_bin(path: str | Path, images: dict[int, ColmapImage]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images.values():
            f.write(struct.pack("<idddddddi", im.id, *im.qvec, *im.tvec, im.camera_id))
            f.write(im.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(im.point3D_ids)))
            for pid in im.point3D_ids:
                f.write(struct.pack("<ddq", 0.0, 0.0, int(pid)))


def write_points3D_bin(path: str | Path, ids, xyz, rgb) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(ids)))
        for pid, p, c in zip(ids, xyz, rgb):
            f.write(struct.pack("<QdddBBBd", int(pid), *map(float, p), *map(int, c), 0.0))
            f.write(struct.pack("<Q", 0))
