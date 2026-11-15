"""Scene loading for COLMAP-format datasets (LLFF, Mip-NeRF 360, DTU exports, own captures).

Expected layout, the one shipped by ``nerf_llff_data`` and ``360_v2``::

    <scene>/
        images/            full-resolution images
        images_<f>/        optional pre-downsampled copies (factor f)
        sparse/0/          cameras.bin, images.bin, points3D.bin
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import colmap_io
from .geometry import qvec_to_rotmat, scale_intrinsics

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class SceneData:
    """All images and ground-truth (full-view COLMAP) cameras of one scene.

    ``images`` is uint8 (N, H, W, 3). ``point_views[j]`` lists the image indices
    in which SfM point ``j`` was observed; it is what lets us build an
    initialisation that does not use points seen only by held-out views.
    """

    name: str
    names: list[str]
    images: np.ndarray
    K: np.ndarray  # (N, 3, 3)
    w2c: np.ndarray  # (N, 4, 4)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    points_rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))
    point_views: list[np.ndarray] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.images.shape[1])

    @property
    def width(self) -> int:
        return int(self.images.shape[2])

    def __len__(self) -> int:
        return len(self.names)

    def points_seen_by(self, view_idx: list[int], min_views: int = 2) -> np.ndarray:
        """Boolean mask over SfM points observed in at least ``min_views`` of ``view_idx``."""
        chosen = set(int(i) for i in view_idx)
        mask = np.zeros(len(self.points), dtype=bool)
        for j, views in enumerate(self.point_views):
            mask[j] = sum(1 for v in views if int(v) in chosen) >= min_views
        return mask


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix in IMAGE_EXTS)


def load_colmap_scene(root: str | Path, factor: int = 1, name: str | None = None) -> SceneData:
    """Load a COLMAP scene, using ``images_<factor>`` if present, else resizing on the fly."""
    root = Path(root)
    sparse = root / "sparse" / "0"
    if not sparse.exists():
        sparse = root / "sparse"
    cams = colmap_io.read_cameras_bin(sparse / "cameras.bin")
    ims = colmap_io.read_images_bin(sparse / "images.bin")
    pt_ids, xyz, rgb, _ = colmap_io.read_points3D_bin(sparse / "points3D.bin")

    records = sorted(ims.values(), key=lambda im: im.name)
    full_dir = root / "images"
    small_dir = root / f"images_{factor}"
    use_small = factor > 1 and small_dir.exists()
    img_dir = small_dir if use_small else full_dir
    files = _list_images(img_dir)

    # LLFF's pre-downsampled folders rename the files (image000.png, ...), so the
    # i-th sorted file corresponds to the i-th sorted COLMAP image name.
    by_name = {p.name: p for p in files}
    if all(r.name in by_name for r in records):
        paths = [by_name[r.name] for r in records]
    elif len(files) == len(records):
        paths = files
    else:
        raise FileNotFoundError(
            f"{img_dir} has {len(files)} images but COLMAP registered {len(records)}; cannot match them"
        )

    images, Ks, w2cs = [], [], []
    for rec, path in zip(records, paths):
        cam = cams[rec.camera_id]
        img = Image.open(path).convert("RGB")
        K = cam.K()
        dist = cam.distortion()
        if np.abs(dist).max() > 1e-6:
            img, K = _undistort(img, K, dist, cam.width, cam.height)
        if factor > 1 and not use_small:
            new_size = (round(cam.width / factor), round(cam.height / factor))
            img = img.resize(new_size, Image.LANCZOS)
        K = scale_intrinsics(K, img.width / cam.width, img.height / cam.height)
        w2c = np.eye(4)
        w2c[:3, :3] = qvec_to_rotmat(rec.qvec)
        w2c[:3, 3] = rec.tvec
        images.append(np.asarray(img, dtype=np.uint8))
        Ks.append(K)
        w2cs.append(w2c)

    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise ValueError(f"images have different sizes {shapes}; resize the scene to a common size first")

    # Per-point visibility: which images observed each SfM point.
    id_to_row = {int(pid): j for j, pid in enumerate(pt_ids)}
    views: list[list[int]] = [[] for _ in range(len(pt_ids))]
    for i, rec in enumerate(records):
        for pid in np.unique(rec.point3D_ids):
            j = id_to_row.get(int(pid))
            if j is not None:
                views[j].append(i)

    return SceneData(
        name=name or root.name,
        names=[r.name for r in records],
        images=np.stack(images),
        K=np.stack(Ks),
        w2c=np.stack(w2cs),
        points=xyz,
        points_rgb=rgb,
        point_views=[np.array(v, dtype=np.int64) for v in views],
        image_paths=[str(p) for p in paths],
    )


def _undistort(img: Image.Image, K: np.ndarray, dist: np.ndarray, width: int, height: int):
    """Remove lens distortion so that the pinhole model used by the renderer is exact."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - depends on the environment
        warnings.warn("opencv-python is not installed; lens distortion is ignored")
        return img, K
    arr = np.asarray(img)
    sx, sy = arr.shape[1] / width, arr.shape[0] / height
    K_img = scale_intrinsics(K, sx, sy)
    map1, map2 = cv2.initUndistortRectifyMap(
        K_img, dist, None, K_img, (arr.shape[1], arr.shape[0]), cv2.CV_32FC1
    )
    out = cv2.remap(arr, map1, map2, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(out), K
