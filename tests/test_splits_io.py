import numpy as np
import pytest
from PIL import Image

from fewview3d import colmap_io as cio
from fewview3d.dataset import load_colmap_scene
from fewview3d.geometry import rotmat_to_qvec
from fewview3d.splits import load_split, make_split, save_split


def test_llff_style_split():
    s = make_split(20, 3)
    assert s["test"] == [0, 8, 16]
    assert len(s["train"]) == 3 and not set(s["train"]) & set(s["test"])
    assert s["train"][0] == 1 and s["train"][-1] == 19  # evenly spaced over the non-test images
    assert make_split(20, 3) == s  # deterministic


def test_full_split_and_errors():
    s = make_split(20, 0)
    assert len(s["train"]) == 17
    with pytest.raises(ValueError):
        make_split(1, 3)


def test_split_roundtrip_by_name(tmp_path):
    names = [f"img{i:02d}.png" for i in range(20)]
    s = make_split(20, 6)
    save_split(tmp_path / "s.json", names, s)
    assert load_split(tmp_path / "s.json", names)["train"] == s["train"]
    with pytest.raises(KeyError):
        load_split(tmp_path / "s.json", names[:5])


def _write_scene(root, n=6, w=40, h=30, model="PINHOLE"):
    rng = np.random.default_rng(0)
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    params = np.array([35.0, 36.0, w / 2, h / 2]) if model == "PINHOLE" else np.array([35.0, w / 2, h / 2])
    cams = {1: cio.ColmapCamera(1, model, w, h, params)}
    images, Rs, ts = {}, [], []
    for i in range(n):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = rng.normal(size=3)
        Rs.append(Q)
        ts.append(t)
        # written in reverse order to check that loading sorts by name
        name = f"im_{n - 1 - i:02d}.png"
        ids = np.array([10, 11, -1]) if i < 2 else np.array([11, 12])
        images[i + 1] = cio.ColmapImage(i + 1, rotmat_to_qvec(Q), t, 1, name, ids)
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(root / "images" / name)
    cio.write_cameras_bin(root / "sparse/0/cameras.bin", cams)
    cio.write_images_bin(root / "sparse/0/images.bin", images)
    xyz = rng.normal(size=(3, 3))
    cio.write_points3D_bin(root / "sparse/0/points3D.bin", [10, 11, 12], xyz, [[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    return Rs, ts, xyz


def test_colmap_roundtrip_and_scene_loading(tmp_path):
    Rs, ts, xyz = _write_scene(tmp_path)
    scene = load_colmap_scene(tmp_path)
    assert scene.names == sorted(scene.names) and len(scene) == 6
    assert scene.images.shape == (6, 30, 40, 3)
    # names were written reversed, so sorted index 5 is COLMAP image 1
    assert np.allclose(scene.w2c[5][:3, :3], Rs[0], atol=1e-9)
    assert np.allclose(scene.w2c[5][:3, 3], ts[0])
    assert np.allclose(scene.K[0], [[35, 0, 20], [0, 36, 15], [0, 0, 1]])
    assert np.allclose(scene.points, xyz) and scene.points_rgb[1].tolist() == [0, 255, 0]
    # point 10 is seen only by COLMAP images 1 and 2 = sorted indices 5 and 4
    assert sorted(scene.point_views[0].tolist()) == [4, 5]
    assert scene.points_seen_by([4, 5]).tolist() == [True, True, False]
    assert scene.points_seen_by([0, 1]).tolist() == [False, True, True]
    assert scene.points_seen_by([0, 5]).tolist() == [False, True, False]


def test_downsampling_scales_intrinsics(tmp_path):
    _write_scene(tmp_path, model="SIMPLE_PINHOLE")
    scene = load_colmap_scene(tmp_path, factor=2)
    assert scene.images.shape == (6, 15, 20, 3)
    assert np.allclose(scene.K[0], [[17.5, 0, 10], [0, 17.5, 7.5], [0, 0, 1]])
