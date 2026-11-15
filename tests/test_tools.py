"""Tests for the command-line tools, the aggregation tables and the prior runners (no GPU needed)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fewview3d import colmap_io as cio
from fewview3d.aggregate import collect, leakage_table, prior_gain_rows, report, seed_mean_std
from fewview3d.geometry import rotmat_to_qvec
from fewview3d.priors import mv_prior_path, save_mv_prior
from fewview3d.priors.vggt_runner import predictions_to_prior, preprocess, vggt_input_size
from fewview3d.synthetic import fake_mono_disparity, fake_multiview_prior, make_synthetic_scene

ROOT = Path(__file__).resolve().parents[1]


def test_vggt_input_size_and_preprocess():
    assert vggt_input_size(504, 378) == (518, 392)  # LLFF at 1/8 resolution
    assert vggt_input_size(378, 504) == (392, 518)
    for w, h in [(504, 378), (1297, 840), (640, 640), (300, 900)]:
        ww, hh = vggt_input_size(w, h)
        assert ww % 14 == 0 and hh % 14 == 0 and max(ww, hh) == 518
    x = preprocess(np.random.default_rng(0).integers(0, 255, (2, 378, 504, 3), dtype=np.uint8))
    assert x.shape == (2, 3, 392, 518) and 0 <= x.min() and x.max() <= 1


def test_predictions_to_prior_builds_4x4_poses():
    extr = np.tile(np.eye(4)[:3], (3, 1, 1))
    extr[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 2, 0]]
    prior = predictions_to_prior(["a", "b", "c"], extr, np.tile(np.eye(3), (3, 1, 1)),
                                 np.ones((3, 4, 6)), np.ones((3, 4, 6)))
    assert prior.w2c.shape == (3, 4, 4) and np.allclose(prior.w2c[:, 3], [0, 0, 0, 1])
    assert prior.proc_hw == (4, 6)
    with pytest.raises(ValueError):
        predictions_to_prior(["a"], extr, np.eye(3)[None], np.ones((3, 4, 6)), np.ones((3, 4, 6)))


def _fake_result(root, dataset, scene, pose, prior, seed, psnr):
    run = f"pose-{pose}_prior-{prior}_3v_seed{seed}"
    d = root / "outputs" / dataset / scene / run
    d.mkdir(parents=True)
    res = {"scene": scene, "config": {"n_views": 3, "pose_source": pose, "depth_prior": prior, "seed": seed},
           "stats": {"n_gaussians": 10}, "metrics": {"psnr": psnr, "ssim": 0.5, "lpips": None},
           "diagnostics": {}}
    (d / "results.json").write_text(json.dumps(res))


def test_aggregation_tables(tmp_path):
    # made-up numbers, chosen so that every derived quantity is easy to check by hand
    values = {("fern", "colmap", "none"): 20, ("fern", "colmap", "mono"): 22, ("fern", "vggt", "none"): 17,
              ("fern", "vggt", "mono"): 18, ("room", "colmap", "none"): 24, ("room", "colmap", "mono"): 23,
              ("room", "vggt", "none"): 20, ("room", "vggt", "mono"): 19}
    for (scene, pose, prior), v in values.items():
        for seed in (0, 1):
            _fake_result(tmp_path, "llff", scene, pose, prior, seed, v + seed)
    rows = collect(tmp_path / "outputs")
    assert len(rows) == 16
    cell = [r for r in rows if r["pose"] == "colmap" and r["prior"] == "none"]
    mean, std, n = seed_mean_std(cell, "psnr")
    assert mean == pytest.approx(22.5) and std == pytest.approx(0.5) and n == 2
    gains = {(g["scene"], g["pose"]): g["gain"] for g in prior_gain_rows(rows)}
    assert gains[("fern", "colmap")] == pytest.approx(2) and gains[("room", "vggt")] == pytest.approx(-1)
    assert "| llff | 3 | none | 22.50 | 19.00 | +3.50 |" in leakage_table(rows)
    text = report(rows)
    assert "| llff | 3 | colmap | mono | textureless | -1.00 | 1 | 1 |" in text
    assert "| llff | 3 | colmap | mono | foliage | +2.00 | 1 | 0 |" in text


def _export_colmap(scene, root):
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    K = scene.K[0]
    cams = {1: cio.ColmapCamera(1, "PINHOLE", scene.width, scene.height, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))}
    ids_per_view = [[] for _ in scene.names]
    for j, views in enumerate(scene.point_views):
        for v in views:
            ids_per_view[v].append(j + 1)
    images = {}
    for i, name in enumerate(scene.names):
        Image.fromarray(scene.images[i]).save(root / "images" / name)
        images[i + 1] = cio.ColmapImage(i + 1, rotmat_to_qvec(scene.w2c[i][:3, :3]), scene.w2c[i][:3, 3], 1, name,
                                        np.array(ids_per_view[i], dtype=np.int64))
    cio.write_cameras_bin(root / "sparse/0/cameras.bin", cams)
    cio.write_images_bin(root / "sparse/0/images.bin", images)
    cio.write_points3D_bin(root / "sparse/0/points3D.bin", np.arange(len(scene.points)) + 1, scene.points, scene.points_rgb)


def _run(script, *args):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def test_scripts_end_to_end_on_a_colmap_export(tmp_path):
    """make_splits -> train (both protocols) -> aggregate, through the real command-line entry points."""
    scene, depth, _ = make_synthetic_scene()
    data, work = tmp_path / "data", tmp_path / "work"
    _export_colmap(scene, data / "llff" / "fern")
    common = ["--data-root", data, "--work-root", work, "--dataset", "llff", "--scenes", "fern", "--factor", 1]
    out = _run("make_splits.py", *common, "--views", 3)
    assert "3 train / 3 test of 17 images" in out
    split = json.loads((work / "splits/llff/fern/split_3views.json").read_text())

    # stand-ins for the GPU prior scripts
    pdir = work / "priors/llff/fern"
    save_mv_prior(mv_prior_path(pdir, 3), fake_multiview_prior(scene, depth, split["train"]))
    (pdir / "mono").mkdir()
    for i in split["train"]:
        np.save(pdir / "mono" / (Path(scene.names[i]).stem + ".npy"), fake_mono_disparity(depth[i]))

    fast = ["--max-steps", 60, "--no-with-lpips", "--pose-refine-steps", 20, "--max-init-points", 300,
            "--device", "cpu"]
    _run("train.py", *common, "--n-views", 3, *fast)
    _run("run_sweep.py", *common, "--views", 3, "--seeds", 0, "--cells", "colmap:mono", "vggt:vggt", *fast)
    again = _run("run_sweep.py", *common, "--views", 3, "--seeds", 0, "--cells", "colmap:mono", "vggt:vggt", *fast)
    assert again.count("(done)") == 2  # resumable
    runs = sorted(p.parent.name for p in (work / "outputs/llff/fern").glob("*/results.json"))
    assert runs == ["pose-colmap_prior-mono_3v_seed0", "pose-colmap_prior-none_3v_seed0",
                    "pose-vggt_prior-vggt_3v_seed0"]
    assert (work / "outputs/llff/fern" / runs[0] / "renders").exists()
    table = _run("aggregate.py", "--work-root", work, "--out", work / "results")
    assert "## Main table" in table and (work / "results/results.csv").exists()
