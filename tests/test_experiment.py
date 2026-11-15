"""End-to-end tests on the synthetic scene with the CPU reference rasteriser."""

import json

import numpy as np
import pytest
import torch

from fewview3d.experiment import ExperimentConfig, build_protocol, resolve_depth_align, run_experiment
from fewview3d.geometry import invert_se3
from fewview3d.priors import load_mv_prior, mv_prior_path, save_mv_prior
from fewview3d.splits import make_split
from fewview3d.synthetic import fake_mono_disparity, fake_multiview_prior, make_synthetic_scene

FAST = dict(max_steps=150, refine_start=50, refine_every=50, reset_every=10_000, with_lpips=False,
            save_renders=False, pose_refine_steps=60, max_init_points=400)


@pytest.fixture(scope="module")
def world():
    scene, depth, _ = make_synthetic_scene()
    return scene, depth, make_split(len(scene), 3)


def write_priors(tmp_path, scene, depth, split, **kw):
    tr = split["train"]
    save_mv_prior(mv_prior_path(tmp_path, len(tr)), fake_multiview_prior(scene, depth, tr, **kw))
    (tmp_path / "mono").mkdir(exist_ok=True)
    for i in tr:
        np.save(tmp_path / "mono" / (scene.names[i].rsplit(".", 1)[0] + ".npy"), fake_mono_disparity(depth[i]))
    return tmp_path


def test_colmap_protocol_trains_and_writes_results(world, tmp_path):
    scene, _, split = world
    res = run_experiment(scene, split, ExperimentConfig(**FAST), out_dir=tmp_path / "run", log=None)
    assert res["metrics"]["psnr"] > 22
    assert res["pose_refined"] is False and res["diagnostics"]["init"] == "sfm_train"
    assert res["diagnostics"]["n_init_points"] < res["diagnostics"]["sfm_points_total"]
    saved = json.loads((tmp_path / "run" / "results.json").read_text())
    assert saved["train_views"] == [scene.names[i] for i in split["train"]]
    assert len(saved["per_image"]) == len(split["test"])
    assert (tmp_path / "run" / "gaussians.pt").exists()


def test_training_improves_over_initialisation(world):
    scene, _, split = world
    short = run_experiment(scene, split, ExperimentConfig(**{**FAST, "max_steps": 1}), log=None)
    full = run_experiment(scene, split, ExperimentConfig(**FAST), log=None)
    assert full["metrics"]["psnr"] > short["metrics"]["psnr"] + 5


def test_densification_changes_the_number_of_gaussians(world):
    scene, _, split = world
    cfg = ExperimentConfig(**{**FAST, "grow_grad2d": 1e-6, "init": "random", "n_random_points": 300,
                              "refine_start": 20, "refine_every": 25, "refine_stop_frac": 0.9})
    res = run_experiment(scene, split, cfg, log=None)
    assert res["stats"]["n_gaussians"] != res["diagnostics"]["n_init_points"]


@pytest.mark.parametrize("prior", ["mono", "vggt"])
def test_depth_priors_with_colmap_poses(world, tmp_path, prior):
    scene, depth, split = world
    pdir = write_priors(tmp_path, scene, depth, split)
    cfg = ExperimentConfig(depth_prior=prior, **FAST)
    assert resolve_depth_align(cfg) == "scale_shift"  # the prior is in another frame than COLMAP
    res = run_experiment(scene, split, cfg, prior_dir=pdir, log=None)
    assert res["stats"]["final_loss_depth"] is not None and res["stats"]["final_loss_depth"] < 0.2
    assert res["metrics"]["psnr"] > 22


def test_honest_protocol_end_to_end(world, tmp_path):
    scene, depth, split = world
    pdir = write_priors(tmp_path, scene, depth, split, scale=0.37, proc_scale=0.75)
    cfg = ExperimentConfig(pose_source="vggt", depth_prior="vggt", **FAST)
    assert resolve_depth_align(cfg) == "none"
    res = run_experiment(scene, split, cfg, prior_dir=pdir, log=None)
    d = res["diagnostics"]
    # the simulated estimate is exact up to a similarity, so the diagnostics must say so
    assert d["train_pose_error"]["rot_err_deg_max"] < 1e-3
    assert d["train_pose_error"]["rel_centre_err_max"] < 1e-5
    assert d["train_pose_error"]["sim3_scale"] == pytest.approx(0.37, rel=1e-4)
    assert d["focal_rel_error"] < 1e-6 and d["init"] == "prior"
    assert res["pose_refined"] is True
    assert res["metrics"]["psnr"] > 22
    assert res["metrics"]["psnr"] >= res["metrics"]["psnr_before_refine"] - 0.5


def test_test_cameras_are_mapped_into_the_estimated_frame(world, tmp_path):
    scene, depth, split = world
    pdir = write_priors(tmp_path, scene, depth, split, scale=2.0)
    cfg = ExperimentConfig(pose_source="vggt", **FAST)
    train, test, (xyz, _), _, _ = build_protocol(scene, split, cfg, pdir, torch.device("cpu"))
    # distances between cameras are doubled, and the initial points sit in front of the cameras
    c_gt = invert_se3(scene.w2c[split["test"]])[:, :3, 3]
    c_est = invert_se3(test.w2c.numpy())[:, :3, 3]
    assert np.linalg.norm(c_est[0] - c_est[1]) == pytest.approx(2 * np.linalg.norm(c_gt[0] - c_gt[1]), rel=1e-4)
    cam = xyz @ train.w2c[0, :3, :3].numpy().T + train.w2c[0, :3, 3].numpy()
    assert (cam[:, 2] > 0).mean() > 0.99


def test_pose_refinement_recovers_from_noisy_training_poses(world, tmp_path):
    scene, depth, split = world
    pdir = write_priors(tmp_path, scene, depth, split, pose_noise=0.03)
    cfg = ExperimentConfig(pose_source="vggt", **{**FAST, "pose_refine_steps": 150})
    res = run_experiment(scene, split, cfg, prior_dir=pdir, log=None)
    assert res["diagnostics"]["train_pose_error"]["rel_centre_err_mean"] > 1e-3
    assert res["metrics"]["psnr"] > res["metrics"]["psnr_before_refine"]


def test_train_pose_optimisation_runs(world, tmp_path):
    scene, depth, split = world
    pdir = write_priors(tmp_path, scene, depth, split, pose_noise=0.03)
    cfg = ExperimentConfig(pose_source="vggt", optimize_train_poses=True, **FAST)
    res = run_experiment(scene, split, cfg, prior_dir=pdir, log=None)
    assert "train_pose_error_after_opt" in res["diagnostics"]


def test_honest_protocol_refuses_leaky_inputs(world, tmp_path):
    scene, depth, split = world
    pdir = write_priors(tmp_path, scene, depth, split)
    # full-view SfM points are not allowed with estimated poses
    with pytest.raises(ValueError, match="leak"):
        run_experiment(scene, split, ExperimentConfig(pose_source="vggt", init="sfm_all", **FAST), prior_dir=pdir, log=None)
    # a prior computed from other views (here: it saw a test view) is rejected
    other = [split["train"][0], split["test"][1], split["train"][2]]
    save_mv_prior(mv_prior_path(pdir, 3), fake_multiview_prior(scene, depth, other))
    with pytest.raises(ValueError, match="Recompute"):
        run_experiment(scene, split, ExperimentConfig(pose_source="vggt", **FAST), prior_dir=pdir, log=None)


def test_prior_file_roundtrip(world, tmp_path):
    scene, depth, split = world
    p = fake_multiview_prior(scene, depth, split["train"])
    save_mv_prior(tmp_path / "p.npz", p)
    q = load_mv_prior(tmp_path / "p.npz", expected_names=p.names)
    assert q.names == p.names and np.allclose(q.w2c, p.w2c) and np.allclose(q.depth, p.depth)


def test_confidence_weighting_protects_against_a_bad_depth_region(world, tmp_path):
    """Mechanism check on synthetic data: a corrupted, low-confidence depth region should
    damage geometry when confidence is ignored and not when it is used."""
    scene, depth, split = world
    pdir = write_priors(tmp_path, scene, depth, split, bad_region=True)
    base = dict(pose_source="vggt", depth_prior="vggt", lambda_depth=1.0, **{**FAST, "max_steps": 300})
    losses = {}
    for mode in ("none", "threshold"):
        cfg = ExperimentConfig(conf_mode=mode, conf_lo_q=0.3, **base)
        losses[mode] = run_experiment(scene, split, cfg, prior_dir=pdir, log=None)["metrics"]["psnr"]
    assert losses["threshold"] > losses["none"]
