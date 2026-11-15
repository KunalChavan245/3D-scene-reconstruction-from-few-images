"""One experiment = one scene, one split, one pose source, one depth prior, one seed.

``run_experiment`` wires everything together and writes ``results.json``. The
two protocol axes the study is about are:

``pose_source``
    ``colmap``: the standard few-view protocol. Training and test poses come from
    COLMAP run on *all* images of the scene, so information from held-out views
    leaks into the training cameras (and, with ``init=sfm_all``, into the
    initial point cloud).

    ``vggt``: the honest protocol. Poses, intrinsics and the initial points come
    from a multi-view model that saw only the training views. Test cameras are
    mapped into that frame and refined photometrically at evaluation time.

``depth_prior``
    ``none``, ``mono`` (per-image relative inverse depth, Pearson loss) or
    ``vggt`` (multi-view depth, confidence-weighted L1).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset import SceneData
from .gaussians import create_optimizers, init_gaussians, random_points_in_bounds
from .geometry import (
    align_cameras_sim3,
    apply_sim3_to_c2w,
    camera_extent,
    invert_se3,
    pose_errors,
    scale_intrinsics,
    se3_exp,
    unproject_depth,
)
from .losses import confidence_depth_loss, confidence_weights, pearson_depth_loss, photometric_loss
from .metrics import image_metrics
from .pose_refine import refine_camera_pose
from .priors import MultiViewPrior, load_mono_prior, load_mv_prior, mv_prior_path, resize_map
from .render import pick_backend, render


@dataclass
class ExperimentConfig:
    # protocol
    pose_source: str = "colmap"  # colmap | vggt
    depth_prior: str = "none"  # none | mono | vggt
    init: str = "auto"  # auto | sfm_all | sfm_train | random | prior
    n_views: int = 3
    seed: int = 0
    # depth loss
    lambda_depth: float = 0.05
    depth_align: str = "auto"  # auto | none | scale | scale_shift
    conf_mode: str = "soft"  # none | threshold | soft
    conf_lo_q: float = 0.2
    conf_hi_q: float = 0.8
    depth_start_step: int = 0
    # optimisation
    max_steps: int = 10_000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    lambda_dssim: float = 0.2
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    optimize_train_poses: bool = False
    lr_pose: float = 1e-4
    # densification (gsplat DefaultStrategy)
    refine_start: int = 500
    refine_stop_frac: float = 0.5
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 2e-4
    prune_opa: float = 0.005
    # initialisation
    max_init_points: int = 100_000
    n_random_points: int = 10_000
    init_conf_q: float = 0.3
    sfm_min_views: int = 2
    # evaluation
    test_pose_refine: str = "auto"  # auto | on | off
    pose_refine_steps: int = 200
    pose_refine_lr: float = 2e-3
    with_lpips: bool = True
    save_renders: bool = True
    # runtime
    backend: str = "auto"  # auto | gsplat | torch
    device: str = "auto"

    def run_name(self) -> str:
        return f"pose-{self.pose_source}_prior-{self.depth_prior}_{self.n_views}v_seed{self.seed}"


@dataclass
class Views:
    """A set of cameras with their images, on the training device."""

    images: torch.Tensor  # (N, H, W, 3) float in [0, 1]
    K: torch.Tensor  # (N, 3, 3)
    w2c: torch.Tensor  # (N, 4, 4)
    names: list[str]


@dataclass
class DepthPrior:
    kind: str  # "disparity" (relative inverse depth) or "depth"
    depth: torch.Tensor  # (N, H, W)
    conf: torch.Tensor | None = None


def _device(cfg: ExperimentConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_views(images: np.ndarray, K: np.ndarray, w2c: np.ndarray, names: list[str], device) -> Views:
    return Views(
        images=torch.as_tensor(images, dtype=torch.float32, device=device) / 255.0,
        K=torch.as_tensor(K, dtype=torch.float32, device=device),
        w2c=torch.as_tensor(w2c, dtype=torch.float32, device=device),
        names=list(names),
    )


# --------------------------------------------------------------------------- #
# Protocol setup
# --------------------------------------------------------------------------- #
def resolve_init(cfg: ExperimentConfig) -> str:
    if cfg.init != "auto":
        return cfg.init
    return "prior" if cfg.pose_source == "vggt" else "sfm_train"


def resolve_depth_align(cfg: ExperimentConfig) -> str:
    """Multi-view depth shares the frame of the poses only if both come from the same model."""
    if cfg.depth_align != "auto":
        return cfg.depth_align
    return "none" if cfg.pose_source == "vggt" else "scale_shift"


def points_from_prior(prior: MultiViewPrior, images: np.ndarray, cfg: ExperimentConfig):
    """Unproject the confident part of the predicted depth maps into one coloured point cloud."""
    h, w = prior.proc_hw
    c2w = invert_se3(prior.w2c)
    xyz, rgb = [], []
    for i in range(len(prior.names)):
        pts = unproject_depth(prior.depth[i].astype(np.float64), prior.K[i], c2w[i]).reshape(-1, 3)
        img = torch.as_tensor(images[i], dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
        col = torch.nn.functional.interpolate(img, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        col = col[0].permute(1, 2, 0).reshape(-1, 3).clamp(0, 1).numpy()
        conf = prior.conf[i].reshape(-1)
        keep = (conf >= np.quantile(conf, cfg.init_conf_q)) & (prior.depth[i].reshape(-1) > 0)
        xyz.append(pts[keep])
        rgb.append(col[keep])
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    if len(xyz) > cfg.max_init_points:
        pick = np.random.default_rng(cfg.seed).choice(len(xyz), cfg.max_init_points, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    return xyz, rgb


def build_protocol(scene: SceneData, split: dict, cfg: ExperimentConfig, prior_dir: str | Path | None, device):
    """Return training views, test views (initial poses), init points, depth prior and diagnostics."""
    tr, te = split["train"], split["test"]
    train_names = [scene.names[i] for i in tr]
    H, W = scene.height, scene.width
    diag: dict = {}
    mv: MultiViewPrior | None = None
    if cfg.pose_source == "vggt" or cfg.depth_prior == "vggt":
        if prior_dir is None:
            raise ValueError("a prior directory is required for pose_source=vggt or depth_prior=vggt")
        mv = load_mv_prior(mv_prior_path(prior_dir, len(tr)), expected_names=train_names)

    init = resolve_init(cfg)
    if cfg.pose_source == "colmap":
        K_tr, w2c_tr = scene.K[tr], scene.w2c[tr]
        K_te, w2c_te = scene.K[te], scene.w2c[te]
        if init == "sfm_all":
            xyz, rgb = scene.points, scene.points_rgb / 255.0
        elif init == "sfm_train":
            mask = scene.points_seen_by(tr, cfg.sfm_min_views)
            xyz, rgb = scene.points[mask], scene.points_rgb[mask] / 255.0
            diag["sfm_points_total"] = int(len(scene.points))
        elif init == "random":
            xyz, rgb = random_points_in_bounds(scene.points, cfg.n_random_points, cfg.seed)
        else:
            raise ValueError(f"init={init!r} is not available with COLMAP poses")
    elif cfg.pose_source == "vggt":
        h, w = mv.proc_hw
        K_tr = scale_intrinsics(mv.K, W / w, H / h)
        w2c_tr = mv.w2c
        if init == "prior":
            xyz, rgb = points_from_prior(mv, scene.images[tr], cfg)
        elif init == "random":
            ref, _ = points_from_prior(mv, scene.images[tr], cfg)
            xyz, rgb = random_points_in_bounds(ref, cfg.n_random_points, cfg.seed)
        else:
            raise ValueError(f"init={init!r} would leak full-view SfM points into the honest protocol")
        # Map the ground-truth test cameras into the estimated frame (initial guess only).
        gt_c2w_tr, est_c2w_tr = invert_se3(scene.w2c[tr]), invert_se3(w2c_tr)
        s, R, t = align_cameras_sim3(gt_c2w_tr, est_c2w_tr)
        w2c_te = invert_se3(apply_sim3_to_c2w(invert_se3(scene.w2c[te]), s, R, t))
        K_te = np.tile(K_tr.mean(axis=0), (len(te), 1, 1))
        diag["train_pose_error"] = pose_errors(est_c2w_tr, gt_c2w_tr)
        gt_focal = float(scene.K[tr][:, [0, 1], [0, 1]].mean())
        diag["focal_rel_error"] = float(abs(K_tr[:, [0, 1], [0, 1]].mean() - gt_focal) / gt_focal)
    else:
        raise ValueError(f"unknown pose_source {cfg.pose_source!r}")
    if len(xyz) < 10:
        raise ValueError(f"only {len(xyz)} initial points; try init=sfm_all or a lower sfm_min_views")
    diag["n_init_points"] = int(len(xyz))
    diag["init"] = init

    train = _to_views(scene.images[tr], K_tr, w2c_tr, train_names, device)
    test = _to_views(scene.images[te], K_te, w2c_te, [scene.names[i] for i in te], device)

    prior = None
    if cfg.depth_prior == "mono":
        maps = load_mono_prior(prior_dir, train_names)
        prior = DepthPrior("disparity", torch.stack([resize_map(m, H, W) for m in maps]).to(device))
    elif cfg.depth_prior == "vggt":
        depth = torch.stack([resize_map(d, H, W) for d in mv.depth]).to(device)
        conf = torch.stack([resize_map(c, H, W) for c in mv.conf]).to(device)
        prior = DepthPrior("depth", depth, conf)
    elif cfg.depth_prior != "none":
        raise ValueError(f"unknown depth_prior {cfg.depth_prior!r}")
    return train, test, (xyz, rgb), prior, diag


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_gaussians(train: Views, init_points, cfg: ExperimentConfig, prior: DepthPrior | None = None, log=print):
    """Optimise Gaussians on the training views. Returns (params, final train w2c, stats)."""
    from gsplat import DefaultStrategy

    device = train.images.device
    backend = pick_backend(cfg.backend, device)
    max_sh = cfg.sh_degree if backend == "gsplat" else 0
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    N, H, W = train.images.shape[:3]

    scene_scale = camera_extent(invert_se3(train.w2c.cpu().numpy())) * 1.1
    xyz, rgb = init_points
    params = init_gaussians(xyz, rgb, sh_degree=max_sh, device=device, seed=cfg.seed)
    optimizers = create_optimizers(params, scene_scale, cfg)
    means_decay = 0.01 ** (1.0 / max(cfg.max_steps, 1))

    pose_delta, pose_opt = None, None
    if cfg.optimize_train_poses:
        pose_delta = torch.zeros(N, 6, device=device, requires_grad=True)
        pose_opt = torch.optim.Adam([pose_delta], lr=cfg.lr_pose)

    strategy = DefaultStrategy(
        prune_opa=cfg.prune_opa,
        grow_grad2d=cfg.grow_grad2d,
        refine_start_iter=cfg.refine_start,
        refine_stop_iter=int(cfg.max_steps * cfg.refine_stop_frac),
        refine_every=cfg.refine_every,
        reset_every=cfg.reset_every,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    weights = None
    if prior is not None and prior.kind == "depth":
        weights = torch.stack(
            [confidence_weights(None if prior.conf is None else prior.conf[i], prior.depth[i], cfg.conf_mode,
                                cfg.conf_lo_q, cfg.conf_hi_q) for i in range(N)]
        )
    align = resolve_depth_align(cfg)

    t0 = time.time()
    last = {}
    for step in range(cfg.max_steps):
        i = int(rng.integers(N))
        w2c = train.w2c[i]
        if pose_delta is not None:
            w2c = se3_exp(pose_delta[i]) @ w2c
        sh_deg = min(step // max(cfg.sh_degree_interval, 1), max_sh)
        rgb_r, depth_r, _, info = render(params, w2c[None], train.K[i][None], W, H, sh_degree=sh_deg, backend=backend)
        strategy.step_pre_backward(params, optimizers, state, step, info)

        loss_rgb = photometric_loss(rgb_r, train.images[i][None], cfg.lambda_dssim)
        loss = loss_rgb
        loss_d = None
        if prior is not None and cfg.lambda_depth > 0 and step >= cfg.depth_start_step:
            if prior.kind == "disparity":
                loss_d = pearson_depth_loss(depth_r[0], prior.depth[i])
            else:
                loss_d = confidence_depth_loss(depth_r[0], prior.depth[i], weights[i], align)
            loss = loss + cfg.lambda_depth * loss_d
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if pose_opt is not None:
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
        for group in optimizers["means"].param_groups:
            group["lr"] *= means_decay
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

        last = {"loss_rgb": loss_rgb.item(), "loss_depth": None if loss_d is None else loss_d.item()}
        if log and (step % max(cfg.max_steps // 10, 1) == 0 or step == cfg.max_steps - 1):
            log(f"step {step:6d}  rgb {last['loss_rgb']:.4f}  depth {last['loss_depth']}  "
                f"gaussians {params['means'].shape[0]}")

    w2c_final = train.w2c
    if pose_delta is not None:
        with torch.no_grad():
            w2c_final = se3_exp(pose_delta) @ train.w2c
    stats = {
        "train_seconds": time.time() - t0,
        "n_gaussians": int(params["means"].shape[0]),
        "scene_scale": scene_scale,
        "backend": backend,
        "sh_degree_used": max_sh,
        **{f"final_{k}": v for k, v in last.items()},
    }
    return params, w2c_final, stats


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else None


def evaluate(params, test: Views, cfg: ExperimentConfig, refine: bool, out_dir: Path | None = None) -> dict:
    """Render the test views and score them, optionally after photometric pose refinement."""
    device = test.images.device
    backend = pick_backend(cfg.backend, device)
    sh_deg = cfg.sh_degree if backend == "gsplat" else 0
    H, W = test.images.shape[1:3]
    frozen = {k: v.detach() for k, v in params.items()}
    per_image = []
    for i, name in enumerate(test.names):
        w2c = test.w2c[i]
        row = {"name": name}
        if refine:
            with torch.no_grad():
                rgb0, _, _, _ = render(frozen, w2c[None], test.K[i][None], W, H, sh_degree=sh_deg, backend=backend)
            row["psnr_before_refine"] = image_metrics(rgb0[0], test.images[i], with_lpips=False)["psnr"]
            w2c, info = refine_camera_pose(
                frozen, w2c, test.K[i], test.images[i], steps=cfg.pose_refine_steps, lr=cfg.pose_refine_lr,
                sh_degree=sh_deg, backend=backend, lambda_dssim=cfg.lambda_dssim,
            )
            row.update(info)
        with torch.no_grad():
            rgb, depth, _, _ = render(frozen, w2c[None], test.K[i][None], W, H, sh_degree=sh_deg, backend=backend)
        row.update(image_metrics(rgb[0], test.images[i], with_lpips=cfg.with_lpips))
        per_image.append(row)
        if out_dir is not None and cfg.save_renders:
            _save_png(out_dir / "renders" / f"{Path(name).stem}.png", rgb[0])
            np.save(out_dir / "renders" / f"{Path(name).stem}_depth.npy", depth[0].cpu().numpy().astype(np.float32))
    summary = {k: _mean([r.get(k) for r in per_image]) for k in ("psnr", "ssim", "lpips", "psnr_before_refine")}
    return {"mean": summary, "per_image": per_image, "pose_refined": refine}


def _save_png(path: Path, img: torch.Tensor) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def resolve_test_refine(cfg: ExperimentConfig) -> bool:
    if cfg.test_pose_refine == "auto":
        return cfg.pose_source == "vggt"
    return cfg.test_pose_refine == "on"


def run_experiment(
    scene: SceneData,
    split: dict,
    cfg: ExperimentConfig,
    out_dir: str | Path | None = None,
    prior_dir: str | Path | None = None,
    log=print,
) -> dict:
    """Train and evaluate one configuration. Writes ``results.json`` into ``out_dir`` if given."""
    device = _device(cfg)
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    train, test, init_points, prior, diag = build_protocol(scene, split, cfg, prior_dir, device)
    params, w2c_final, stats = train_gaussians(train, init_points, cfg, prior, log)

    if cfg.optimize_train_poses and cfg.pose_source == "vggt":
        # The scene followed the refined training cameras, so re-anchor the test cameras on them.
        tr, te = split["train"], split["test"]
        est_c2w = invert_se3(w2c_final.cpu().numpy())
        gt_c2w = invert_se3(scene.w2c[tr])
        s, R, t = align_cameras_sim3(gt_c2w, est_c2w)
        w2c_te = invert_se3(apply_sim3_to_c2w(invert_se3(scene.w2c[te]), s, R, t))
        test.w2c = torch.as_tensor(w2c_te, dtype=torch.float32, device=device)
        diag["train_pose_error_after_opt"] = pose_errors(est_c2w, gt_c2w)

    metrics = evaluate(params, test, cfg, resolve_test_refine(cfg), out)
    result = {
        "scene": scene.name,
        "run": cfg.run_name(),
        "config": asdict(cfg),
        "train_views": train.names,
        "test_views": test.names,
        "resolution": [scene.width, scene.height],
        "diagnostics": diag,
        "stats": stats,
        "metrics": metrics["mean"],
        "pose_refined": metrics["pose_refined"],
        "per_image": metrics["per_image"],
    }
    if out is not None:
        (out / "results.json").write_text(json.dumps(result, indent=2))
        torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "gaussians.pt")
    if log:
        log(f"[{scene.name}] {cfg.run_name()}: {result['metrics']}")
    return result
