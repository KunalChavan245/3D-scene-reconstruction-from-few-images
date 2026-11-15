"""CPU-only sanity check of the whole pipeline on a synthetic scene (about a minute, no GPU, no downloads).

    python scripts/demo_synthetic.py

It trains from 3 views under both protocols and writes assets/sandbox_check.png.
The scene is synthetic and the "estimated" poses and depth are simulated, so the
numbers it prints say that the code works, not how good any method is.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

import _common  # noqa: F401

from fewview3d.experiment import ExperimentConfig, run_experiment
from fewview3d.priors import mv_prior_path, save_mv_prior
from fewview3d.splits import make_split
from fewview3d.synthetic import fake_multiview_prior, make_synthetic_scene


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="assets/sandbox_check.png")
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()

    scene, gt_depth, _ = make_synthetic_scene(width=96, height=72)
    split = make_split(len(scene), 3)
    common = dict(max_steps=args.steps, refine_start=100, refine_every=100, reset_every=10_000, with_lpips=False,
                  max_init_points=600, pose_refine_steps=100)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prior = fake_multiview_prior(scene, gt_depth, split["train"], pose_noise=0.02, depth_noise=0.02)
        save_mv_prior(mv_prior_path(tmp / "priors", 3), prior)
        for label, cfg in {
            "COLMAP-style poses, no prior": ExperimentConfig(**common),
            "Honest poses + multi-view depth": ExperimentConfig(pose_source="vggt", depth_prior="vggt", **common),
        }.items():
            res = run_experiment(scene, split, cfg, tmp / label.replace(" ", "_"), tmp / "priors", log=None)
            renders = sorted((tmp / label.replace(" ", "_") / "renders").glob("*.png"))
            depths = sorted((tmp / label.replace(" ", "_") / "renders").glob("*_depth.npy"))
            from PIL import Image

            results[label] = (res, np.asarray(Image.open(renders[1])), np.load(depths[1]))
            print(f"{label}: PSNR {res['metrics']['psnr']:.2f} dB, SSIM {res['metrics']['ssim']:.3f} "
                  f"(3 held-out views, synthetic scene)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_i = split["test"][1]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8))
    for ax, i in zip(axes[0, :3], split["train"]):
        ax.imshow(scene.images[i])
        ax.set_title(f"training view {i}", fontsize=10)
    axes[0, 3].imshow(scene.images[test_i])
    axes[0, 3].set_title("held-out view (ground truth)", fontsize=10)
    vmin, vmax = gt_depth[test_i].min(), gt_depth[test_i].max()
    for col, (label, (res, img, _)) in enumerate(results.items()):
        axes[1, col].imshow(img)
        axes[1, col].set_title(f"{label}\nrender, PSNR {res['per_image'][1]['psnr']:.1f} dB", fontsize=9)
    axes[1, 2].imshow(gt_depth[test_i], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 2].set_title("held-out depth (ground truth)", fontsize=9)
    honest = list(results.values())[1]
    d = honest[2]
    # the honest run lives in the estimated frame; rescale its depth to the ground-truth scale for display
    d = d * np.median(gt_depth[test_i]) / max(np.median(d), 1e-9)
    axes[1, 3].imshow(d, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 3].set_title("honest run: rendered depth\n(median-scaled for display)", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Sandbox check on a synthetic scene (CPU reference rasteriser). Not an experimental result.",
                 fontsize=11)
    fig.tight_layout(h_pad=2.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()
