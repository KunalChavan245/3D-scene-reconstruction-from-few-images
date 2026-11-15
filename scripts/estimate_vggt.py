"""Honest poses and multi-view depth: run VGGT on the training views of each split only.

    python scripts/estimate_vggt.py --dataset llff --views 3 6 9

Writes priors/<dataset>/<scene>/vggt_<n>views.npz. Needs a GPU and the vggt package.
"""

import argparse

from _common import add_data_args, factor_of, scenes_of

from fewview3d.dataset import load_colmap_scene
from fewview3d.geometry import invert_se3, pose_errors
from fewview3d.priors import mv_prior_path, save_mv_prior
from fewview3d.priors.vggt_runner import run_vggt
from fewview3d.registry import DATASETS, prior_dir, scene_dir, split_dir
from fewview3d.splits import load_split, split_filename


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--views", type=int, nargs="*", default=None)
    p.add_argument("--checkpoint", default="facebook/VGGT-1B")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained(args.checkpoint).to(args.device).eval()
    views = args.views if args.views is not None else DATASETS[args.dataset]["views"]
    for name in scenes_of(args):
        scene = load_colmap_scene(scene_dir(args.data_root, args.dataset, name), factor_of(args), name)
        for n in views:
            split = load_split(split_dir(args.work_root, args.dataset, name) / split_filename(n), scene.names)
            tr = split["train"]
            prior = run_vggt(scene.images[tr], [scene.names[i] for i in tr], model=model, device=args.device)
            path = mv_prior_path(prior_dir(args.work_root, args.dataset, name), len(tr))
            save_mv_prior(path, prior)
            # Diagnostic only: how far are the estimated poses from full-view COLMAP?
            err = pose_errors(invert_se3(prior.w2c), invert_se3(scene.w2c[tr]))
            print(f"{path}: rot err {err['rot_err_deg_mean']:.2f} deg, "
                  f"relative centre err {err['rel_centre_err_mean']:.3f}")


if __name__ == "__main__":
    main()
