"""Monocular depth prior: run Depth Anything V2 on the training images of each split.

    python scripts/precompute_mono_depth.py --dataset llff --views 3 6 9

Writes priors/<dataset>/<scene>/mono/<image>.npy. Needs a GPU and `transformers`.
"""

import argparse

from _common import add_data_args, factor_of, scenes_of

from fewview3d.dataset import load_colmap_scene
from fewview3d.priors.mono_runner import DEFAULT_MODEL, load_pipeline, predict_disparity, save_disparity
from fewview3d.registry import DATASETS, prior_dir, scene_dir, split_dir
from fewview3d.splits import load_split, split_filename


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--views", type=int, nargs="*", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    pipe = load_pipeline(args.model, args.device)
    views = args.views if args.views is not None else DATASETS[args.dataset]["views"]
    for name in scenes_of(args):
        scene = load_colmap_scene(scene_dir(args.data_root, args.dataset, name), factor_of(args), name)
        needed = set()
        for n in views:
            needed |= set(load_split(split_dir(args.work_root, args.dataset, name) / split_filename(n), scene.names)["train"])
        for i in sorted(needed):
            path = save_disparity(prior_dir(args.work_root, args.dataset, name), scene.names[i],
                                  predict_disparity(pipe, scene.images[i]))
        print(f"{name}: {len(needed)} depth maps in {path.parent}")


if __name__ == "__main__":
    main()
