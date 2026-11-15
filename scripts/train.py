"""Train and evaluate one configuration on one scene.

    # standard (leaky) protocol, no prior
    python scripts/train.py --dataset llff --scenes fern --n-views 3

    # honest protocol with confidence-weighted multi-view depth
    python scripts/train.py --dataset llff --scenes fern --n-views 3 --pose-source vggt --depth-prior vggt
"""

import argparse

from _common import add_config_args, add_data_args, config_from_args, factor_of, scenes_of

from fewview3d.dataset import load_colmap_scene
from fewview3d.experiment import run_experiment
from fewview3d.registry import output_dir, prior_dir, scene_dir, split_dir
from fewview3d.splits import load_split, split_filename


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    add_config_args(p)
    p.add_argument("--tag", default="", help="suffix for the output folder, e.g. an ablation name")
    args = p.parse_args()
    cfg = config_from_args(args)
    for name in scenes_of(args):
        scene = load_colmap_scene(scene_dir(args.data_root, args.dataset, name), factor_of(args), name)
        split = load_split(split_dir(args.work_root, args.dataset, name) / split_filename(cfg.n_views), scene.names)
        run = cfg.run_name() + (f"_{args.tag}" if args.tag else "")
        run_experiment(scene, split, cfg, output_dir(args.work_root, args.dataset, name, run),
                       prior_dir(args.work_root, args.dataset, name))


if __name__ == "__main__":
    main()
