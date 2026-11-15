"""Write the train/test split files that every experiment reads.

    python scripts/make_splits.py --dataset llff --views 3 6 9
"""

import argparse

from _common import add_data_args, factor_of, scenes_of

from fewview3d.dataset import load_colmap_scene
from fewview3d.registry import DATASETS, scene_dir, split_dir
from fewview3d.splits import make_split, save_split, split_filename


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--views", type=int, nargs="*", default=None, help="training view counts; 0 = all views")
    p.add_argument("--test-every", type=int, default=8)
    args = p.parse_args()
    views = args.views if args.views is not None else DATASETS[args.dataset]["views"]
    for name in scenes_of(args):
        scene = load_colmap_scene(scene_dir(args.data_root, args.dataset, name), factor_of(args), name)
        for n in views:
            split = make_split(len(scene), n, args.test_every)
            path = split_dir(args.work_root, args.dataset, name) / split_filename(n)
            save_split(path, scene.names, split, {"dataset": args.dataset, "scene": name, "test_every": args.test_every})
            print(f"{path}: {len(split['train'])} train / {len(split['test'])} test of {len(scene)} images")


if __name__ == "__main__":
    main()
