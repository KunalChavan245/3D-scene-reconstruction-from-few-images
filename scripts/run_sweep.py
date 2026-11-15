"""Run the main experiment grid: scenes x view counts x seeds x (pose source, depth prior).

    python scripts/run_sweep.py --dataset llff --views 3 --seeds 0 1 2

Finished runs (those with a results.json) are skipped, so the sweep can be
interrupted and resumed, which matters on Kaggle/Colab session limits.
"""

import argparse
import traceback

from _common import add_config_args, add_data_args, config_from_args, factor_of, scenes_of

from fewview3d.dataset import load_colmap_scene
from fewview3d.experiment import run_experiment
from fewview3d.registry import DATASETS, output_dir, prior_dir, scene_dir, split_dir
from fewview3d.splits import load_split, split_filename

# (pose_source, depth_prior): the six cells of the main table.
GRID = [
    ("colmap", "none"), ("colmap", "mono"), ("colmap", "vggt"),
    ("vggt", "none"), ("vggt", "mono"), ("vggt", "vggt"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    add_config_args(p, skip=("pose_source", "depth_prior", "n_views", "seed"))
    p.add_argument("--views", type=int, nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--cells", nargs="*", default=None, help="subset of the grid, e.g. colmap:none vggt:vggt")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    views = args.views if args.views is not None else DATASETS[args.dataset]["views"]
    grid = [tuple(c.split(":")) for c in args.cells] if args.cells else GRID
    failures = []
    for name in scenes_of(args):
        scene = load_colmap_scene(scene_dir(args.data_root, args.dataset, name), factor_of(args), name)
        for n in views:
            split = load_split(split_dir(args.work_root, args.dataset, name) / split_filename(n), scene.names)
            for pose, prior in grid:
                for seed in args.seeds:
                    cfg = config_from_args(args, pose_source=pose, depth_prior=prior, n_views=n, seed=seed)
                    run = cfg.run_name() + (f"_{args.tag}" if args.tag else "")
                    out = output_dir(args.work_root, args.dataset, name, run)
                    if (out / "results.json").exists():
                        print(f"skip {out} (done)")
                        continue
                    try:
                        run_experiment(scene, split, cfg, out, prior_dir(args.work_root, args.dataset, name))
                    except Exception:  # keep the sweep going, report at the end
                        traceback.print_exc()
                        failures.append(str(out))
    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
