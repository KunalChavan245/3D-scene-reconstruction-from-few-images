"""Shared command-line plumbing for the scripts."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fewview3d.experiment import ExperimentConfig  # noqa: E402
from fewview3d.registry import DATASETS  # noqa: E402


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default="data", help="folder holding <dataset>/<scene>/ in COLMAP layout")
    p.add_argument("--work-root", default=".", help="where splits/, priors/ and outputs/ are written")
    p.add_argument("--dataset", default="llff", choices=sorted(DATASETS))
    p.add_argument("--scenes", nargs="*", default=None, help="default: every scene of the dataset")
    p.add_argument("--factor", type=int, default=None, help="downsampling factor (default: the dataset's)")


def scenes_of(args) -> list[str]:
    return args.scenes or DATASETS[args.dataset]["scenes"]


def factor_of(args) -> int:
    return args.factor or DATASETS[args.dataset]["factor"]


def add_config_args(p: argparse.ArgumentParser, skip: tuple[str, ...] = ()) -> None:
    """Expose every ExperimentConfig field as a ``--flag``."""
    for f in dataclasses.fields(ExperimentConfig):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(flag, type=type(f.default), default=f.default)


def config_from_args(args, **overrides) -> ExperimentConfig:
    values = {f.name: getattr(args, f.name) for f in dataclasses.fields(ExperimentConfig) if hasattr(args, f.name)}
    values.update(overrides)
    return ExperimentConfig(**values)
