"""Train/test splits for few-view novel view synthesis.

The split follows the protocol used by RegNeRF, FSGS and DNGaussian on LLFF and
Mip-NeRF 360: images are sorted by name, every ``test_every``-th image is held
out for testing, and ``n_train`` training views are picked at evenly spaced
positions among the remaining images.

Every experiment reads its split from a JSON file written by this module, so
all methods are compared on exactly the same views.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_split(n_images: int, n_train: int, test_every: int = 8) -> dict:
    """Return ``{"train": [...], "test": [...]}`` as indices into the sorted image list.

    ``n_train <= 0`` keeps every non-test image (the full-view setting).
    """
    if n_images < 2:
        raise ValueError("need at least 2 images")
    all_idx = np.arange(n_images)
    test = all_idx[::test_every]
    rest = np.array([i for i in all_idx if i not in set(test.tolist())])
    if n_train <= 0 or n_train >= len(rest):
        train = rest
    else:
        pick = np.round(np.linspace(0, len(rest) - 1, n_train)).astype(int)
        train = rest[pick]
    if len(set(train.tolist())) != len(train):
        raise ValueError("duplicate training views; too few images for this n_train")
    return {"train": train.tolist(), "test": test.tolist()}


def save_split(path: str | Path, names: list[str], split: dict, meta: dict | None = None) -> None:
    """Write a split with both indices and image names (names are the ground truth)."""
    out = {
        "train": split["train"],
        "test": split["test"],
        "train_names": [names[i] for i in split["train"]],
        "test_names": [names[i] for i in split["test"]],
        "n_images": len(names),
    }
    if meta:
        out["meta"] = meta
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


def load_split(path: str | Path, names: list[str] | None = None) -> dict:
    """Read a split file. If ``names`` is given, indices are re-derived from the stored names."""
    data = json.loads(Path(path).read_text())
    if names is not None:
        lookup = {n: i for i, n in enumerate(names)}
        missing = [n for n in data["train_names"] + data["test_names"] if n not in lookup]
        if missing:
            raise KeyError(f"split refers to images not in the scene: {missing[:3]}")
        data["train"] = [lookup[n] for n in data["train_names"]]
        data["test"] = [lookup[n] for n in data["test_names"]]
    if set(data["train"]) & set(data["test"]):
        raise ValueError("split has overlapping train and test views")
    return data


def split_filename(n_train: int) -> str:
    return f"split_{n_train}views.json" if n_train > 0 else "split_full.json"
