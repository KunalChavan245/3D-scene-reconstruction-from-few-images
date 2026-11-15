"""Datasets, scenes and provisional scene-type labels used by the study."""

from __future__ import annotations

from pathlib import Path

DATASETS = {
    # Forward-facing captures, 20-62 images each. Standard few-view benchmark (3/6/9 views, 1/8 resolution).
    "llff": {
        "scenes": ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"],
        "factor": 8,
        "views": [3, 6, 9],
        "camera_path": "forward-facing",
    },
    # Unbounded 360-degree captures, 125-311 images each. Few-view papers use 12 or 24 views.
    "mipnerf360": {
        "scenes": ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"],
        "factor": 8,
        "views": [12, 24],
        "camera_path": "360",
    },
}

# Scene-level tags for the "where does a prior help or hurt" analysis. These are a
# first pass made from the image content and are meant to be refined (ideally into
# per-region masks) once the first renders exist. A scene can carry several tags.
SCENE_TYPES = {
    "llff": {
        "fern": ["foliage", "thin-structures"],
        "flower": ["foliage"],
        "fortress": ["textured", "lambertian"],
        "horns": ["thin-structures", "reflective"],
        "leaves": ["foliage", "thin-structures"],
        "orchids": ["foliage", "thin-structures"],
        "room": ["textureless", "indoor"],
        "trex": ["thin-structures", "reflective", "indoor"],
    },
    "mipnerf360": {
        "bicycle": ["outdoor", "foliage", "thin-structures"],
        "bonsai": ["indoor", "thin-structures"],
        "counter": ["indoor", "reflective"],
        "garden": ["outdoor", "foliage", "textured"],
        "kitchen": ["indoor", "reflective"],
        "room": ["indoor", "textureless"],
        "stump": ["outdoor", "foliage"],
    },
}


def scene_dir(data_root: str | Path, dataset: str, scene: str) -> Path:
    return Path(data_root) / dataset / scene


def split_dir(work_root: str | Path, dataset: str, scene: str) -> Path:
    return Path(work_root) / "splits" / dataset / scene


def prior_dir(work_root: str | Path, dataset: str, scene: str) -> Path:
    return Path(work_root) / "priors" / dataset / scene


def output_dir(work_root: str | Path, dataset: str, scene: str, run_name: str) -> Path:
    return Path(work_root) / "outputs" / dataset / scene / run_name
