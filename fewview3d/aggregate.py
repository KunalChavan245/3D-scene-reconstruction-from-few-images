"""Collect results.json files into the tables the paper needs. Pure Python, no pandas."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .registry import SCENE_TYPES

METRICS = ("psnr", "ssim", "lpips")


def collect(outputs_root: str | Path) -> list[dict]:
    """Read every ``outputs/<dataset>/<scene>/<run>/results.json`` into a flat row."""
    rows = []
    for path in sorted(Path(outputs_root).glob("*/*/*/results.json")):
        res = json.loads(path.read_text())
        cfg = res["config"]
        row = {
            "dataset": path.parts[-4],
            "scene": res["scene"],
            "run": path.parts[-2],
            "n_views": cfg["n_views"],
            "pose": cfg["pose_source"],
            "prior": cfg["depth_prior"],
            "seed": cfg["seed"],
            "n_gaussians": res["stats"]["n_gaussians"],
            "psnr_before_refine": res["metrics"].get("psnr_before_refine"),
        }
        row["tag"] = row["run"].split(f"seed{cfg['seed']}", 1)[-1].lstrip("_")
        for m in METRICS:
            row[m] = res["metrics"].get(m)
        err = res["diagnostics"].get("train_pose_error")
        row["pose_rot_err_deg"] = err["rot_err_deg_mean"] if err else None
        row["pose_centre_err"] = err["rel_centre_err_mean"] if err else None
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def seed_mean_std(rows: list[dict], metric: str) -> tuple[float | None, float | None, int]:
    """Average over scenes within each seed, then mean and std across seeds."""
    by_seed = defaultdict(list)
    for r in rows:
        if r[metric] is not None:
            by_seed[r["seed"]].append(r[metric])
    per_seed = [float(np.mean(v)) for v in by_seed.values()]
    if not per_seed:
        return None, None, 0
    return float(np.mean(per_seed)), float(np.std(per_seed)), len(per_seed)


def _fmt(mean, std=None, digits=2):
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None else "")


def group(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def main_table(rows: list[dict]) -> str:
    """Mean ± std over seeds for every (dataset, views, pose source, prior) cell."""
    lines = ["| Dataset | Views | Poses | Depth prior | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Scenes | Seeds |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (ds, n, pose, prior), grp in sorted(group(rows, ("dataset", "n_views", "pose", "prior")).items()):
        cells = [_fmt(*seed_mean_std(grp, m)[:2], digits=2 if m == "psnr" else 3) for m in METRICS]
        seeds = seed_mean_std(grp, "psnr")[2]
        lines.append(f"| {ds} | {n} | {pose} | {prior} | " + " | ".join(cells)
                     + f" | {len({r['scene'] for r in grp})} | {seeds} |")
    return "\n".join(lines)


def scene_means(rows: list[dict], metric: str = "psnr") -> dict:
    """{(dataset, n_views, pose, prior, scene): mean over seeds}."""
    return {k: _avg([r[metric] for r in v])
            for k, v in group(rows, ("dataset", "n_views", "pose", "prior", "scene")).items()}


def leakage_table(rows: list[dict], metric: str = "psnr") -> str:
    """How much each configuration loses when full-view COLMAP poses are replaced by honest ones."""
    sm = scene_means(rows, metric)
    lines = [f"| Dataset | Views | Depth prior | {metric.upper()} COLMAP poses | {metric.upper()} honest poses | Gap |",
             "|---|---|---|---|---|---|"]
    keys = sorted({(d, n, p) for (d, n, _, p, _) in sm})
    for d, n, prior in keys:
        scenes = sorted({s for (dd, nn, _, pp, s) in sm if (dd, nn, pp) == (d, n, prior)})
        pairs = [(sm.get((d, n, "colmap", prior, s)), sm.get((d, n, "vggt", prior, s))) for s in scenes]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        a, b = np.mean([x for x, _ in pairs]), np.mean([y for _, y in pairs])
        lines.append(f"| {d} | {n} | {prior} | {a:.2f} | {b:.2f} | {a - b:+.2f} |")
    return "\n".join(lines)


def prior_gain_rows(rows: list[dict], metric: str = "psnr") -> list[dict]:
    """Per scene: metric with a prior minus metric without, under the same poses and view count."""
    sm = scene_means(rows, metric)
    out = []
    for (d, n, pose, prior, scene), val in sorted(sm.items()):
        base = sm.get((d, n, pose, "none", scene))
        if prior == "none" or base is None or val is None:
            continue
        out.append({"dataset": d, "n_views": n, "pose": pose, "prior": prior, "scene": scene,
                    "gain": val - base, "types": SCENE_TYPES.get(d, {}).get(scene, [])})
    return out


def scene_type_table(rows: list[dict], metric: str = "psnr") -> str:
    """Average gain of each prior over no prior, broken down by scene type.

    ``hurt`` counts the scenes where adding the prior lowered the metric.
    """
    gains = prior_gain_rows(rows, metric)
    by = defaultdict(list)
    for g in gains:
        for t in g["types"] or ["untagged"]:
            by[(g["dataset"], g["n_views"], g["pose"], g["prior"], t)].append(g["gain"])
    lines = [f"| Dataset | Views | Poses | Depth prior | Scene type | Δ{metric.upper()} vs no prior | Scenes | Hurt |",
             "|---|---|---|---|---|---|---|---|"]
    for (d, n, pose, prior, t), vals in sorted(by.items()):
        lines.append(f"| {d} | {n} | {pose} | {prior} | {t} | {np.mean(vals):+.2f} | {len(vals)} | "
                     f"{sum(v < 0 for v in vals)} |")
    return "\n".join(lines)


def per_scene_table(rows: list[dict], metric: str = "psnr") -> str:
    lines = [f"| Dataset | Views | Poses | Depth prior | Scene | Δ{metric.upper()} vs no prior |", "|---|---|---|---|---|---|"]
    for g in prior_gain_rows(rows, metric):
        lines.append(f"| {g['dataset']} | {g['n_views']} | {g['pose']} | {g['prior']} | {g['scene']} | {g['gain']:+.2f} |")
    return "\n".join(lines)


def report(rows: list[dict]) -> str:
    rows = [r for r in rows if not r.get("tag")]  # ablation runs are tabulated separately
    parts = [
        "# Results", f"{len(rows)} runs.",
        "## Main table", main_table(rows),
        "## Pose leakage: full-view COLMAP poses vs honest poses", leakage_table(rows),
        "## Where does a prior help or hurt? (by scene type)", scene_type_table(rows),
        "## Per-scene gain of each prior", per_scene_table(rows),
    ]
    return "\n\n".join(parts) + "\n"
