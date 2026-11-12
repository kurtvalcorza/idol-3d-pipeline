#!/usr/bin/env python3
"""Fit the input-view camera calibration in ``render.py`` (field-of-view scale, yaw and pitch offsets) by leave-one-out.

The render stage scores each splat by the silhouette IoU between its render from Idol's predicted camera and the mask
Idol saw. Idol does not document how its camera maps to the input image, so ``render.CALIBRATION`` holds one fixed
reading for every splat. Fitting that reading on the same splats it is then scored on overstates it (the 2026-11-10
fit scored 0.87 in-sample and 0.48-0.73 on the first new splats), so this tool reports **held-out** scores:

1. render every splat at every grid point once (``toward_camera`` convention, ``CONVENTION_SIZE`` px, as the stage);
2. for each splat, choose the grid point with the best mean IoU over the *other* splats and score it on this one;
3. choose the final calibration on all splats, and accept it only if every held-out score beats that splat's
   uncalibrated reading.

Input: a directory holding the render stage's ``state/<id>.npz`` (arrays + float mask) and ``outputs/<id>/<id>_camera.json``
for each sample. Output: a JSON record (grid, held-out scores, choice) on stdout and, with ``--out``, in a file plus
``<out>.tables.json`` with every IoU.

Usage (from the repository root, NumPy + Pillow only):
    python tools/fit_camera_calibration.py <run-root> --out fit.json
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idol_3d_pipeline.metrics import silhouette_iou  # noqa: E402
from idol_3d_pipeline.render import DEFAULT_CONVENTION, render  # noqa: E402

SIZE = 256  # tutorial_stages.CONVENTION_SIZE
GRID = {"fov_scale": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], "yaw_offset_deg": list(range(-60, 61, 10)), "pitch_offset_deg": [-10, 0, 10]}


def load(run_root: Path, name: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    stored = np.load(run_root / "state" / f"{name}.npz")
    arrays = {k: stored[k] for k in ("xyz", "scaling", "rotation", "opacity", "rgb")}
    camera = json.loads((run_root / "outputs" / name / f"{name}_camera.json").read_text(encoding="utf-8"))
    return arrays, stored["mask"], camera


def reading(camera: dict[str, float], fov_scale: float, yaw_deg: float, pitch_deg: float) -> dict[str, float]:
    return {**camera, "fov": float(camera["fov"]) * fov_scale, "yaw": float(camera["yaw"]) + math.radians(yaw_deg), "pitch": float(camera["pitch"]) + math.radians(pitch_deg)}


def key_of(fov_scale: float, yaw_deg: float, pitch_deg: float) -> str:
    return f"{float(fov_scale):g}|{float(yaw_deg):g}|{float(pitch_deg):g}"  # %g keeps 0.725 distinct from 0.72


def score_splat(args: tuple[str, str, dict[str, list[float]]]) -> tuple[str, dict[str, float], float]:
    run_root, name, grid = Path(args[0]), args[1], args[2]
    arrays, mask, camera = load(run_root, name)
    table = {}
    for f, y, p in itertools.product(grid["fov_scale"], grid["yaw_offset_deg"], grid["pitch_offset_deg"]):
        table[key_of(f, y, p)] = silhouette_iou(render(arrays, reading(camera, f, y, p), size=SIZE, convention=DEFAULT_CONVENTION)["alpha"], mask)
    uncalibrated = silhouette_iou(render(arrays, camera, size=SIZE, convention=DEFAULT_CONVENTION)["alpha"], mask)
    return name, table, uncalibrated


def best(tables: dict[str, dict[str, float]], names: list[str]) -> tuple[str, float]:
    """Best mean IoU over ``names``; ties go to the smaller yaw, then pitch, offset."""
    keys = next(iter(tables.values())).keys()
    means = {k: float(np.mean([tables[n][k] for n in names])) for k in keys}
    key = max(means, key=lambda k: (means[k], -abs(float(k.split("|")[1])), -abs(float(k.split("|")[2]))))
    return key, means[key]


def as_calibration(key: str) -> dict[str, float]:
    f, y, p = key.split("|")
    return {"fov_scale": float(f), "yaw_offset_deg": float(y), "pitch_offset_deg": float(p)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--samples", nargs="+", default=["backpack", "armchair", "porcini"])
    parser.add_argument("--current", nargs=3, type=float, default=[0.6, 25, 0], metavar=("FOV_SCALE", "YAW_DEG", "PITCH_DEG"), help="the calibration in render.py now")
    parser.add_argument("--fov", nargs="+", type=float, default=GRID["fov_scale"], help="field-of-view scales to try")
    parser.add_argument("--yaw", nargs="+", type=float, default=GRID["yaw_offset_deg"], help="yaw offsets (degrees) to try")
    parser.add_argument("--pitch", nargs="+", type=float, default=GRID["pitch_offset_deg"], help="pitch offsets (degrees) to try")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path)
    a = parser.parse_args(argv)
    grid = {"fov_scale": a.fov, "yaw_offset_deg": a.yaw, "pitch_offset_deg": a.pitch}
    if a.current not in [[f, y, p] for f, y, p in itertools.product(a.fov, a.yaw, a.pitch)]:
        grid = {"fov_scale": sorted({*a.fov, a.current[0]}), "yaw_offset_deg": sorted({*a.yaw, a.current[1]}), "pitch_offset_deg": sorted({*a.pitch, a.current[2]})}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(score_splat, [(str(a.run_root), n, grid) for n in a.samples]))
    tables = {n: t for n, t, _ in results}
    uncalibrated = {n: round(u, 4) for n, _, u in results}
    current = key_of(*a.current)
    held_out = {}
    for n in a.samples:
        others = [m for m in a.samples if m != n]
        key, train_mean = best(tables, others)
        held_out[n] = {"fitted_on": others, "calibration": as_calibration(key), "train_mean_iou": round(train_mean, 4), "held_out_iou": round(tables[n][key], 4), "uncalibrated_iou": uncalibrated[n], "beats_uncalibrated": tables[n][key] > uncalibrated[n]}
    final_key, final_mean = best(tables, a.samples)
    record = {
        "grid": {**grid, "size": SIZE, "convention": DEFAULT_CONVENTION, "points": len(tables[a.samples[0]])},
        "samples": a.samples,
        "uncalibrated_iou": uncalibrated,
        "current_calibration": {"calibration": as_calibration(current), "iou": {n: round(tables[n][current], 4) for n in a.samples}},
        "leave_one_out": held_out,
        "held_out_mean_iou": round(float(np.mean([h["held_out_iou"] for h in held_out.values()])), 4),
        "final": {"calibration": as_calibration(final_key), "in_sample_iou": {n: round(tables[n][final_key], 4) for n in a.samples}, "in_sample_mean_iou": round(final_mean, 4)},
        "accepted": all(h["beats_uncalibrated"] for h in held_out.values()),
    }
    if a.out:
        a.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        a.out.with_suffix(".tables.json").write_text(json.dumps({n: {k: round(v, 4) for k, v in t.items()} for n, t in tables.items()}) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("uncalibrated_iou", "current_calibration", "leave_one_out", "held_out_mean_iou", "final", "accepted")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
