"""Checks a generated splat can be held to without ground-truth 3D: structure, file integrity, and input-view agreement.

Idol predicts the sides the picture never showed, so there is no reference to score them against here. What can be
checked is (1) that the Gaussians are well formed and inside Idol's documented bounds, (2) that the exported ``.ply``
reads back to the same Gaussians, and (3) how well the splat's silhouette, rendered from the camera Idol predicts,
covers the object mask of the image Idol saw (intersection over union). (3) measures the visible side only and uses
the approximate renderer in ``render.py``; it is a consistency check, not a quality score for the unseen sides.
"""

from __future__ import annotations

# ruff: noqa: E501  -- messages and records are kept on one line each
from pathlib import Path
from typing import Any

import numpy as np

BOUND = 0.5  # Idol: the object is inside [-0.5, 0.5]^3 (z-up)
BOUND_TOLERANCE = 0.05  # Gaussian centres may sit slightly outside after the decoder's offsets


def silhouette_iou(rendered_alpha: np.ndarray, mask: np.ndarray, *, threshold: float = 0.5) -> float:
    """IoU of two binary silhouettes after thresholding; the mask is resized to the render's size first."""
    rendered = np.asarray(rendered_alpha) >= threshold
    reference = np.asarray(mask, dtype=np.float32)
    if reference.shape != rendered.shape:
        from PIL import Image

        reference = np.asarray(Image.fromarray((reference * 255).astype(np.uint8)).resize(rendered.shape[::-1], Image.BILINEAR), dtype=np.float32) / 255.0
    reference = reference >= threshold
    union = np.logical_or(rendered, reference).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(rendered, reference).sum() / union)


def splat_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Count, finiteness, bounds and the spread of opacity and scale; ``problems`` lists anything out of contract."""
    xyz, scaling, opacity, rotation = (np.asarray(arrays[k]) for k in ("xyz", "scaling", "opacity", "rotation"))
    problems = []
    finite = all(np.isfinite(np.asarray(arrays[k])).all() for k in ("xyz", "scaling", "opacity", "rotation", "rgb"))
    if not finite:
        problems.append("non-finite values")
    outside = int((np.abs(xyz) > BOUND + BOUND_TOLERANCE).any(axis=1).sum())
    if outside:
        problems.append(f"{outside} centres outside [-0.55, 0.55]^3")
    norms = np.linalg.norm(rotation, axis=1)
    if np.abs(norms - 1).max(initial=0.0) > 1e-3:
        problems.append("rotations are not unit quaternions")
    if (opacity < 0).any() or (opacity > 1).any():
        problems.append("opacity outside [0, 1]")

    def quantiles(values: np.ndarray) -> dict[str, float]:
        q = np.quantile(values, [0.05, 0.5, 0.95]) if values.size else [float("nan")] * 3
        return {"p05": round(float(q[0]), 5), "median": round(float(q[1]), 5), "p95": round(float(q[2]), 5)}

    return {
        "gaussians": int(len(xyz)),
        "finite": bool(finite),
        "extent": [[round(float(v), 4) for v in xyz.min(axis=0)], [round(float(v), 4) for v in xyz.max(axis=0)]] if len(xyz) else None,
        "outside_bounds": outside,
        "opacity": quantiles(opacity.reshape(-1)),
        "largest_scale": quantiles(scaling.max(axis=1)),
        "problems": problems,
        "verdict": "well-formed" if not problems else "out of contract",
    }


PLY_FIELDS = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


def read_ply(path: str | Path) -> dict[str, Any]:
    """Read a 3DGS ``.ply`` written by ``idol.Gaussian.save_ply``: vertex count, field names, finiteness."""
    from plyfile import PlyData

    data = PlyData.read(str(path))
    vertex = data["vertex"].data
    names = tuple(vertex.dtype.names)
    values = np.stack([np.asarray(vertex[name], dtype=np.float64) for name in names], axis=1) if len(vertex) else np.zeros((0, len(names)))
    return {"vertices": int(len(vertex)), "fields": list(names), "standard_fields": names == PLY_FIELDS, "finite": bool(np.isfinite(values).all()), "values": values}


def ply_roundtrip(path: str | Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """The exported file holds the same Gaussians: count, standard 3DGS fields, finite values, and the y-up positions.

    ``save_ply`` turns the z-up object to the y-up convention viewers use: (x, y, z) -> (x, -z, y)."""
    ply = read_ply(path)
    xyz = np.asarray(arrays["xyz"], dtype=np.float64)
    expected = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
    positions = ply["values"][:, :3] if ply["vertices"] else np.zeros((0, 3))
    same_count = ply["vertices"] == len(xyz)
    max_error = float(np.abs(positions - expected).max(initial=0.0)) if same_count else float("inf")
    ok = same_count and ply["standard_fields"] and ply["finite"] and max_error < 1e-5
    return {"vertices": ply["vertices"], "standard_fields": ply["standard_fields"], "finite": ply["finite"], "position_max_abs_error": max_error, "verified": bool(ok)}
