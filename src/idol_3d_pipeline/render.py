"""A small NumPy splat renderer for previews and the input-view silhouette check.

It is **not** the 3D Gaussian Splatting rasterizer: it projects each Gaussian with the EWA approximation (Zwicker et
al., 2002; Kerbl et al., 2023), adds a 0.3 px^2 low-pass, cuts each footprint at 3 sigma and at ``MAX_RADIUS`` pixels,
and composites front to back per pixel by exact depth sort, with colour from the degree-0 spherical harmonic only
(the only degree Idol predicts). It exists so the tutorial can show and check a splat without a CUDA extension; for a
faithful view, open the ``.ply`` in a Gaussian-splat viewer.

Camera convention (Idol's ``decode_camera``): the predicted camera latent is a direction ``d`` and a distance; the
SDK reports ``yaw = atan2(d_x, d_y)`` and ``pitch = asin(d_z)`` in the z-up object frame. The SDK does not document
whether ``d`` points from the object to the camera or along the view, nor the image's handedness, so ``CONVENTIONS``
enumerates the four readings and the tutorial reports the silhouette agreement of each instead of assuming one.
"""

from __future__ import annotations

# ruff: noqa: E501  -- messages and records are kept on one line each
import math
from typing import Any

import numpy as np

MAX_RADIUS = 24  # px; larger footprints are cut (documented approximation)
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
LOW_PASS = 0.3
NEAR = 1e-3
CONVENTIONS = ("toward_camera", "toward_camera_mirrored", "view_direction", "view_direction_mirrored")
DEFAULT_CONVENTION = "toward_camera"


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) unit quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def camera_pose(camera: dict[str, float], convention: str = DEFAULT_CONVENTION) -> dict[str, Any]:
    """Camera centre and the right/up/forward axes (rows) for one reading of Idol's camera, looking at the origin."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention {convention!r} is not one of {CONVENTIONS}")
    yaw, pitch, radius = float(camera["yaw"]), float(camera["pitch"]), float(camera["radius"])
    d = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    centre = radius * (d if convention.startswith("toward_camera") else -d)
    forward = -centre / np.linalg.norm(centre)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up or down
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    if convention.endswith("mirrored"):
        right = -right
    up = np.cross(right, forward)
    return {"centre": centre, "axes": np.stack([right, up, forward]), "fov": float(camera["fov"]), "convention": convention}


def orbit(camera: dict[str, float], yaw_offset: float) -> dict[str, float]:
    """The same camera turned about the object's vertical (z) axis by ``yaw_offset`` radians."""
    return {**camera, "yaw": float(camera["yaw"]) + yaw_offset}


def render(arrays: dict[str, np.ndarray], camera: dict[str, float], *, size: int = 384, convention: str = DEFAULT_CONVENTION, background: float = 1.0, max_pairs: int = 12_000_000) -> dict[str, Any]:
    """Render Gaussians to an RGB image and an accumulated-opacity map.

    ``arrays``: ``xyz`` (N, 3), ``scaling`` (N, 3), ``rotation`` (N, 4, w first), ``opacity`` (N,), ``rgb`` (N, 3) in
    [0, 1]. Returns ``image`` (size, size, 3) uint8, ``alpha`` (size, size) float32, and counts for the record."""
    pose = camera_pose(camera, convention)
    xyz = np.asarray(arrays["xyz"], dtype=np.float64)
    axes = pose["axes"]
    cam = (xyz - pose["centre"]) @ axes.T  # x right, y up, z forward (depth)
    focal = (size / 2.0) / math.tan(math.radians(pose["fov"]) / 2.0)
    keep = cam[:, 2] > NEAR
    cam = cam[keep]
    scales = np.asarray(arrays["scaling"], dtype=np.float64)[keep]
    rot = quaternion_to_matrix(np.asarray(arrays["rotation"], dtype=np.float64)[keep])
    opacity = np.asarray(arrays["opacity"], dtype=np.float64).reshape(-1)[keep]
    rgb = np.asarray(arrays["rgb"], dtype=np.float64)[keep]
    accumulated = np.zeros(size * size)
    colour = np.zeros((size * size, 3))
    record = {"gaussians": int(len(xyz)), "in_front": int(keep.sum()), "pairs": 0, "cut_at_max_radius": 0, "size": size, "convention": convention}
    if len(cam):
        # 3D covariance in camera axes, then the EWA Jacobian of the perspective projection (v grows downwards)
        rs = rot * scales[:, None, :]
        cov_world = rs @ rs.transpose(0, 2, 1)
        cov_cam = axes @ cov_world @ axes.T
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        jac = np.zeros((len(cam), 2, 3))
        jac[:, 0, 0] = focal / z
        jac[:, 0, 2] = -focal * x / z**2
        jac[:, 1, 1] = -focal / z
        jac[:, 1, 2] = focal * y / z**2
        cov2 = jac @ cov_cam @ jac.transpose(0, 2, 1)
        a, b, c = cov2[:, 0, 0] + LOW_PASS, cov2[:, 0, 1], cov2[:, 1, 1] + LOW_PASS
        det = a * c - b * b
        valid = det > 1e-12
        u = size / 2.0 + focal * x / z
        v = size / 2.0 - focal * y / z
        half_trace = (a + c) / 2.0
        lam = half_trace + np.sqrt(np.maximum(half_trace**2 - det, 0.0))
        radius = np.ceil(3.0 * np.sqrt(lam)).astype(np.int64)
        record["cut_at_max_radius"] = int(((radius > MAX_RADIUS) & valid).sum())
        radius = np.clip(radius, 0, MAX_RADIUS)
        on_screen = valid & (u + radius >= 0) & (u - radius < size) & (v + radius >= 0) & (v - radius < size)
        inv_a, inv_b, inv_c = c / np.where(valid, det, 1.0), -b / np.where(valid, det, 1.0), a / np.where(valid, det, 1.0)
        pix_parts, depth_parts, alpha_parts, index_parts = [], [], [], []
        for r in np.unique(radius[on_screen]):
            members = np.flatnonzero(on_screen & (radius == r))
            offsets = np.arange(-r, r + 1)
            ox, oy = np.meshgrid(offsets, offsets)
            ox, oy = ox.ravel(), oy.ravel()
            chunk = max(1, max_pairs // max(1, len(ox)))
            for start in range(0, len(members), chunk):
                g = members[start : start + chunk]
                px = np.floor(u[g])[:, None].astype(np.int64) + ox[None, :]
                py = np.floor(v[g])[:, None].astype(np.int64) + oy[None, :]
                dx = px + 0.5 - u[g][:, None]
                dy = py + 0.5 - v[g][:, None]
                power = -0.5 * (inv_a[g][:, None] * dx * dx + 2 * inv_b[g][:, None] * dx * dy + inv_c[g][:, None] * dy * dy)
                alpha = np.minimum(ALPHA_MAX, opacity[g][:, None] * np.exp(np.minimum(power, 0.0)))
                ok = (alpha >= ALPHA_MIN) & (px >= 0) & (px < size) & (py >= 0) & (py < size)
                pix_parts.append((py * size + px)[ok])
                alpha_parts.append(alpha[ok])
                depth_parts.append(np.broadcast_to(z[g][:, None], px.shape)[ok])
                index_parts.append(np.broadcast_to(g[:, None], px.shape)[ok])
        if pix_parts:
            pix = np.concatenate(pix_parts)
            alpha = np.concatenate(alpha_parts)
            depth = np.concatenate(depth_parts)
            index = np.concatenate(index_parts)
            order = np.lexsort((depth, pix))  # per pixel, nearest first
            pix, alpha, index = pix[order], alpha[order], index[order]
            log_t = np.log1p(-alpha)
            inclusive = np.cumsum(log_t)
            starts = np.flatnonzero(np.r_[True, pix[1:] != pix[:-1]])
            base = np.where(starts > 0, inclusive[np.maximum(starts - 1, 0)], 0.0)
            seg = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(pix)]))
            transmittance = np.exp(inclusive - log_t - base[seg])  # light that reaches this Gaussian
            weight = alpha * transmittance
            accumulated = np.bincount(pix, weight, minlength=size * size)
            for ch in range(3):
                colour[:, ch] = np.bincount(pix, weight * rgb[index, ch], minlength=size * size)
            record["pairs"] = int(len(pix))
    accumulated = np.clip(accumulated, 0.0, 1.0)
    image = colour + (1.0 - accumulated)[:, None] * background
    image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(size, size, 3)
    return {"image": image, "alpha": accumulated.reshape(size, size).astype(np.float32), **record}
