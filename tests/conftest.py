"""Shared fixtures: tools on the import path, synthetic splats, and a 3DGS .ply writer that mirrors idol.Gaussian.save_ply."""

from __future__ import annotations

# ruff: noqa: E501  -- messages and records are kept on one line each
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sdk_module(relative: str):
    """Load one vendored SDK file without importing the ``idol`` package (whose __init__ needs torch)."""
    path = ROOT / "src" / "idol" / relative
    spec = importlib.util.spec_from_file_location(f"idol_vendored_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_splat(n: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    """A ball of Gaussians inside Idol's [-0.5, 0.5]^3 box, z-up."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    xyz = direction * rng.uniform(0.0, 0.3, size=(n, 1))
    rotation = rng.normal(size=(n, 4))
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)
    return {
        "xyz": xyz.astype(np.float32),
        "scaling": rng.uniform(0.01, 0.03, size=(n, 3)).astype(np.float32),
        "rotation": rotation.astype(np.float32),
        "opacity": rng.uniform(0.5, 1.0, size=n).astype(np.float32),
        "rgb": rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32),
    }


def write_3dgs_ply(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """The file layout of ``idol.Gaussian.save_ply`` (z-up turned y-up), built with plyfile directly."""
    from plyfile import PlyData, PlyElement

    xyz = arrays["xyz"].astype(np.float64)
    xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
    n = len(xyz)
    f_dc = (arrays["rgb"] - 0.5) / 0.28209479177387814
    eps = 1e-6
    clipped = arrays["opacity"].clip(eps, 1 - eps)
    opacity = np.log(clipped / (1 - clipped))[:, None]
    attributes = np.concatenate([xyz, np.zeros((n, 3)), f_dc, opacity, np.log(arrays["scaling"]), arrays["rotation"]], axis=1).astype(np.float32)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    elements = np.empty(n, dtype=[(name, "f4") for name in names])
    for column, name in enumerate(names):
        elements[name] = attributes[:, column]
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


@pytest.fixture
def splat() -> dict[str, np.ndarray]:
    return synthetic_splat()
