"""Cross-check the renderer's maths against the independent Lean implementation (``lean/Idol3d/Reference/Float.lean``).

Doubles cross the process boundary as raw 64-bit patterns. Skipped unless the executable is built:
``cd lean && lake build idol3d-ref`` (see ``docs/LEAN.md``).
"""

from __future__ import annotations

import math
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from idol_3d_pipeline import CONVENTIONS, camera_pose, render
from idol_3d_pipeline.render import quaternion_to_matrix

EXE = Path(__file__).resolve().parents[1] / "lean" / ".lake" / "build" / "bin" / "idol3d-ref"
pytestmark = pytest.mark.skipif(not EXE.exists(), reason="Lean reference not built (cd lean && lake build idol3d-ref)")
N = 300
TIGHT = {"rtol": 1e-12, "atol": 1e-12}


def _bits(x: float) -> str:
    return str(struct.unpack("<Q", struct.pack("<d", float(x)))[0])


def _float(b: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(b)))[0]


def lean(fn: str, rows: list[list[float]]) -> list[np.ndarray]:
    text = "".join(f"{fn} {' '.join(_bits(v) for v in row)}\n" for row in rows)
    out = subprocess.run([str(EXE)], input=text, capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(rows) and not any(line.startswith("error") for line in out), out[:3]
    return [np.array([_float(b) for b in line.split()]) for line in out]


def _rng():
    return np.random.default_rng(20261113)


def test_camera_pose_all_conventions():
    rng = _rng()
    rows = [[rng.uniform(-7, 7), rng.uniform(-1.55, 1.55), rng.uniform(0.3, 30), float(i % 4)] for i in range(N)]
    rows += [[0.3, math.pi / 2, 2.0, float(c)] for c in range(4)]  # straight overhead: the [1, 0, 0] fallback
    for row, got in zip(rows, lean("camera_pose", rows), strict=True):
        pose = camera_pose({"yaw": row[0], "pitch": row[1], "radius": row[2], "fov": 40.0}, CONVENTIONS[int(row[3])])
        np.testing.assert_allclose(got, np.concatenate([pose["centre"], pose["axes"].ravel()]), **TIGHT)


def test_quaternion_to_matrix():
    rng = _rng()
    rows = [list(rng.normal(size=4) * rng.uniform(0.01, 10)) for _ in range(N)] + [[0.0, 0.0, 0.0, 0.0]]
    for row, got in zip(rows, lean("quaternion_to_matrix", rows), strict=True):
        np.testing.assert_allclose(got, quaternion_to_matrix(np.array([row]))[0].ravel(), **TIGHT)


def test_sdk_decode_round_trip():
    """``decode_camera``'s yaw and pitch, then ``camera_pose``'s direction: the decoded direction comes back."""
    rng = _rng()
    rows = [list(rng.normal(size=3)) for _ in range(N)] + [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [1e-9, 0.0, 0.0]]
    for row, got in zip(rows, lean("decode_then_direction", rows), strict=True):
        v = np.array(row)
        d = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([0.0, 0.0, 1.0])
        yaw, pitch = math.atan2(d[0], d[1]), math.asin(float(np.clip(d[2], -1, 1)))
        np.testing.assert_allclose(got[:2], [yaw, pitch], **TIGHT)
        np.testing.assert_allclose(got[2:], camera_pose({"yaw": yaw, "pitch": pitch, "radius": 1.0, "fov": 40.0})["centre"],
                                   atol=1e-12)
        np.testing.assert_allclose(got[2:], d, atol=1e-9)


def test_compositing_matches_render():
    """Stacked Gaussians at one pixel: Lean's front-to-back compositing of their individual alphas against ``render``."""
    rng = _rng()
    cam, size = {"yaw": 0.0, "pitch": 0.0, "radius": 3.0, "fov": 30.0}, 16
    for _ in range(40):
        k = int(rng.integers(1, 7))
        ys = np.linspace(0.4, -0.4, k)
        opacities, reds = rng.uniform(0.05, 0.95, k), rng.uniform(0, 1, k)
        blobs = [{"xyz": np.array([[0.0, y, 0.0]]), "scaling": np.full((1, 3), 0.2), "rotation": np.array([[1.0, 0, 0, 0]]),
                  "opacity": np.array([o]), "rgb": np.array([[r, 0, 0]])} for y, o, r in zip(ys, opacities, reds, strict=True)]
        alphas = [float(render(b, cam, size=size)["alpha"][8, 8]) for b in blobs]
        both = render({key: np.concatenate([b[key] for b in blobs]) for key in blobs[0]}, cam, size=size, background=1.0)
        ((acc, red),) = lean("composite", [alphas + list(reds) + [1.0]])
        assert both["alpha"][8, 8] == pytest.approx(acc, abs=1e-6)
        assert both["image"][8, 8][0] == pytest.approx(round(red * 255), abs=1)
