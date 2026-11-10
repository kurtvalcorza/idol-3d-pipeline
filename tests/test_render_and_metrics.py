"""The NumPy splat renderer, the structural checks and the .ply read-back."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import synthetic_splat, write_3dgs_ply
from idol_3d_pipeline import (
    CALIBRATION,
    CONVENTIONS,
    calibrated,
    camera_pose,
    orbit,
    ply_roundtrip,
    render,
    silhouette_iou,
    splat_summary,
)

CAMERA = {"yaw": 0.0, "pitch": 0.0, "radius": 2.0, "fov": 40.0}


def one_gaussian(xyz=(0.0, 0.0, 0.0), rgb=(1.0, 0.0, 0.0), opacity=0.9, scale=0.05):
    return {
        "xyz": np.array([xyz], np.float32),
        "scaling": np.full((1, 3), scale, np.float32),
        "rotation": np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        "opacity": np.array([opacity], np.float32),
        "rgb": np.array([rgb], np.float32),
    }


def test_a_gaussian_at_the_origin_renders_at_the_image_centre_and_nowhere_near_the_corner():
    out = render(one_gaussian(), CAMERA, size=64)
    assert out["alpha"][32, 32] > 0.8
    assert out["alpha"][0, 0] == 0.0
    assert tuple(out["image"][0, 0]) == (255, 255, 255)
    assert out["image"][32, 32][0] > 200 and out["image"][32, 32][1] < 80


def test_nearer_gaussians_hide_farther_ones():
    a, b = one_gaussian((0, 0.3, 0), (1, 0, 0), 0.99), one_gaussian((0, -0.3, 0), (0, 0, 1), 0.99)
    both = {k: np.concatenate([b[k], a[k]]) for k in a}  # input order must not matter
    centre = render(both, CAMERA, size=64)["image"][32, 32]
    assert centre[0] > 200 and centre[2] < 40, "the camera sits on +y (yaw 0), so the red Gaussian at y=+0.3 is in front"


def test_alpha_is_bounded_and_empty_scenes_are_background():
    out = render(synthetic_splat(), CAMERA, size=96)
    assert out["alpha"].min() >= 0.0 and out["alpha"].max() <= 1.0
    empty = {k: v[:0] for k, v in one_gaussian().items()}
    blank = render(empty, CAMERA, size=16)
    assert blank["alpha"].max() == 0.0 and (blank["image"] == 255).all()


def test_gaussians_behind_the_camera_are_dropped():
    out = render(one_gaussian((0, 5.0, 0)), CAMERA, size=32)
    assert out["in_front"] == 0 and out["alpha"].max() == 0.0


def test_conventions_differ_and_orbit_turns_about_z():
    poses = {c: camera_pose(CAMERA, c) for c in CONVENTIONS}
    assert np.allclose(poses["toward_camera"]["centre"], [0, 2, 0])
    assert np.allclose(poses["view_direction"]["centre"], [0, -2, 0])
    assert np.allclose(poses["toward_camera_mirrored"]["axes"][0], -poses["toward_camera"]["axes"][0])
    turned = camera_pose(orbit(CAMERA, math.pi / 2), "toward_camera")["centre"]
    assert np.allclose(turned, [2, 0, 0], atol=1e-9)
    with pytest.raises(ValueError):
        camera_pose(CAMERA, "sideways")


def test_silhouette_iou():
    a = np.zeros((10, 10))
    a[:5] = 1
    assert silhouette_iou(a, a) == 1.0
    assert silhouette_iou(a, 1 - a) == 0.0
    assert math.isnan(silhouette_iou(np.zeros((4, 4)), np.zeros((4, 4))))
    assert silhouette_iou(a, np.kron(a, np.ones((2, 2)))) == pytest.approx(1.0)  # mask resized to the render first


def test_splat_summary_flags_out_of_contract_values(splat):
    assert splat_summary(splat)["verdict"] == "well-formed"
    broken = dict(splat)
    broken["xyz"] = splat["xyz"].copy()
    broken["xyz"][0] = [0.9, 0, 0]
    broken["opacity"] = splat["opacity"].copy()
    broken["opacity"][1] = float("nan")
    summary = splat_summary(broken)
    assert summary["verdict"] == "out of contract"
    assert summary["outside_bounds"] == 1 and not summary["finite"]


def test_ply_roundtrip_verifies_the_exported_positions(tmp_path, splat):
    path = tmp_path / "splat.ply"
    write_3dgs_ply(path, splat)
    record = ply_roundtrip(path, splat)
    assert record["verified"] and record["vertices"] == len(splat["xyz"]) and record["standard_fields"]
    moved = dict(splat)
    moved["xyz"] = splat["xyz"] + 0.01
    assert not ply_roundtrip(path, moved)["verified"]


def test_calibration_scales_fov_and_turns_yaw_only():
    camera = {"yaw": 1.0, "pitch": 0.2, "radius": 3.0, "fov": 40.0}
    c = calibrated(camera)
    assert c["fov"] == pytest.approx(40.0 * CALIBRATION["fov_scale"])
    assert c["yaw"] == pytest.approx(1.0 + math.radians(CALIBRATION["yaw_offset_deg"]))
    assert (c["pitch"], c["radius"]) == (0.2, 3.0)
    assert camera["fov"] == 40.0, "the predicted camera is not modified in place"
