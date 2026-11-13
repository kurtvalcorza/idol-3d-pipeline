"""Property tests: the theorems proved in ``lean/Idol3d/`` restated against the Python they formalise.

Each test names the theorems it checks (``lean:`` in the docstring); ``docs/LEAN.md`` has the map. The proofs are
over exact reals; these check the floating-point code has the same properties. Skips without Hypothesis.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from conftest import load_sdk_module, write_3dgs_ply
from idol_3d_pipeline import CONVENTIONS, camera_pose, orbit, render, silhouette_iou
from idol_3d_pipeline.metrics import ply_roundtrip, read_ply
from idol_3d_pipeline.render import quaternion_to_matrix

settings.register_profile("idol3d", max_examples=150, deadline=None)
settings.load_profile("idol3d")

angles = st.floats(-4 * math.pi, 4 * math.pi, allow_nan=False)
pitches = st.floats(-1.5, 1.5, allow_nan=False)  # cos(pitch) > 1e-6: away from straight overhead
radii = st.floats(0.5, 20, allow_nan=False)
coords = st.floats(-10, 10, allow_nan=False)
conventions = st.sampled_from(CONVENTIONS)


def rot_z(phi: float) -> np.ndarray:
    return np.array([[math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0], [0, 0, 1]])


def direction(yaw: float, pitch: float) -> np.ndarray:
    return np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])


# ------------------------------------------------------------------------------------------------- Camera.lean


@given(st.tuples(coords, coords, coords).map(np.array).filter(lambda v: np.linalg.norm(v) > 1e-3))
@example(np.array([0.0, 5.960464477539063e-08, 3.0]))  # near the pole, where the float round trip is worst
def test_renderer_rebuilds_the_sdk_direction(v):
    """lean: direction_decode, direction_unit — the yaw and pitch ``decode_camera`` reports
    (``atan2(dx, dy)``, ``asin(dz)``) put the camera back on the decoded direction.

    Floating point: near the pole ``dz`` is within an ulp of ±1, and ``cos(asin(dz))`` loses the horizontal part up to
    about √(2·eps) ≈ 2e-8, hence the tolerance (exact over the reals)."""
    d = v / np.linalg.norm(v)
    yaw, pitch = math.atan2(d[0], d[1]), math.asin(float(np.clip(d[2], -1, 1)))
    pose = camera_pose({"yaw": yaw, "pitch": pitch, "radius": 1.0, "fov": 40.0}, "toward_camera")
    np.testing.assert_allclose(pose["centre"], d, atol=5e-8)


@given(st.tuples(coords, coords, coords).map(np.array).filter(lambda v: np.linalg.norm(v) > 1e-2),
       st.floats(0.05, 2.0), st.floats(0.02, 0.5))
def test_sdk_decode_camera_round_trip(v, inverse_distance, half_width):
    """lean: direction_decode, halfWidth_to_edge — the vendored SDK's own ``decode_camera`` reports
    ``yaw = atan2(dx, dy)``, ``pitch = asin(dz)`` of its (float32) unit direction, which ``camera_pose`` turns back
    into that direction, and ``fov = 2 atan(w / radius)``. Skips without torch (the SDK is hash-pinned in
    test_contract_and_sdk.py). The SDK works in float32, so near the pole the round trip loses up to √(2·eps32) ≈ 5e-4."""
    torch = pytest.importorskip("torch")
    camera = load_sdk_module("camera.py")
    latent = torch.tensor([*v, inverse_distance, half_width], dtype=torch.float32)
    decoded = camera.decode_camera(latent)
    d32 = (latent[:3] / torch.norm(latent[:3])).tolist()
    assert decoded["yaw"] == pytest.approx(math.atan2(d32[0], d32[1]), abs=1e-6)
    assert decoded["pitch"] == pytest.approx(math.asin(max(-1.0, min(1.0, d32[2]))), abs=1e-6)
    pose = camera_pose({**decoded, "radius": 1.0}, "toward_camera")
    np.testing.assert_allclose(pose["centre"], v / np.linalg.norm(v), atol=1e-3)
    assert math.tan(math.radians(decoded["fov"]) / 2) == pytest.approx(half_width / decoded["radius"], rel=1e-5)


@given(angles, pitches, radii, conventions)
def test_camera_axes(yaw, pitch, radius, convention):
    """lean: len_centre, forward_eq, cameraAxes_eq, cameraAxes_orthogonal, det_cameraAxes, up_z_plain — the camera is
    at distance ``radius``, looks at the origin, its axes are orthonormal and left-handed (det -1) in every convention,
    and the plain conventions are upright."""
    pose = camera_pose({"yaw": yaw, "pitch": pitch, "radius": radius, "fov": 40.0}, convention)
    axes, centre = pose["axes"], pose["centre"]
    assert np.linalg.norm(centre) == pytest.approx(radius, rel=1e-9)
    side = 1.0 if convention.startswith("toward_camera") else -1.0
    np.testing.assert_allclose(axes[2], -side * direction(yaw, pitch), atol=1e-9)
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(axes) == pytest.approx(-1.0, abs=1e-9)
    if not convention.endswith("mirrored"):
        assert axes[1, 2] == pytest.approx(math.cos(pitch), abs=1e-9)


@given(angles, pitches, radii, st.booleans())
def test_mirrored_is_upside_down(yaw, pitch, radius, toward):
    """lean: cameraAxes_mirrored — a ``*_mirrored`` reading is the plain one turned 180° about the view axis
    (``diag(-1, -1, 1)``): the picture is upside down, not mirrored."""
    plain, mirrored = ("toward_camera", "toward_camera_mirrored") if toward else ("view_direction", "view_direction_mirrored")
    cam = {"yaw": yaw, "pitch": pitch, "radius": radius, "fov": 40.0}
    flipped = np.diag([-1, -1, 1]) @ camera_pose(cam, plain)["axes"]
    np.testing.assert_allclose(camera_pose(cam, mirrored)["axes"], flipped, atol=1e-12)


@given(angles, angles, pitches, radii, conventions)
def test_orbit_turns_clockwise(yaw, theta, pitch, radius, convention):
    """lean: centre_orbit — ``orbit(camera, θ)`` puts the camera at the old centre turned by -θ about z."""
    cam = {"yaw": yaw, "pitch": pitch, "radius": radius, "fov": 40.0}
    np.testing.assert_allclose(
        camera_pose(orbit(cam, theta), convention)["centre"], rot_z(-theta) @ camera_pose(cam, convention)["centre"], atol=1e-9
    )


@given(st.floats(0.05, 0.4), st.floats(1.5, 6.0))
def test_half_width_reaches_the_image_edge(half_width, radius):
    """lean: halfWidth_to_edge — with the SDK's ``fov = 2 atan(w / radius)``, a point at the object's distance and
    ``w/2`` to the right lands three quarters of the way across the image (``w`` would reach the edge)."""
    size = 64
    fov = math.degrees(2 * math.atan(half_width / radius))
    cam = {"yaw": 0.0, "pitch": 0.0, "radius": radius, "fov": fov}  # camera on +y looking at the origin; right is -x
    blob = {"xyz": np.array([[-half_width / 2, 0.0, 0.0]]), "scaling": np.full((1, 3), 1e-4),
            "rotation": np.array([[1.0, 0, 0, 0]]), "opacity": np.array([0.99]), "rgb": np.zeros((1, 3))}
    alpha = render(blob, cam, size=size)["alpha"]
    row, col = np.unravel_index(np.argmax(alpha), alpha.shape)
    assert abs(col + 0.5 - 0.75 * size) <= 1.0 and abs(row + 0.5 - size / 2) <= 1.0


# ------------------------------------------------------------------------------------------------- Quaternion.lean


@given(st.tuples(coords, coords, coords, coords).map(np.array).filter(lambda q: np.linalg.norm(q) > 1e-3))
def test_quaternion_to_matrix_is_a_rotation(q):
    """lean: quaternion_to_matrix_rotation, quatMatrix_eq_H — a proper rotation for any non-zero quaternion."""
    r = quaternion_to_matrix(q[None])[0]
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------------------------------------- Ewa.lean


def _footprint(xyz, scaling, quat, cam, size):
    """The projected 2-D covariance (with the low-pass) and centre of one Gaussian, from the EWA formulas."""
    pose = camera_pose(cam)
    axes = pose["axes"]
    x, y, z = (np.asarray(xyz) - pose["centre"]) @ axes.T
    focal = (size / 2) / math.tan(math.radians(cam["fov"]) / 2)
    rs = quaternion_to_matrix(np.asarray(quat, float)[None])[0] * np.asarray(scaling)[None, :]
    jac = np.array([[focal / z, 0, -focal * x / z**2], [0, -focal / z, focal * y / z**2]])
    cov2 = jac @ axes @ rs @ rs.T @ axes.T @ jac.T + 0.3 * np.eye(2)
    return cov2, size / 2 + focal * x / z, size / 2 - focal * y / z


@given(st.tuples(st.floats(-0.2, 0.2), st.floats(-0.2, 0.2), st.floats(-0.2, 0.2)),
       st.tuples(st.floats(0.005, 0.05), st.floats(0.005, 0.05), st.floats(0.005, 0.05)),
       st.tuples(coords, coords, coords, coords).filter(lambda q: np.linalg.norm(q) > 1e-3))
def test_footprint_covers_the_three_sigma_ellipse(xyz, scaling, quat):
    """lean: cov2_gram, gram2_psd, det_lowpass_ge, valid_always, inverse_correct, lamMax_ge, lamMax_char,
    three_sigma_cover, pixel_enumerated — every pixel whose centre is within 3σ of the projected Gaussian, and whose
    alpha clears 1/255, is drawn with exactly the Gaussian's alpha; nothing is drawn outside the footprint."""
    size = 48
    cam = {"yaw": 0.3, "pitch": 0.2, "radius": 2.0, "fov": 40.0}
    cov2, u, v = _footprint(xyz, scaling, quat, cam, size)
    assert np.linalg.det(cov2) >= 0.09 - 1e-9  # valid_always
    lam = np.linalg.eigvalsh(cov2).max()
    radius = math.ceil(3 * math.sqrt(lam))
    assume(radius <= 24 and 0 <= u < size and 0 <= v < size)  # MAX_RADIUS not reached, centre on screen
    opacity = 1.0  # at the ALPHA_MAX clamp
    out = render({"xyz": np.array([xyz]), "scaling": np.array([scaling]), "rotation": np.array([quat], float),
                  "opacity": np.array([opacity]), "rgb": np.zeros((1, 3))}, cam, size=size)["alpha"]
    inv = np.linalg.inv(cov2)
    py, px = np.mgrid[0:size, 0:size]
    dx, dy = px + 0.5 - u, py + 0.5 - v
    maha = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    expected = np.minimum(0.99, opacity * np.exp(-0.5 * maha))
    inside = (maha <= 9) & (expected >= 1 / 255)
    np.testing.assert_allclose(out[inside], expected[inside], rtol=1e-4, atol=1e-6)
    far = (np.abs(px - math.floor(u)) > radius) | (np.abs(py - math.floor(v)) > radius)
    assert (out[far] == 0).all()


# ------------------------------------------------------------------------------------------------- Composite.lean


@given(st.lists(st.floats(0.05, 0.95), min_size=1, max_size=6), st.lists(st.floats(0, 1), min_size=6, max_size=6))
def test_front_to_back_compositing(opacities, reds):
    """lean: sum_weights, exp_sum_log, transmittance_mem, accumulated_mem, image_mem — Gaussians stacked along the
    view axis composite to ``1 − Π(1 − αₖ)`` alpha and ``Σ αₖ Tₖ cₖ + T·background`` colour, nearest first."""
    size, cam = 16, {"yaw": 0.0, "pitch": 0.0, "radius": 3.0, "fov": 30.0}
    k = len(opacities)
    ys = np.linspace(0.4, -0.4, k)  # camera on +y: the first is nearest
    blobs = [{"xyz": np.array([[0.0, y, 0.0]]), "scaling": np.full((1, 3), 0.2), "rotation": np.array([[1.0, 0, 0, 0]]),
              "opacity": np.array([o]), "rgb": np.array([[r, 0, 0]])} for y, o, r in zip(ys, opacities, reds[:k], strict=True)]
    alone = [render(b, cam, size=size, background=0.0) for b in blobs]
    both = render({key: np.concatenate([b[key] for b in reversed(blobs)]) for key in blobs[0]}, cam, size=size, background=1.0)
    a = np.array([o["alpha"][8, 8] for o in alone], float)
    transmittance = np.concatenate([[1.0], np.cumprod(1 - a)[:-1]])
    assert both["alpha"][8, 8] == pytest.approx(1 - np.prod(1 - a), abs=1e-6)
    expected_red = float((a * transmittance * np.array(reds[:k])).sum() + np.prod(1 - a) * 1.0)
    assert both["image"][8, 8][0] == pytest.approx(round(expected_red * 255), abs=1)


# ------------------------------------------------------------------------------------------------- Metrics.lean


def test_sdk_save_ply_axes(tmp_path):
    """lean: plyAxes_rotation — the vendored SDK's own ``Gaussian.save_ply``: a proper rotation sending +z to -y, and
    ``ply_roundtrip`` accepts the file it writes. Skips without torch."""
    torch = pytest.importorskip("torch")
    gaussian = load_sdk_module("gaussian.py")
    g = object.__new__(gaussian.Gaussian)  # set the activated values directly; the constructor only activates raw ones
    g.xyz = torch.eye(3)
    g.features_dc = torch.zeros(3, 1, 3)
    g.scaling = torch.full((3, 3), 0.01)
    g.rotation = torch.tensor([[1.0, 0, 0, 0]] * 3)
    g.opacity = torch.full((3, 1), 0.5)
    path = tmp_path / "sdk.ply"
    g.save_ply(str(path))
    m = read_ply(path)["values"][:, :3].T
    assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(m @ [0, 0, 1], [0, -1, 0], atol=1e-6)
    arrays = {"xyz": np.eye(3), "scaling": np.full((3, 3), 0.01), "rotation": np.tile([1.0, 0, 0, 0], (3, 1)),
              "opacity": np.full(3, 0.5), "rgb": np.full((3, 3), 0.5)}
    assert ply_roundtrip(path, arrays)["verified"]


def test_ply_axes_are_a_rotation_that_puts_up_along_minus_y(tmp_path):
    """lean: plyAxes_rotation — the ``.ply`` axis change is a proper rotation, and the object's up (+z) lands on -y,
    which is up for a y-down viewer, down for a y-up one."""
    points = np.eye(3)
    arrays = {"xyz": points, "scaling": np.full((3, 3), 0.01), "rotation": np.tile([1.0, 0, 0, 0], (3, 1)),
              "opacity": np.full(3, 0.5), "rgb": np.full((3, 3), 0.5)}
    path = tmp_path / "axes.ply"
    write_3dgs_ply(path, arrays)
    m = read_ply(path)["values"][:, :3].T  # columns: images of e_x, e_y, e_z
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(m @ [0, 0, 1], [0, -1, 0], atol=1e-6)
    assert ply_roundtrip(path, arrays)["verified"]


@given(st.integers(1, 6).flatmap(lambda h: st.tuples(st.lists(st.booleans(), min_size=h * 4, max_size=h * 4),
                                                       st.lists(st.booleans(), min_size=h * 4, max_size=h * 4))))
def test_silhouette_iou(ab):
    """lean: silhouetteIoU_mem, silhouetteIoU_comm, silhouetteIoU_empty — in [0, 1], symmetric, NaN when both are empty."""
    a, b = (np.array(x, float).reshape(-1, 4) for x in ab)
    iou = silhouette_iou(a, b)
    if not (a.astype(bool) | b.astype(bool)).any():
        assert math.isnan(iou)
    else:
        assert 0 <= iou <= 1 and iou == silhouette_iou(b, a)
        inter, union = (a.astype(bool) & b.astype(bool)).sum(), (a.astype(bool) | b.astype(bool)).sum()
        assert iou == pytest.approx(inter / union)
