"""The stage runner's NumPy-only stages on synthetic state, and its refusal paths (no torch, no network)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT, load_tool, synthetic_splat, write_3dgs_ply

stages = load_tool("tutorial_stages")


def _run(tmp_path: Path, stage: str, *extra: str) -> int:
    return stages.main(["--root", str(tmp_path / "run"), "--weights", str(tmp_path / "weights"), "--stage", stage, *extra])


def _fake_generation(tmp_path: Path, name: str = "ball") -> dict:
    run = tmp_path / "run"
    (run / "state").mkdir(parents=True, exist_ok=True)
    out = run / "outputs" / name
    out.mkdir(parents=True)
    arrays = synthetic_splat(3000)
    camera = {"yaw": 0.3, "pitch": 0.2, "radius": 2.0, "fov": 40.0}
    mask = np.zeros((128, 128), np.float32)
    np.savez_compressed(run / "state" / f"{name}.npz", mask=mask, **arrays)
    write_3dgs_ply(out / f"{name}.ply", arrays)
    (out / f"{name}_camera.json").write_text(json.dumps(camera))
    Image.new("RGB", (128, 128)).save(out / f"{name}_input.png")
    (run / "state" / "generated.json").write_text(json.dumps([name]))
    return {"arrays": arrays, "camera": camera}


def test_render_stage_checks_structure_file_and_silhouette(tmp_path):
    _fake_generation(tmp_path)
    # the mask is the render itself from the default reading, so that reading must win with IoU 1
    from idol_3d_pipeline import render

    stored = np.load(tmp_path / "run" / "state" / "ball.npz")
    arrays = {k: stored[k] for k in ("xyz", "scaling", "rotation", "opacity", "rgb")}
    camera = json.loads((tmp_path / "run" / "outputs" / "ball" / "ball_camera.json").read_text())
    alpha = render(arrays, camera, size=stages.CONVENTION_SIZE, convention="toward_camera")["alpha"]
    np.savez_compressed(tmp_path / "run" / "state" / "ball.npz", mask=alpha, **arrays)
    assert _run(tmp_path, "render") == 0
    record = json.loads((tmp_path / "run" / "outputs" / "evaluation.json").read_text())
    sample = record["samples"][0]
    assert sample["structure"]["verdict"] == "well-formed"
    assert sample["ply_roundtrip"]["verified"]
    assert sample["convention_used"] == "toward_camera" and sample["silhouette_iou"] == 1.0
    assert (tmp_path / "run" / "outputs" / "previews.png").is_file()


def test_a_stage_run_out_of_order_fails_with_an_instruction(tmp_path):
    assert _run(tmp_path, "render") == 2
    error = json.loads((tmp_path / "run" / "state" / "render.error.json").read_text())
    assert "generated.json is missing" in error["message"] and "from the top" in error["message"]


def test_byod_refuses_bad_paths_before_loading_any_model(tmp_path):
    assert _run(tmp_path, "byod") == 2
    assert "BYOD_PATH is empty" in json.loads((tmp_path / "run" / "state" / "byod.error.json").read_text())["message"]
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (16, 16)).save(tiny)
    assert _run(tmp_path, "byod", "--byod-path", str(tiny)) == 2
    error = json.loads((tmp_path / "run" / "state" / "byod.error.json").read_text())
    assert error["type"] == "ValueError" and "tiny.png" in error["message"] and "shorter side" in error["message"]


def test_byod_zip_extraction_ignores_entries_that_escape(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        buffer = tmp_path / "a.png"
        Image.new("RGB", (128, 128)).save(buffer)
        zf.write(buffer, "inside/a.png")
        zf.writestr("../escape.png", b"x")
    found = stages._byod_images(str(archive))
    assert [name for name, _ in found] == ["a"]
    assert not (tmp_path / "escape.png").exists()


def test_vae_check_needs_a_token(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert _run(tmp_path, "vae_check") == 2


def test_stage_list_matches_the_notebook():
    template = load_tool("notebook_template").TEMPLATE
    called = set()
    for cell in template["cells"]:
        for stage in stages.STAGES:
            if f"run_stage('{stage}'" in cell.get("code", ""):
                called.add(stage)
    assert called == set(stages.STAGES) - {"weights"}  # 'weights' is the generator's infrastructure cell
    assert (ROOT / "tools" / "tutorial_stages.py").is_file()
