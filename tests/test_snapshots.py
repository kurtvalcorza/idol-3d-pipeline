"""Manifests, pins and the staging/verification contract (no network: a fake downloader writes the bytes)."""

from __future__ import annotations

# ruff: noqa: E501  -- messages and records are kept on one line each
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from idol_3d_pipeline import (
    SNAPSHOTS,
    UnpinnedSnapshotError,
    encoder_overrides,
    read_manifest,
    require_pinned,
    stage_snapshot,
    verify_snapshot,
)
from idol_3d_pipeline.pipeline import NON_COMMERCIAL_ORIGIN, UPSTREAM_ENCODERS, check_upstream_config, git_blob_sha1

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"


def test_every_snapshot_has_a_manifest_matching_the_package_pins():
    for key, spec in SNAPSHOTS.items():
        manifest = read_manifest(key, WEIGHTS)
        assert manifest["format"] == "dimer_hf_snapshot"
        assert manifest["modelKey"] == key
        assert (manifest["modelId"], manifest["revision"]) == (spec["model_id"], spec["revision"])
        assert len(spec["revision"]) == 40
        assert manifest["license"] == spec["license"]


def test_idol_and_klein_are_fully_pinned_and_only_safetensors_weights_are_listed():
    for key in ("idol", "flux2-vae-klein"):
        manifest = read_manifest(key, WEIGHTS)
        require_pinned(key, manifest)
        for entry in manifest["files"]:
            assert len(entry["sha256"]) == 64 and entry["bytes"] > 0
            assert not entry["path"].endswith((".bin", ".pt", ".pth", ".ckpt", ".pkl"))
        assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
    idol = {e["path"]: e for e in read_manifest("idol", WEIGHTS)["files"]}
    assert sum(e["bytes"] for p, e in idol.items() if p.endswith(".safetensors")) == 3_078_979_896
    klein = {e["path"]: e for e in read_manifest("flux2-vae-klein", WEIGHTS)["files"]}
    assert klein["vae/diffusion_pytorch_model.safetensors"]["sha256"] == "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"


def test_all_three_snapshots_are_pinned_and_dinov3_is_the_recorded_file():
    for key in SNAPSHOTS:
        require_pinned(key, read_manifest(key, WEIGHTS))
    dino = {e["path"]: e for e in read_manifest("dinov3-vith16plus", WEIGHTS)["files"]}
    assert dino["model.safetensors"]["bytes"] == 3_362_432_800
    assert dino["model.safetensors"]["sha256"] == "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"
    assert "pinStatus" not in read_manifest("dinov3-vith16plus", WEIGHTS)


def test_an_unpinned_manifest_is_refused_with_the_pin_command():
    manifest = dict(read_manifest("dinov3-vith16plus", WEIGHTS))
    manifest["files"] = [dict(e, sha256=None) for e in manifest["files"]]
    with pytest.raises(UnpinnedSnapshotError, match="pin_snapshot.py dinov3-vith16plus"):
        require_pinned("dinov3-vith16plus", manifest)

def test_no_snapshot_is_the_non_commercial_flux2_dev_origin():
    assert NON_COMMERCIAL_ORIGIN not in {spec["model_id"] for spec in SNAPSHOTS.values()}
    overrides = encoder_overrides(WEIGHTS)
    assert set(overrides) == set(UPSTREAM_ENCODERS)
    assert all(Path(path).parent == WEIGHTS for path in overrides.values())


def _fake_snapshot(tmp_path: Path, key: str, files: dict[str, bytes], *, pinned: bool = True) -> Path:
    spec = SNAPSHOTS[key]
    entries = [{"path": p, "bytes": len(b) if pinned else None, "sha256": hashlib.sha256(b).hexdigest() if pinned else None} for p, b in files.items()]
    (tmp_path / key).mkdir(parents=True)
    manifest = {"format": "dimer_hf_snapshot", "formatVersion": 1, "modelKey": key, "modelId": spec["model_id"], "revision": spec["revision"], "license": spec["license"], "files": entries, "totalBytes": sum(len(b) for b in files.values())}
    (tmp_path / key / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_stage_then_verify_fetches_only_missing_files_at_the_pinned_revision(tmp_path):
    files = {"vae/config.json": b"{}", "vae/diffusion_pytorch_model.safetensors": b"weights"}
    root = _fake_snapshot(tmp_path, "flux2-vae-klein", files)
    calls = []

    def downloader(model_id, relative, revision, directory, token):
        calls.append((model_id, relative, revision, token))
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_bytes(files[relative])

    assert stage_snapshot("flux2-vae-klein", root, token="secret", downloader=downloader) == list(files)
    assert all(c[0] == "black-forest-labs/FLUX.2-klein-4B" and c[2] == SNAPSHOTS["flux2-vae-klein"]["revision"] for c in calls)
    assert all(c[3] is None for c in calls), "an ungated snapshot must not receive the token"
    assert verify_snapshot("flux2-vae-klein", root)["files"] == list(files)
    assert stage_snapshot("flux2-vae-klein", root, downloader=downloader) == []


def test_the_real_downloader_never_falls_back_to_an_implicit_token(monkeypatch, tmp_path):
    import huggingface_hub

    from idol_3d_pipeline import pipeline

    seen = {}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: seen.update(k))
    pipeline._hub_download("speridlabs/idol", "config.json", "0" * 40, tmp_path, None)
    assert seen["token"] is False
    pipeline._hub_download("x/y", "f", "0" * 40, tmp_path, "secret")
    assert seen["token"] == "secret"


def test_tampered_or_missing_files_are_refused(tmp_path):
    files = {"vae/config.json": b"{}", "vae/diffusion_pytorch_model.safetensors": b"weights"}
    root = _fake_snapshot(tmp_path, "flux2-vae-klein", files)
    with pytest.raises(FileNotFoundError):
        verify_snapshot("flux2-vae-klein", root)
    for relative, data in files.items():
        (root / "flux2-vae-klein" / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / "flux2-vae-klein" / relative).write_bytes(data)
    (root / "flux2-vae-klein" / "vae/diffusion_pytorch_model.safetensors").write_bytes(b"weighTs")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot("flux2-vae-klein", root)


def test_a_gated_snapshot_needs_a_token_and_receives_it(tmp_path):
    files = {"config.json": b"{}", "model.safetensors": b"w", "LICENSE.md": b"l"}
    root = _fake_snapshot(tmp_path, "dinov3-vith16plus", files)
    with pytest.raises(PermissionError, match="HF_TOKEN"):
        stage_snapshot("dinov3-vith16plus", root, token=None, downloader=lambda *a: None)
    seen = []

    def downloader(model_id, relative, revision, directory, token):
        seen.append(token)
        (directory / relative).write_bytes(files[relative])

    stage_snapshot("dinov3-vith16plus", root, token="secret", downloader=downloader)
    assert seen == ["secret"] * 3


def test_a_manifest_for_another_revision_is_refused(tmp_path):
    root = _fake_snapshot(tmp_path, "idol", {"config.json": b"{}"})
    path = root / "idol" / "dimer-base-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["revision"] = "0" * 40
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="package pins"):
        verify_snapshot("idol", root)


def test_upstream_config_check_reads_the_pinned_encoders(tmp_path):
    config = {"image_encoders": dict(UPSTREAM_ENCODERS), "image_size": 1024}
    (tmp_path / "idol").mkdir()
    (tmp_path / "idol" / "config.json").write_text(json.dumps(config))
    assert check_upstream_config(tmp_path)["image_size"] == 1024
    config["image_encoders"]["flux2_vae"] = "someone/else"
    (tmp_path / "idol" / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="image_encoders"):
        check_upstream_config(tmp_path)


def test_git_blob_sha1_matches_the_hub_oid_recorded_for_idol_config():
    entry = next(e for e in read_manifest("idol", WEIGHTS)["files"] if e["path"] == "config.json")
    # git's own definition: the empty blob is e69de29...
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert len(entry["gitBlobSha1"]) == 40
    assert shutil.which  # keeps the import used on minimal runners
