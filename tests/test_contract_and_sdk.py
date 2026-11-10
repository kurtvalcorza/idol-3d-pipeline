"""The input contract, the mask replica against the vendored SDK's preprocessing, the samples, and the vendored SDK pins."""

from __future__ import annotations

# ruff: noqa: E501  -- messages and records are kept on one line each
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import load_sdk_module
from idol_3d_pipeline import SAMPLES, SDK_COMMIT, fetch_samples, prepare_with_mask, validate_request
from idol_3d_pipeline.samples import sample_url, verify_bytes

ROOT = Path(__file__).resolve().parents[1]
SDK_HASHES = json.loads((ROOT / "docs" / "upstream-sdk.json").read_text(encoding="utf-8"))


def test_vendored_sdk_is_byte_identical_to_the_pinned_upstream_commit():
    assert SDK_HASHES["commit"] == SDK_COMMIT
    for relative, digest in SDK_HASHES["files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, f"{relative} drifted from idol-sdk@{SDK_COMMIT[:7]}"
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "src" / "idol").rglob("*.py")}
    assert on_disk == set(SDK_HASHES["files"]), "every vendored file is recorded, and nothing else sits in src/idol"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image": Image.new("RGB", (32, 32))}, "shorter side"),
        ({"image": Image.new("RGB", (5000, 100))}, "longer side"),
        ({"image": Image.new("RGB", (256, 256)), "mode": "sketch"}, "not one of"),
        ({"image": Image.new("RGB", (256, 256)), "steps": 0}, "steps"),
        ({"image": Image.new("RGB", (256, 256)), "num_gaussians": 100_001}, "multiple of 32"),
        ({"image": "photo.jpg"}, "PIL.Image.Image"),
    ],
)
def test_invalid_requests_are_refused_with_the_rule(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        validate_request(**kwargs)


def test_alpha_decides_whether_birefnet_runs():
    rgba = Image.new("RGBA", (128, 128), (255, 0, 0, 0))
    assert validate_request(rgba)["has_alpha"] is True
    assert validate_request(Image.new("RGB", (128, 128)))["has_alpha"] is False


def _fake_remove_background(image: Image.Image) -> Image.Image:
    """A deterministic stand-in for BiRefNet: a centred disc."""
    width, height = image.size
    yy, xx = np.mgrid[:height, :width]
    disc = ((xx - width * 0.55) ** 2 + (yy - height * 0.45) ** 2 < (min(width, height) * 0.3) ** 2).astype(np.uint8) * 255
    out = image.copy()
    out.putalpha(Image.fromarray(disc, mode="L"))
    return out


class _Remover:
    def remove_background(self, image):
        return _fake_remove_background(image)


@pytest.mark.parametrize("margin", [1.2, 1.7])
def test_mask_replica_reproduces_the_sdk_composite_exactly(margin):
    preprocess = load_sdk_module("preprocess.py")
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 255, (300, 420, 3), dtype=np.uint8))
    sdk = preprocess.preprocess_image(image, background_remover=_Remover(), image_size=256, crop_margin=margin)
    composite, mask = prepare_with_mask(image, _fake_remove_background, crop_margin=margin, image_size=256)
    assert np.array_equal(np.asarray(composite), np.asarray(sdk))
    assert mask.shape == (256, 256) and mask.min() >= 0.0 and mask.max() <= 1.0 and mask.mean() > 0.1


def test_samples_are_pinned_to_the_sdk_commit_and_verified(tmp_path):
    assert {s["id"]: s["preset"]["mode"] for s in SAMPLES} == {"backpack": "synthetic", "armchair": "synthetic", "porcini": "real"}
    for sample in SAMPLES:
        assert sample_url(sample).startswith((f"https://raw.githubusercontent.com/speridlabs/idol-sdk/{SDK_COMMIT}/demo/examples/", "https://inaturalist-open-data.s3.amazonaws.com/photos/"))
        assert sample["license"]
        assert len(sample["sha256"]) == 64 and sample["preset"]["num_gaussians"] % 32 == 0
        with pytest.raises(ValueError, match="refusing to decode"):
            verify_bytes(sample, b"not the image")
    with pytest.raises(ValueError, match="refusing to decode"):
        fetch_samples(tmp_path, downloader=lambda url, limit: b"tampered")
