"""Pinned sample inputs: example images that ship with the Idol SDK, fetched by commit and digest-verified.

The images are ``demo/examples/*.webp`` of ``speridlabs/idol-sdk`` at the pinned commit, a repository released under
Apache-2.0; each comes with the settings the upstream demo runs it with (``demo/examples/presets.json``). The SDK does
not say how the images were made (their names suggest renders and generated images, which is the ``synthetic`` input
kind Idol's default mode is for); the tutorial states that, and a user's own photographs go through BYOD in ``real``
mode.
"""

from __future__ import annotations

# ruff: noqa: E501
import hashlib
import io
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .pipeline import SDK_COMMIT, SDK_REPOSITORY

RAW_BASE = f"https://raw.githubusercontent.com/{SDK_REPOSITORY}/{SDK_COMMIT}/demo/"
SAMPLE_LICENSE = "apache-2.0 (the idol-sdk repository's LICENSE; the images' generator is not stated upstream)"

# id, upstream path, bytes, SHA-256 (of the file at SDK_COMMIT), and the upstream demo preset
SAMPLES: list[dict[str, Any]] = [
    {
        "id": "backpack",
        "path": "examples/obj-backpack.webp",
        "bytes": 40590,
        "sha256": "a059c2d72c6420be3471edfc1588f211b1789477c0d5a8776a1f0698c76fdaf9",
        "preset": {"mode": "synthetic", "seed": 42, "steps": 50, "cfg_strength": 3.0, "rescale_t": 3.0, "num_gaussians": 262144},
        "note": "the upstream demo's default example",
    },
    {
        "id": "armchair",
        "path": "examples/nb-armchair.webp",
        "bytes": 29064,
        "sha256": "78e92542fe244d0e174802102446addfab7eef9b14e652d97ec1151deb24ff09",
        "preset": {"mode": "synthetic", "seed": 7, "steps": 50, "cfg_strength": 3.0, "rescale_t": 3.0, "num_gaussians": 262144},
        "note": "first entry of the upstream demo's presets.json",
    },
]


def sample_url(sample: dict[str, Any]) -> str:
    return RAW_BASE + sample["path"]


def verify_bytes(sample: dict[str, Any], data: bytes) -> None:
    if len(data) != sample["bytes"] or hashlib.sha256(data).hexdigest() != sample["sha256"]:
        raise ValueError(f"sample {sample['id']}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()} != pinned {sample['bytes']} bytes, {sample['sha256']}; refusing to decode it")


def _download(url: str, limit: int) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read(limit + 1)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_samples(cache: str | Path, *, downloader: Any = None) -> dict[str, bytes]:
    """``{id: bytes}`` of every pinned sample, from ``cache`` when present and verified, else downloaded and cached."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    fetch = downloader or _download
    out: dict[str, bytes] = {}
    for sample in SAMPLES:
        path = cache / Path(sample["path"]).name
        data = path.read_bytes() if path.is_file() else b""
        try:
            verify_bytes(sample, data)
        except ValueError:
            data = fetch(sample_url(sample), sample["bytes"])
            verify_bytes(sample, data)
            path.write_bytes(data)
        out[sample["id"]] = data
    return out


def open_sample(data: bytes) -> Any:
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    return image
