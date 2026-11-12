"""Single-image to 3D Gaussian splat with Idol (``speridlabs/idol``), on licence-clean encoders.

Idol's published ``config.json`` loads two gated image encoders: DINOv3 ViT-H+/16 and the VAE of
``black-forest-labs/FLUX.2-dev`` (FLUX Non-Commercial License). This package loads Idol through the upstream SDK
(carried verbatim under ``src/idol`` at a pinned commit) with both encoders replaced by local, digest-verified
snapshots:

* ``flux2-vae-klein`` -- the VAE of ``black-forest-labs/FLUX.2-klein-4B`` (Apache-2.0, ungated), presumed to be the
  FLUX.2 [dev] VAE stored in bf16 (the optional ``vae_check`` stage of the tutorial tests that presumption); and
* ``dinov3-vith16plus`` -- DINOv3 ViT-H+/16 from its gated origin, fetched with the runtime's ``HF_TOKEN``.

FLUX.2-dev is never contacted on the default path. Every snapshot is pinned by an immutable Hub commit and per-file
digests recorded in ``weights/<key>/dimer-base-manifest.json``; a snapshot whose manifest lacks digests (DINOv3 until
``tools/pin_snapshot.py`` has run with access) is refused, never trusted on first use.

This module imports only the standard library, NumPy and Pillow at import time; torch and the SDK are imported when a
model is loaded, so the contract and verification code is testable without them.
"""

from __future__ import annotations

# ruff: noqa: E501  -- refusal messages name the condition and the corrective action on one line
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

MODEL_ID = "speridlabs/idol"
MODEL_REVISION = "2731c2291fc65a797c5cb627a5e5526fc1490902"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "idol"
SDK_REPOSITORY = "speridlabs/idol-sdk"
SDK_COMMIT = "8db700349a5570aac390e4780fa32e3eed6b587a"
MANIFEST_NAME = "dimer-base-manifest.json"
PIN_COMMAND = "python tools/pin_snapshot.py"
DEFAULT_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"

# Snapshot key -> Hub id, revision, the encoder it replaces in Idol's config.json (None for Idol itself), gating.
SNAPSHOTS: dict[str, dict[str, Any]] = {
    MODEL_KEY: {"model_id": MODEL_ID, "revision": MODEL_REVISION, "encoder": None, "gated": False, "license": "apache-2.0"},
    "flux2-vae-klein": {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "encoder": "flux2_vae",
        "gated": False,
        "license": "apache-2.0",
    },
    "dinov3-vith16plus": {
        "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "revision": "c807c9eeea853df70aec4069e6f56b28ddc82acc",
        "encoder": "dinov3",
        "gated": True,
        "license": "dinov3-license",
    },
}
# The encoders Idol's own config.json names; the pipeline replaces both and checks that the replacement is complete.
UPSTREAM_ENCODERS = {"dinov3": "facebook/dinov3-vith16plus-pretrain-lvd1689m", "flux2_vae": "black-forest-labs/FLUX.2-dev"}
NON_COMMERCIAL_ORIGIN = "black-forest-labs/FLUX.2-dev"

# Input contract (Idol resizes the short side to 1024 px; the limits keep a hosted runtime's memory bounded).
MODES = ("synthetic", "real")  # crop margin 1.2 (renders, generated images) or 1.7 (photographs), from the SDK
MIN_INPUT_SIDE = 64
MAX_INPUT_SIDE = 4096
MIN_GAUSSIANS, MAX_GAUSSIANS, GAUSSIANS_PER_ANCHOR = 32_768, 1_048_576, 32
DEFAULT_GAUSSIANS = 262_144
MAX_STEPS = 100
IMAGE_SIZE = 1024  # Idol config.json "image_size"
SH_C0 = 0.28209479177387814  # degree-0 spherical-harmonic constant: rgb = 0.5 + SH_C0 * f_dc


class UnpinnedSnapshotError(RuntimeError):
    """A manifest carries no immutable revision, or a file without a recorded digest."""


class InputRefused(ValueError):
    """A user-supplied input breaks the data contract. The tutorial shows it as one message, without a traceback (DAT20)."""


ACCEPTED_PIL_MODES = ("RGB", "RGBA", "L", "LA", "P")  # anything else is refused; L/LA/P are converted to RGB(A), and the record says so


# ---------------------------------------------------------------------------------------------------------
# Snapshot manifests
# ---------------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """The Hub's ``oid`` for a non-LFS file: SHA-1 over ``b"blob <len>\\0" + data``."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read_manifest(key: str, weights_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(weights_root) if weights_root is not None else DEFAULT_WEIGHTS_ROOT
    path = root / key / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_pinned(key: str, manifest: dict[str, Any]) -> None:
    """Refuse a manifest that does not match the package pins or lists a file without size and SHA-256."""
    expected = SNAPSHOTS[key]
    if manifest.get("modelId") != expected["model_id"] or manifest.get("revision") != expected["revision"]:
        raise ValueError(
            f"{key}: manifest pins {manifest.get('modelId')}@{manifest.get('revision')}, package pins {expected['model_id']}@{expected['revision']}; regenerate the notebook from one repository revision"
        )
    unpinned = [entry["path"] for entry in manifest.get("files", []) if not entry.get("sha256") or not entry.get("bytes")]
    if unpinned:
        hint = " with an HF_TOKEN whose account has accepted the model's terms" if expected["gated"] else ""
        raise UnpinnedSnapshotError(
            f"{key} ({expected['model_id']}) has no recorded size/SHA-256 for {unpinned}: run `{PIN_COMMAND} {key}`{hint}, commit the manifest, then regenerate the notebook. Unverified weights are never loaded."
        )


def verify_snapshot(key: str, weights_root: str | Path | None = None) -> dict[str, Any]:
    """Check every manifest-listed file of one snapshot by size and SHA-256; raise naming the first mismatch."""
    root = Path(weights_root) if weights_root is not None else DEFAULT_WEIGHTS_ROOT
    manifest = read_manifest(key, root)
    require_pinned(key, manifest)
    for entry in manifest["files"]:
        path = root / key / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{key}: snapshot file missing: {path}")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{key}/{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"{key}/{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "key": key,
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "license": manifest.get("license"),
        "files": [entry["path"] for entry in manifest["files"]],
        "total_bytes": manifest["totalBytes"],
    }


def _hub_download(model_id: str, relative_path: str, revision: str, directory: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    # token=False sends no credential at all; None would let huggingface_hub fall back to HF_TOKEN or a cached login.
    hf_hub_download(model_id, relative_path, revision=revision, local_dir=str(directory), token=token if token else False)


def stage_snapshot(
    key: str,
    weights_root: str | Path | None = None,
    *,
    token: str | None = None,
    downloader: Callable[[str, str, str, Path, str | None], None] | None = None,
) -> list[str]:
    """Fetch the manifest-listed files of one snapshot that are absent locally, at the pinned revision only.

    A gated snapshot needs ``token``; an ungated one is fetched without it, so a token is never sent where it is not
    needed. Returns the relative paths fetched; ``verify_snapshot`` must still run afterwards."""
    root = Path(weights_root) if weights_root is not None else DEFAULT_WEIGHTS_ROOT
    manifest = read_manifest(key, root)
    require_pinned(key, manifest)
    spec = SNAPSHOTS[key]
    missing = [entry["path"] for entry in manifest["files"] if not (root / key / entry["path"]).is_file()]
    if not missing:
        return []
    if spec["gated"] and not token:
        raise PermissionError(
            f"{key} ({spec['model_id']}) is gated: set HF_TOKEN to a token of an account that has accepted its terms at https://huggingface.co/{spec['model_id']} (in Colab: the key icon, a secret named HF_TOKEN, notebook access on)"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(spec["model_id"], relative_path, spec["revision"], root / key, token if spec["gated"] else None)
    return missing


def check_gated_access(key: str, token: str | None, *, api: Any = None) -> dict[str, Any]:
    """Before any download, ask the Hub who the token belongs to and whether that account may read the gated snapshot.

    Turns the Hub's two refusals into one actionable message: an unknown or revoked token (the Hub answers as if no one
    were logged in, HTTP 401) versus a valid token whose account has no access yet (HTTP 403)."""
    spec = SNAPSHOTS[key]
    if not token:
        raise PermissionError(f"{key} ({spec['model_id']}) is gated: set HF_TOKEN to a token of an account that has accepted its terms at https://huggingface.co/{spec['model_id']}")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    try:
        account = api.whoami(token=token).get("name")
    except Exception as error:
        raise PermissionError(
            f"the Hugging Face Hub does not recognise this HF_TOKEN ({type(error).__name__}): it is mistyped, revoked or expired. Create a new token at https://huggingface.co/settings/tokens (Read, or fine-grained with 'Read access to contents of all public gated repos you can access'), put it in the HF_TOKEN secret, and run the Token cell again."
        ) from None
    try:
        api.auth_check(spec["model_id"], token=token)
    except Exception as error:
        raise PermissionError(
            f"HF_TOKEN belongs to account '{account}', which cannot read {spec['model_id']} ({type(error).__name__}): request access at https://huggingface.co/{spec['model_id']} with that account (approval is manual), or, for a fine-grained token, enable 'Read access to contents of all public gated repos you can access'."
        ) from None
    return {"account": account, "model_id": spec["model_id"], "access": True}


def encoder_overrides(weights_root: str | Path | None = None) -> dict[str, str]:
    """``image_encoders`` for ``idol.IdolPipeline``: every encoder Idol's config names, replaced by a local snapshot."""
    root = Path(weights_root) if weights_root is not None else DEFAULT_WEIGHTS_ROOT
    overrides = {spec["encoder"]: str(root / key) for key, spec in SNAPSHOTS.items() if spec["encoder"]}
    if set(overrides) != set(UPSTREAM_ENCODERS):
        raise RuntimeError(f"encoder overrides {sorted(overrides)} do not cover Idol's encoders {sorted(UPSTREAM_ENCODERS)}")
    return overrides


def check_upstream_config(weights_root: str | Path | None = None) -> dict[str, Any]:
    """Read Idol's pinned config.json and confirm it names exactly the encoders this package replaces."""
    root = Path(weights_root) if weights_root is not None else DEFAULT_WEIGHTS_ROOT
    config = json.loads((root / MODEL_KEY / "config.json").read_text(encoding="utf-8"))
    if config.get("image_encoders") != UPSTREAM_ENCODERS:
        raise ValueError(f"Idol config.json image_encoders {config.get('image_encoders')} != the pinned expectation {UPSTREAM_ENCODERS}")
    if config.get("image_size") != IMAGE_SIZE:
        raise ValueError(f"Idol config.json image_size {config.get('image_size')} != {IMAGE_SIZE}")
    return config


# ---------------------------------------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------------------------------------


def validate_request(image: Any, *, mode: str = "synthetic", steps: int = 50, num_gaussians: int = DEFAULT_GAUSSIANS, name: str = "image") -> dict[str, Any]:
    """Check one request before any model runs; return its record. Messages name the rule and the fix."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"{name}: expected a PIL.Image.Image, got {type(image).__name__}; open the file with PIL.Image.open first")
    width, height = image.size
    if image.mode not in ACCEPTED_PIL_MODES:
        raise InputRefused(f"{name}: image mode {image.mode} is not accepted ({', '.join(ACCEPTED_PIL_MODES)}); save it as an 8-bit RGB or RGBA PNG or JPEG")
    if min(width, height) < MIN_INPUT_SIDE:
        raise InputRefused(f"{name}: shorter side {min(width, height)} px < {MIN_INPUT_SIDE} px; supply a larger image of one object")
    if max(width, height) > MAX_INPUT_SIDE:
        raise InputRefused(f"{name}: longer side {max(width, height)} px > {MAX_INPUT_SIDE} px; downscale it first (the pipeline never resizes silently beyond Idol's own 1024 px step)")
    if mode not in MODES:
        raise InputRefused(f"{name}: mode {mode!r} is not one of {MODES}: 'synthetic' for renders and generated images (crop 1.2), 'real' for photographs (crop 1.7)")
    if not 1 <= int(steps) <= MAX_STEPS:
        raise InputRefused(f"{name}: steps {steps} outside 1..{MAX_STEPS}")
    if not MIN_GAUSSIANS <= int(num_gaussians) <= MAX_GAUSSIANS or int(num_gaussians) % GAUSSIANS_PER_ANCHOR:
        raise InputRefused(f"{name}: num_gaussians {num_gaussians} must be a multiple of {GAUSSIANS_PER_ANCHOR} within {MIN_GAUSSIANS}..{MAX_GAUSSIANS}")
    alpha = image.mode in ("RGBA", "LA") and np.asarray(image.getchannel(image.getbands()[-1])).min() < 255
    return {
        "id": name,
        "size": [width, height],
        "mode_pil": image.mode,
        "converted": None if image.mode in ("RGB", "RGBA") else f"{image.mode} -> {'RGBA' if 'A' in image.getbands() or image.mode == 'P' else 'RGB'} by the SDK's own conversion",
        "has_alpha": bool(alpha),
        "background_removal": "skipped: the image's own alpha is used" if alpha else "BiRefNet predicts the object mask",
        "request": {"mode": mode, "steps": int(steps), "num_gaussians": int(num_gaussians)},
        "verdict": "accepted",
    }


def fit_within_limit(image: Image.Image, *, limit: int = MAX_INPUT_SIDE) -> tuple[Image.Image, dict[str, Any] | None]:
    """Shrink an image whose longer side exceeds ``limit`` to fit it, keeping the aspect ratio; return it and a record.

    For Bring Your Own Data: camera photographs are often larger than the 4096 px the request contract accepts. Idol's
    first step resizes the short side to 1024 px (Lanczos), so for any image no more elongated than 4:1 this shrink
    changes nothing Idol uses. The record names the original and new size, so the change is never silent. Images
    within the limit, and modes the contract refuses, are returned unchanged with ``None``; ``validate_request``
    still judges the result (a very elongated image can still be refused for its shorter side)."""
    width, height = image.size
    if max(width, height) <= limit or image.mode not in ACCEPTED_PIL_MODES:
        return image, None
    scale = limit / max(width, height)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    source = image.convert("RGBA") if image.mode == "P" else image  # a palette image cannot be filtered
    record = {"from": [width, height], "to": list(size), "filter": "lanczos", "converted": "P -> RGBA" if image.mode == "P" else None}
    return source.resize(size, Image.LANCZOS), record


def prepare_with_mask(image: Image.Image, remove_background: Callable[[Image.Image], Image.Image], *, crop_margin: float, image_size: int = IMAGE_SIZE) -> tuple[Image.Image, np.ndarray]:
    """``idol.preprocess.preprocess_image`` step for step, also returning the cropped object mask (float in [0, 1]).

    Idol's own function returns only the RGB composite; the mask is needed for the input-view silhouette check. The
    tutorial compares this composite with the SDK's byte for byte, so a drift in either is reported, not hidden."""
    width, height = image.size
    scale = image_size / min(width, height)
    image = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.LANCZOS)
    if not (image.mode == "RGBA" and np.array(image.getchannel(3), dtype=np.int32).min() < 255):
        image = remove_background(image.convert("RGB"))
    image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(3)))
    alpha = np.array(image.getchannel(3))
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        raise ValueError("no object found in the image (the alpha channel is empty)")
    left, top, right, bottom = xs.min(), ys.min(), xs.max(), ys.max()
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half = max(right - left, bottom - top) / 2 * crop_margin
    image = image.crop([int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)])
    image = image.resize((image_size, image_size), Image.LANCZOS)
    composite = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    composite.paste(image, mask=image.split()[3])
    return composite, np.asarray(image.getchannel(3), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------


def gaussian_arrays(gaussian: Any) -> dict[str, np.ndarray]:
    """Activated Gaussian parameters as float32 NumPy arrays (z-up object frame inside [-0.5, 0.5]^3)."""

    def arr(t: Any) -> np.ndarray:
        return t.detach().float().cpu().numpy().astype(np.float32)

    features_dc = arr(gaussian.features_dc).reshape(-1, 3)
    return {
        "xyz": arr(gaussian.xyz),
        "scaling": arr(gaussian.scaling),
        "rotation": arr(gaussian.rotation),  # unit quaternions (w, x, y, z)
        "opacity": arr(gaussian.opacity).reshape(-1),
        "rgb": np.clip(0.5 + SH_C0 * features_dc, 0.0, 1.0).astype(np.float32),
    }


@dataclass
class IdolSplatPipeline:
    """The upstream ``idol.IdolPipeline`` built from verified local snapshots, with both encoders replaced."""

    idol: Any
    device: str
    snapshots: dict[str, dict[str, Any]]

    @classmethod
    def from_snapshots(cls, weights_root: str | Path | None = None, *, device: str | None = None, optimize: bool = False, dinov3_dtype: str = "bf16") -> IdolSplatPipeline:
        root = Path(weights_root) if weights_root is not None else DEFAULT_WEIGHTS_ROOT
        snapshots = {key: verify_snapshot(key, root) for key in SNAPSHOTS}  # refuse before importing model code
        check_upstream_config(root)
        os.environ.setdefault("ATTN_BACKEND", "sdpa")  # flash-attn is not in the lock; make the SDK's choice explicit
        from idol import IdolPipeline

        idol = IdolPipeline(str(root / MODEL_KEY), device=device, dinov3_dtype=dinov3_dtype, optimize=optimize, image_encoders=encoder_overrides(root))
        return cls(idol, str(idol.device), snapshots)

    def remove_background(self, image: Image.Image) -> Image.Image:
        return self.idol.background_remover.remove_background(image)

    def generate(self, image: Image.Image, *, name: str = "image", mode: str = "synthetic", seed: int = 42, steps: int = 50, cfg_strength: float = 3.0, rescale_t: float = 3.0, num_gaussians: int = DEFAULT_GAUSSIANS) -> dict[str, Any]:
        """Run Idol on one image; return the Gaussians, the predicted camera, the model input and its mask."""
        from idol.preprocess import CROP_MARGIN

        record = validate_request(image, mode=mode, steps=steps, num_gaussians=num_gaussians, name=name)
        started = time.perf_counter()
        output = self.idol(image, mode=mode, seed=seed, steps=steps, cfg_strength=cfg_strength, rescale_t=rescale_t, num_gaussians=num_gaussians)
        seconds = time.perf_counter() - started
        composite, mask = prepare_with_mask(image, self.remove_background, crop_margin=CROP_MARGIN[mode])
        replica_matches = bool(np.array_equal(np.asarray(composite), np.asarray(output.preprocessed_image)))
        return {
            "record": record,
            "gaussian": output.gaussian,
            "arrays": gaussian_arrays(output.gaussian),
            "camera": dict(output.camera),
            "model_input": output.preprocessed_image,
            "mask": mask,
            "mask_replica_matches_sdk": replica_matches,
            "settings": {"mode": mode, "seed": seed, "steps": steps, "cfg_strength": cfg_strength, "rescale_t": rescale_t, "num_gaussians": num_gaussians},
            "seconds": round(seconds, 2),
            "device": self.device,
        }
