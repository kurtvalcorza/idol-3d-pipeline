#!/usr/bin/env python3
"""Record per-file sizes and SHA-256 digests in a snapshot manifest, at the revision the manifest already pins.

The revision itself is chosen by a reviewed commit, not by this tool: it only fills digests for the files the
manifest lists, so the pin cannot drift to whatever ``main`` is today. For each listed file it asks the Hub for the
file's metadata at the pinned revision:

* an LFS file (the weights): size and SHA-256 come from the Hub's LFS record -- nothing large is downloaded;
* a small file (configs, licence): it is downloaded at the revision, its git blob SHA-1 is checked against the Hub's
  ``oid``, and its SHA-256 is computed from those bytes.

A digest already recorded must match, or the tool stops without writing. Gated snapshots (DINOv3) need ``HF_TOKEN``
from an account that has accepted the model's terms.

Usage (from the repository root):
    HF_TOKEN=... python tools/pin_snapshot.py dinov3-vith16plus
    python tools/pin_snapshot.py dinov3-vith16plus --dry-run
Then commit the manifest, regenerate the notebook (python tools/build_notebook.py) and run pytest.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", help="snapshot key under weights/, e.g. dinov3-vith16plus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi, hf_hub_download

    path = ROOT / "weights" / args.key / "dimer-base-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    repo, revision = manifest["modelId"], manifest["revision"]
    token = os.environ.get("HF_TOKEN") or None
    infos = {info.path: info for info in HfApi(token=token).get_paths_info(repo, [f["path"] for f in manifest["files"]], revision=revision, expand=True)}
    for entry in manifest["files"]:
        info = infos.get(entry["path"])
        if info is None:
            raise SystemExit(f"{repo}@{revision} has no file {entry['path']}")
        if getattr(info, "lfs", None):
            size, sha256, blob = int(info.lfs.size), info.lfs.sha256, None
        else:
            with tempfile.TemporaryDirectory() as tmp:
                data = Path(hf_hub_download(repo, entry["path"], revision=revision, local_dir=tmp, token=token)).read_bytes()
            blob = git_blob_sha1(data)
            if blob != info.blob_id:
                raise SystemExit(f"{entry['path']}: git blob {blob} != Hub oid {info.blob_id}; refusing to record it")
            size, sha256 = len(data), hashlib.sha256(data).hexdigest()
        for field, value in (("bytes", size), ("sha256", sha256)):
            if entry.get(field) not in (None, value):
                raise SystemExit(f"{entry['path']}: recorded {field} {entry[field]} != Hub {value}; nothing written")
        entry.update({"bytes": size, "sha256": sha256})
        if blob:
            entry["gitBlobSha1"] = blob
        print({"file": entry["path"], "bytes": size, "sha256": sha256})
    manifest["totalBytes"] = sum(f["bytes"] for f in manifest["files"])
    manifest.pop("pinStatus", None)
    if args.dry_run:
        print("dry run: manifest not written")
        return 0
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}; now run: python tools/build_notebook.py && pytest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
