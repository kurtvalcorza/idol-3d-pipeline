"""Stage runner for the standalone Idol tutorial (NOTEBOOK_SPEC §25.14 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage generate

Nothing is installed into the notebook kernel. Each stage is a separate process, so a stage starts from files only:
the verified snapshots under ``--weights``, the digest-pinned sample images cached beside them, and the records of
earlier stages. Learner-facing exports go to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On
failure a stage writes ``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook
re-raises in the kernel.

Stages: weights -> prepare -> generate -> render, plus the optional ``byod`` and ``vae_check``.
Only ``weights``, ``generate``, ``byod`` and ``vae_check`` need the network or a GPU; ``render`` is NumPy only.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

STEM = "idol_3d"
SAMPLE_CACHE = "samples"
BYOD_MAX_IMAGES = 4
BYOD_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CONVENTION_SIZE = 256
VIEW_SIZE = 320
TURNTABLE = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, the snapshots and sample cache under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def runtime_versions() -> dict[str, Any]:
    import diffusers
    import numpy
    import PIL
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or platform.machine(),
    }


def panel(rows: list[list[tuple[str, Any]]], path: Path, tile: int = 256) -> Path:
    """A labelled grid: one row per item, one column per (label, uint8 array or PIL image)."""
    import numpy as np
    from PIL import Image, ImageDraw

    label_h = 18
    columns = max(len(r) for r in rows)
    sheet = Image.new("RGB", (tile * columns, (tile + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, (label, image) in enumerate(row):
            im = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
            scale = tile / max(im.size)
            im = im.convert("RGB").resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
            x, y = c * tile, r * (tile + label_h)
            sheet.paste(im, (x + (tile - im.width) // 2, y + label_h + (tile - im.height) // 2))
            draw.text((x + 4, y + 3), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def require_cuda(stage: str) -> None:
    import torch

    if not torch.cuda.is_available() and not os.environ.get("IDOL_ALLOW_CPU"):
        raise RuntimeError(
            f"stage '{stage}' needs a CUDA GPU that this environment's torch can use: Idol's 0.37 B-parameter denoiser runs 50 guided steps over 8,192 latent tokens (on a CPU, an estimated tens of minutes per image; not measured). If a GPU is attached, its driver may be too old for the CUDA 13 build in the lock (see Troubleshooting). In Colab choose Runtime -> Change runtime type -> T4 GPU, then Run all (set IDOL_ALLOW_CPU=1 to force the CPU anyway)."
        )


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def install_manifests(run: Run) -> list[str]:
    """Copy the carried manifests (ROOT/weights/<key>/) to the weights directory the stages read (WEIGHTS/<key>/).

    The weights directory outlives a run directory, so a manifest already there must equal the carried one: a
    different one means weights staged for another notebook revision, which is refused rather than mixed."""
    from idol_3d_pipeline import SNAPSHOTS
    from idol_3d_pipeline.pipeline import MANIFEST_NAME

    installed = []
    for key in SNAPSHOTS:
        carried = run.root / "weights" / key / MANIFEST_NAME
        if not carried.is_file():
            if (run.weights / key / MANIFEST_NAME).is_file() and run.root.resolve() == run.weights.resolve().parent:
                continue  # running from a repository checkout: the manifests already sit in weights/
            raise FileNotFoundError(f"carried manifest missing: {carried}; run the Section 2 carrier cell first")
        target = run.weights / key / MANIFEST_NAME
        if target.is_file() and target.read_bytes() != carried.read_bytes():
            raise RuntimeError(f"{target} differs from the manifest this notebook carries: the weights directory was staged by another revision. Delete {run.weights / key} (or start a fresh runtime) and run Section 3 again.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            target.write_bytes(carried.read_bytes())
        installed.append(key)
    return installed


def stage_weights(run: Run) -> None:
    """Stage and digest-verify Idol, the klein VAE and DINOv3 at their pinned revisions."""
    from idol_3d_pipeline import SNAPSHOTS, read_manifest, require_pinned, stage_snapshot, verify_snapshot
    from idol_3d_pipeline.pipeline import NON_COMMERCIAL_ORIGIN, check_upstream_config

    install_manifests(run)
    for key in SNAPSHOTS:  # every manifest must be pinned before a single byte is downloaded
        require_pinned(key, read_manifest(key, run.weights))
    in_manifests = {read_manifest(key, run.weights)["modelId"] for key in SNAPSHOTS}
    records = []
    for key, spec in SNAPSHOTS.items():
        started = time.perf_counter()
        fetched = stage_snapshot(key, run.weights, token=token())
        record = verify_snapshot(key, run.weights)
        record.update({"fetched": fetched, "gated": spec["gated"], "seconds": round(time.perf_counter() - started, 1)})
        records.append(record)
        print({"snapshot": key, "model_id": record["model_id"], "revision": record["revision"][:12], "licence": record["license"], "files": len(record["files"]), "gib": round(record["total_bytes"] / 1024**3, 2), "verified": True, "fetched_now": len(fetched)})
    config = check_upstream_config(run.weights)
    print({"idol_config_encoders": config["image_encoders"], "loaded_from": {"dinov3": "dinov3-vith16plus (same gated model, local verified copy)", "flux2_vae": "flux2-vae-klein (Apache-2.0 VAE, local verified copy)"}, "flux2_dev_in_any_staged_manifest": NON_COMMERCIAL_ORIGIN in in_manifests})
    run.write_state("weights.json", records)


def stage_prepare(run: Run) -> None:
    """Fetch the pinned sample images, validate each request, and show what goes in."""
    from PIL import Image

    from idol_3d_pipeline import SAMPLES, fetch_samples, open_sample, validate_request

    data = fetch_samples(run.weights / SAMPLE_CACHE)
    records, cells = [], []
    for sample in SAMPLES:
        image = open_sample(data[sample["id"]])
        preset = sample["preset"]
        record = validate_request(image, mode=preset["mode"], steps=preset["steps"], num_gaussians=preset["num_gaussians"], name=sample["id"])
        record.update({"source": sample.get("url") or sample["path"], "license": sample["license"], "sha256": sample["sha256"], "preset": preset})
        records.append(record)
        cells.append((f"{sample['id']} ({image.size[0]}x{image.size[1]})", image))
        print({k: record[k] for k in ("id", "size", "mode_pil", "has_alpha", "background_removal", "request", "verdict")})
    probes = (("tiny image", (32, 32), {}), ("unknown mode", (256, 256), {"mode": "sketch"}), ("gaussians not a multiple of 32", (256, 256), {"num_gaussians": 100_001}))
    for probe, size, kwargs in probes:
        try:
            validate_request(Image.new("RGB", size), **kwargs)
        except ValueError as refusal:
            print({"refusal_probe": probe, "refused_with": str(refusal)})
        else:
            raise AssertionError(f"refusal probe '{probe}' was accepted")
    panel([cells], run.out / "inputs.png")
    run.write_state("samples.json", records)


def _generate_one(pipe: Any, run: Run, image: Any, name: str, settings: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    result = pipe.generate(image, name=name, **settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / f"{name}.ply"
    result["gaussian"].save_ply(str(ply))
    result["model_input"].save(out_dir / f"{name}_input.png")
    Image.fromarray((result["mask"] * 255).astype(np.uint8)).save(out_dir / f"{name}_mask.png")
    (out_dir / f"{name}_camera.json").write_text(json.dumps(result["camera"], indent=2), encoding="utf-8")
    np.savez_compressed(run.state / f"{name}.npz", mask=result["mask"], **result["arrays"])
    return {
        "id": name,
        "settings": result["settings"],
        "camera": {k: round(float(v), 4) for k, v in result["camera"].items()},
        "gaussians": int(len(result["arrays"]["xyz"])),
        "seconds": result["seconds"],
        "device": result["device"],
        "mask_replica_matches_sdk": result["mask_replica_matches_sdk"],
        "ply": str(ply.relative_to(run.root)),
        "ply_bytes": ply.stat().st_size,
        "ply_sha256": file_sha256(ply),
    }


def load_pipeline(run: Run) -> Any:
    from idol_3d_pipeline import IdolSplatPipeline

    return IdolSplatPipeline.from_snapshots(run.weights, optimize=bool(run.options.optimize), dinov3_dtype=run.options.dinov3_dtype)


def stage_generate(run: Run) -> None:
    """Load Idol from the verified snapshots and turn each sample into a 3D Gaussian splat."""
    from idol_3d_pipeline import SAMPLES, fetch_samples, open_sample

    require_cuda("generate")
    samples = {r["id"]: r for r in run.read_state("samples.json", "generate")}
    data = fetch_samples(run.weights / SAMPLE_CACHE)
    started = time.perf_counter()
    pipe = load_pipeline(run)
    print({"loaded": "Idol + klein VAE + DINOv3", "device": pipe.device, "optimize": bool(run.options.optimize), "seconds": round(time.perf_counter() - started, 1), **runtime_versions()})
    records = []
    for sample in SAMPLES:
        record = _generate_one(pipe, run, open_sample(data[sample["id"]]), sample["id"], dict(samples[sample["id"]]["preset"]), run.out / sample["id"])
        records.append(record)
        print({k: record[k] for k in ("id", "gaussians", "camera", "seconds", "mask_replica_matches_sdk", "ply", "ply_bytes")})
    from idol_3d_pipeline import SDK_COMMIT

    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    provenance = {"notebook_source": {k: source.get(k) for k in ("repository", "revision", "generator")}, "idol_sdk_commit": SDK_COMMIT, "optimize": bool(run.options.optimize), "dinov3_dtype": run.options.dinov3_dtype, "sample_sha256": {s["id"]: s["sha256"] for s in samples.values()}}
    run.write_output("generation.json", {"samples": records, "provenance": provenance, "runtime": runtime_versions(), "snapshots": pipe.snapshots})
    run.write_state("generated.json", [r["id"] for r in records])


def _evaluate_one(run: Run, name: str, out_dir: Path, convention: str | None = None) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    import numpy as np
    from PIL import Image

    from idol_3d_pipeline import CONVENTIONS, orbit, ply_roundtrip, render, silhouette_iou, splat_summary

    stored = np.load(run.state / f"{name}.npz")
    arrays = {k: stored[k] for k in ("xyz", "scaling", "rotation", "opacity", "rgb")}
    mask = stored["mask"]
    camera = json.loads((out_dir / f"{name}_camera.json").read_text(encoding="utf-8"))
    summary = splat_summary(arrays)
    roundtrip = ply_roundtrip(out_dir / f"{name}.ply", arrays)
    ious = {c: round(silhouette_iou(render(arrays, camera, size=CONVENTION_SIZE, convention=c)["alpha"], mask), 4) for c in CONVENTIONS}
    ranked = sorted(CONVENTIONS, key=lambda c: (-(ious[c] if ious[c] == ious[c] else -1.0), CONVENTIONS.index(c)))  # NaN ranks last
    chosen = convention or ranked[0]
    front = render(arrays, camera, size=VIEW_SIZE, convention=chosen)
    views = [(f"turn {round(math.degrees(t))} deg", render(arrays, orbit(camera, t), size=VIEW_SIZE, convention=chosen)["image"]) for t in TURNTABLE[1:]]
    Image.fromarray(front["image"]).save(out_dir / f"{name}_render_input_view.png")
    record = {
        "id": name,
        "structure": summary,
        "ply_roundtrip": roundtrip,
        "silhouette_iou_by_convention": ious,
        "convention_used": chosen,
        "silhouette_iou": ious[chosen],
        "silhouette_iou_note": "the best of four camera readings, chosen by this IoU, so it is an upper bound for the reading used" if convention is None else "the reading fixed by the caller",
        "renderer": {"pairs": front["pairs"], "cut_at_max_radius": front["cut_at_max_radius"], "kind": "approximate NumPy EWA splatting (render.py), not the 3DGS rasterizer"},
    }
    model_input = Image.open(out_dir / f"{name}_input.png")
    row = [("model input", model_input), ("object mask", (mask * 255).astype(np.uint8)), (f"render, input view (IoU {ious[chosen]:.2f})", front["image"]), *views]
    return record, row


def stage_render(run: Run) -> None:
    """Check each splat: structure, .ply read-back, input-view silhouette IoU; render a turntable preview."""
    names = run.read_state("generated.json", "render")
    records, rows = [], []
    for name in names:
        record, row = _evaluate_one(run, name, run.out / name)
        records.append(record)
        rows.append(row)
        print({"id": name, "verdict": record["structure"]["verdict"], "gaussians": record["structure"]["gaussians"], "ply_verified": record["ply_roundtrip"]["verified"], "silhouette_iou": record["silhouette_iou"], "by_convention": record["silhouette_iou_by_convention"]})
    panel(rows, run.out / "previews.png", tile=VIEW_SIZE)
    conventions = {r["convention_used"] for r in records}
    payload = {
        "samples": records,
        "conventions_chosen": sorted(conventions),
        "consistent_convention": len(conventions) == 1,
        "evidence_level": "structure and file integrity are checked; silhouette IoU covers the visible side only, through an approximate renderer; the unseen sides have no reference here",
    }
    failed = [r["id"] for r in records if r["structure"]["verdict"] != "well-formed" or not r["ply_roundtrip"]["verified"]]
    generation = json.loads((run.out / "generation.json").read_text(encoding="utf-8")) if (run.out / "generation.json").is_file() else {"samples": []}
    failed += [s["id"] for s in generation["samples"] if s.get("mask_replica_matches_sdk") is False]
    payload["verdict"] = "FAILED: " + ", ".join(sorted(set(failed))) if failed else "passed: every splat well-formed, every .ply read back, every mask replica equal to the SDK's input"
    run.write_output("evaluation.json", payload)
    print({"verdict": payload["verdict"], "conventions_chosen": payload["conventions_chosen"], "consistent_convention": payload["consistent_convention"], "previews": "outputs/previews.png", "record": "outputs/evaluation.json"})
    if failed:
        raise RuntimeError(f"checks failed for {sorted(set(failed))}; outputs/evaluation.json records which (the splats are not to be used as results)")


def _byod_images(path_text: str) -> list[tuple[str, Path]]:
    from idol_3d_pipeline import InputRefused

    if not path_text:
        raise InputRefused("BYOD_PATH is empty: point it at an image, a folder of images, or a .zip already in this runtime (or leave it empty in the notebook to get an upload dialog)")
    path = Path(path_text).expanduser()
    if not path.exists():
        raise InputRefused(f"BYOD_PATH {path} does not exist in this runtime")
    archives = [path] if path.is_file() and path.suffix.lower() == ".zip" else (sorted(path.rglob("*.zip")) if path.is_dir() else [])
    for archive_path in archives:
        target = archive_path.with_suffix("")
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                resolved = (target / member.filename).resolve()
                if member.is_dir():
                    continue
                if not resolved.is_relative_to(target.resolve()):
                    print({"skipped_zip_entry": member.filename, "reason": "its path leaves the extraction folder"})
                elif Path(member.filename).suffix.lower() not in BYOD_SUFFIXES:
                    print({"skipped_zip_entry": member.filename, "reason": f"not an image ({', '.join(BYOD_SUFFIXES)})"})
                elif member.file_size >= 64 * 1024 * 1024:
                    print({"skipped_zip_entry": member.filename, "reason": "64 MB or larger"})
                else:
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    resolved.write_bytes(archive.read(member))
    if path.is_file() and path.suffix.lower() == ".zip":
        path = path.with_suffix("")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in BYOD_SUFFIXES)
    if not files:
        raise InputRefused(f"no image ({', '.join(BYOD_SUFFIXES)}) found at {path}")
    if len(files) > BYOD_MAX_IMAGES:
        print({"note": f"{len(files)} images found; the first {BYOD_MAX_IMAGES} are used (each takes as long as one sample in Section 5)"})
    names: set[str] = set()
    out = []
    for file in files[:BYOD_MAX_IMAGES]:
        stem = "byod_" + ("".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in file.stem)[:40] or "image")  # never a sample id
        while stem in names:
            stem += "_"
        names.add(stem)
        out.append((stem, file))
    return out


def stage_byod(run: Run) -> None:
    """Bring Your Own Data: the same validation, generation and checks on your images."""
    import shutil

    from PIL import Image

    from idol_3d_pipeline import InputRefused, validate_request

    opts = run.options
    shutil.rmtree(run.out / f"{STEM}_byod", ignore_errors=True)  # never show results of an earlier BYOD run
    images = _byod_images(opts.byod_path)
    settings = {"mode": opts.byod_mode, "seed": opts.seed, "steps": opts.steps, "num_gaussians": opts.num_gaussians}
    opened = []
    for name, file in images:
        try:
            image = Image.open(file)
            image.load()
        except Exception as error:  # an unreadable file is a refusal, not a crash
            raise InputRefused(f"{file.name}: not a readable image ({type(error).__name__}); supply PNG, JPEG, WebP or BMP") from None
        validate_request(image, mode=settings["mode"], steps=settings["steps"], num_gaussians=settings["num_gaussians"], name=file.name)
        opened.append((name, file, image))
    require_cuda("byod")
    pipe = load_pipeline(run)
    out_dir = run.out / f"{STEM}_byod"
    records, rows = [], []
    for name, file, image in opened:
        generated = _generate_one(pipe, run, image, name, dict(settings), out_dir / name)
        evaluated, row = _evaluate_one(run, name, out_dir / name)
        records.append({"source": file.name, **generated, "evaluation": evaluated})
        rows.append(row)
        print({"id": name, "gaussians": generated["gaussians"], "seconds": generated["seconds"], "ply": generated["ply"], "silhouette_iou": evaluated["silhouette_iou"], "verdict": evaluated["structure"]["verdict"]})
    panel(rows, out_dir / "previews.png", tile=VIEW_SIZE)
    run.write_output(f"{STEM}_byod/byod_result.json", {"settings": settings, "images": records, "data_movement": "files stayed inside this runtime"})
    print({"byod_outputs": str(out_dir.relative_to(run.root)), "images": len(records)})


def stage_activity(run: Run) -> None:
    """Change one thing: regenerate the first sample with fewer Gaussians or fewer steps and compare with the default."""
    from idol_3d_pipeline import SAMPLES, fetch_samples, open_sample

    require_cuda("activity")
    default = {r["id"]: r for r in json.loads((run.out / "evaluation.json").read_text(encoding="utf-8"))["samples"]}
    generated = {s["id"]: s for s in json.loads((run.out / "generation.json").read_text(encoding="utf-8"))["samples"]}
    sample = SAMPLES[0]
    settings = dict(sample["preset"], steps=run.options.steps, num_gaussians=run.options.num_gaussians)
    if (settings["steps"], settings["num_gaussians"]) == (sample["preset"]["steps"], sample["preset"]["num_gaussians"]):
        raise ValueError("nothing changed: set ACTIVITY_STEPS or ACTIVITY_GAUSSIANS to a value other than the default (50 steps, 262,144 Gaussians)")
    name = f"activity_{sample['id']}"
    pipe = load_pipeline(run)
    record = _generate_one(pipe, run, open_sample(fetch_samples(run.weights / SAMPLE_CACHE)[sample["id"]]), name, settings, run.out / name)
    evaluated, row = _evaluate_one(run, name, run.out / name, convention=default[sample["id"]]["convention_used"])
    base = default[sample["id"]]
    comparison = {
        "changed": {k: (sample["preset"][k], settings[k]) for k in ("steps", "num_gaussians") if sample["preset"][k] != settings[k]},
        "seconds": (generated[sample["id"]]["seconds"], record["seconds"]),
        "gaussians": (base["structure"]["gaussians"], evaluated["structure"]["gaussians"]),
        "silhouette_iou_same_convention": (base["silhouette_iou"], evaluated["silhouette_iou"]),
        "ply_bytes": (generated[sample["id"]]["ply_bytes"], record["ply_bytes"]),
        "note": "pairs are (default, changed); one seed and one image, so a difference is an observation about this case, not an effect size",
    }
    panel([row], run.out / name / "activity.png", tile=VIEW_SIZE)
    run.write_output(f"{name}/activity.json", comparison)
    print(comparison)


DEV_VAE = {"model_id": "black-forest-labs/FLUX.2-dev", "revision": "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c", "path": "vae/diffusion_pytorch_model.safetensors"}


def stage_vae_check(run: Run) -> None:
    """Optional: is the klein VAE the FLUX.2-dev VAE in bf16? Needs an HF_TOKEN with FLUX.2-dev access.

    Test 1 compares every dev tensor, cast to bf16, with klein's, bit for bit. Test 2 runs Idol's own
    ``ImageEncoder._flux2_tokens`` with each VAE on the first sample, seeded alike, beside a dev-vs-dev rerun that
    measures this hardware's noise floor. The dev files are downloaded for this check only and never used elsewhere."""
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from idol.conditioning import ImageEncoder
    from idol_3d_pipeline import SAMPLES, fetch_samples, open_sample, verify_snapshot

    if not token():
        raise PermissionError("vae_check needs HF_TOKEN from an account that has accepted the FLUX.2-dev terms at https://huggingface.co/black-forest-labs/FLUX.2-dev")
    verify_snapshot("flux2-vae-klein", run.weights)
    dev_dir = Path(snapshot_download(DEV_VAE["model_id"], revision=DEV_VAE["revision"], allow_patterns=["vae/*"], token=token(), local_dir=str(run.weights / "flux2-vae-dev-check")))
    klein_dir = run.weights / "flux2-vae-klein"
    dev_sd, klein_sd = load_file(str(dev_dir / DEV_VAE["path"])), load_file(str(klein_dir / DEV_VAE["path"]))
    test1: dict[str, Any] = {"dev_file_sha256": file_sha256(dev_dir / DEV_VAE["path"]), "tensors_dev": len(dev_sd), "tensors_klein": len(klein_sd), "only_in_dev": sorted(set(dev_sd) - set(klein_sd)), "only_in_klein": sorted(set(klein_sd) - set(dev_sd)), "bitwise_equal": 0, "not_equal": 0, "max_abs_diff": 0.0}
    for key in sorted(set(dev_sd) & set(klein_sd)):
        a, b = dev_sd[key], klein_sd[key]
        if a.shape != b.shape:
            test1["not_equal"] += 1
            continue
        if a.is_floating_point():
            a = a.to(torch.bfloat16)
        if torch.equal(a, b.to(a.dtype)):
            test1["bitwise_equal"] += 1
        else:
            test1["not_equal"] += 1
            test1["max_abs_diff"] = max(test1["max_abs_diff"], float((a.float() - b.float()).abs().max()))
    test1["pass"] = not test1["only_in_dev"] and not test1["only_in_klein"] and test1["not_equal"] == 0
    print({"test1_weights": test1})
    del dev_sd, klein_sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    image = open_sample(fetch_samples(run.weights / SAMPLE_CACHE)[SAMPLES[0]["id"]]).convert("RGB").resize((1024, 1024))
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0)[None]

    def feature2(folder: Path) -> Any:
        encoder = ImageEncoder(dinov3=torch.nn.Linear(1, 1), flux2_vae=str(folder), device=device)  # DINOv3 is not part of this check
        torch.manual_seed(0)
        with torch.no_grad():
            tokens = encoder._flux2_tokens(pixels.to(device)).float().cpu()
        del encoder
        return tokens

    dev_a, dev_b, klein = feature2(dev_dir), feature2(dev_dir), feature2(klein_dir)
    floor, diff = float((dev_a - dev_b).abs().max()), float((dev_a - klein).abs().max())
    test2 = {"shape": list(dev_a.shape), "dev_vs_dev_max_abs": floor, "dev_vs_klein_max_abs": diff, "equal": bool(torch.equal(dev_a, klein)), "pass": diff <= floor}
    print({"test2_feature2": test2})
    if test1["pass"] and test2["pass"]:
        verdict = "PASS: the klein VAE is the dev VAE in bf16"
    elif test2["pass"]:
        verdict = "PASS (functional only): weights differ, but Idol's VAE conditioning matches within the noise floor"
    else:
        verdict = "FAIL: the klein VAE does not reproduce Idol's dev-VAE conditioning"
    run.write_output("vae_check.json", {"verdict": verdict, "dev": DEV_VAE, "test1_weights": test1, "test2_feature2": test2, "runtime": runtime_versions()})
    print({"verdict": verdict, "record": "outputs/vae_check.json"})


STAGES = {
    "weights": stage_weights,
    "prepare": stage_prepare,
    "generate": stage_generate,
    "render": stage_render,
    "byod": stage_byod,
    "activity": stage_activity,
    "vae_check": stage_vae_check,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the snapshots and the sample cache")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--optimize", action="store_true", help="generate/byod: torch.compile and batched guidance (faster after a slow first image)")
    parser.add_argument("--dinov3-dtype", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--byod-path", default="", help="byod: an image, a directory or a zip")
    parser.add_argument("--byod-mode", default="real", choices=("synthetic", "real"), help="byod: 'real' for photographs, 'synthetic' for renders and generated images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-gaussians", type=int, default=262144)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        if type(exc).__name__ != "InputRefused":  # a refused input is one message, not a traceback (DAT20)
            traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
