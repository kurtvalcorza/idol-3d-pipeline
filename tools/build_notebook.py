#!/usr/bin/env python3
"""Generate the STANDALONE DIMER tutorial notebook (NOTEBOOK_SPEC 3.1 §4) from repository sources — /3.1-token.

Vendored from kurtvalcorza/swin2sr-x4-super-resolution-pipeline (build_notebook.py/3.0) and changed for one gated
model: an optional ``token`` setup cell reads ``HF_TOKEN`` from the hosted runtime's secrets, the install cell passes
it to the stage processes only, and the Prerequisites name the credential instead of saying none is needed
(NOTEBOOK_SPEC RUN13 / MOD4 deviation, recorded in STATUS.md).

/3 replaces the in-kernel install of /2.x with the isolated, hash-locked environment of the version 2.2 reference
notebook (§25.13): nothing is pip-installed into the notebook kernel, so a hosted runtime's preloaded packages are
never replaced and no restart is ever needed (RUN1, RUN10, ENV6). The notebook

1. checks the runtime (Linux x86_64 required; a CUDA GPU is used when present, otherwise the CPU) and the disk, and
   creates a fresh run directory ``ROOT``;
2. writes the carried files — the repository's package under ``src/``, the stage runner, the hash-locked
   requirements, the pinned snapshot manifests, the licence and ``source.json`` — to ``ROOT`` and verifies each
   against ``CARRIED_HASHES`` (the carried text stays visible in that cell: ST5, SRC12);
3. downloads a pinned ``uv`` wheel (URL + size + SHA-256), builds a managed-Python virtual environment, installs the
   lock with ``--require-hashes --only-binary :all:`` into it, and defines ``run_stage``, which runs one stage of
   the carried runner per process and re-raises a failed stage's error message in the kernel;
4. runs the template's learner cells, which call ``run_stage(...)`` and display the files it writes.

The carried files are the repository's files byte for byte (read as UTF-8 text, newlines normalised to LF); the
parity tests fail whenever the carrier and the repository diverge. This file is vendored per repository.

Usage (from the repository root, or with --repo):
    python tools/build_notebook.py            # write tutorials/<notebook_name>
    python tools/build_notebook.py --check    # exit 1 if the committed notebook differs (PAR3)
    python tools/build_notebook.py --out PATH # write elsewhere (review copies)
"""
# ruff: noqa: E501  -- learner-facing prose and generated code are kept on single lines so they render readably
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "build_notebook.py/3.1-token"
NOTEBOOK_SPEC = "3.1"
SOURCE_RECORD = "source.json"


def template_contract() -> dict[str, str]:
    """Keys ``TEMPLATE`` must define (documentation for template authors). Optional keys are marked."""
    return {
        "package": "import name of the repository package",
        "repo_name": "GitHub repository name",
        "stem": "output file stem (notebook cell ids, run directory, export names)",
        "notebook_name": "tutorials/<notebook_name>",
        "profile": "TASK-INFERENCE | MULTI-CAPABILITY | E2E | ARTIFACT-INFERENCE",
        "mode": "REFERENCE | GUIDED | WORKSHOP",
        "run_all": "the Run-all declaration (§28)",
        "byod": "the BYOD declaration (§28)",
        "title": "H1 text",
        "badges": "list of (alt, image_url, link_url)",
        "capability": "one-line capability statement",
        "intro": "markdown paragraphs after the header block (no heading)",
        "learning_objectives": "markdown after the bold label",
        "exclusions": "markdown after the bold label",
        "prerequisites": "list of markdown bullets; the generator appends the External access bullet",
        "weights_key": "MODEL_KEY value (weights/<key>/dimer-base-manifest.json)",
        "carried": "ordered {destination under ROOT: repository-relative source} of every carried file",
        "stage_runner": "destination of the carried stage runner (a key of `carried`)",
        "lock": "destination of the carried hash-locked requirements (a key of `carried`)",
        "managed_python": "exact CPython version uv installs for the isolated environment",
        "uv": "{'version', 'url', 'bytes', 'sha256'} of the pinned manylinux x86_64 uv wheel",
        "disk_gib": "{'weights', 'environment'} free-space needs in GiB",
        "runtime_modules": "modules whose versions the isolated environment prints, e.g. ['torch', 'diffusers']",
        "setup": "list of {'md', 'cell' ('check'|'carrier'|'token'|'install'|'weights'), 'after' (optional)} infrastructure cells",
        "credential": "OPTIONAL markdown: the credential the default path needs (None: none needed)",
        "cells": "list of {'md': str, 'code': str} learner cells; may use {stem}, {MODEL_ID}, {MODEL_REVISION}",
        "closing": "markdown for Interpretation and limits + References",
        "guided": "OPTIONAL {'opening': [markdown cells after the header]}",
    }


REQUIRED_KEYS = [k for k, v in template_contract().items() if not v.startswith("OPTIONAL")]


def load_template(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("notebook_template", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = module.TEMPLATE
    missing = [k for k in REQUIRED_KEYS if k not in template]
    if missing:
        raise SystemExit(f"template missing keys: {missing}")
    return template


def _pins(repo: Path) -> list[str]:
    """The `==` runtime pins of pyproject.toml (ENV2); the lock is compiled from exactly these."""
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    if not block:
        raise SystemExit("pyproject.toml: dependencies block not found")
    pins = re.findall(r'"([^"]+)"', block.group(1))
    bad = [p for p in pins if "==" not in p]
    if bad:
        raise SystemExit(f"unpinned runtime dependency (ENV2): {bad}")
    return pins


def lock_packages(lock_text: str) -> dict[str, str]:
    """`{name: version}` of every requirement in a uv/pip-compile hash lock."""
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", lock_text, re.M)}


def check_lock(pins: list[str], lock_text: str) -> None:
    """Every direct pin must appear in the lock at the same version, and every lock entry must carry a hash."""
    locked = lock_packages(lock_text)
    for pin in pins:
        name, version = pin.split("==", 1)
        if locked.get(name.lower()) != version:
            raise SystemExit(f"lock does not pin {pin} (found {locked.get(name.lower())}); recompile the lock")
    blocks = re.split(r"\n(?=[A-Za-z0-9])", lock_text.split("\n", 2)[-1])
    unhashed = [b.split("==", 1)[0] for b in blocks if "==" in b and "--hash=sha256:" not in b]
    if unhashed:
        raise SystemExit(f"lock entries without --hash: {unhashed}")


def _head_revision(repo: Path) -> str:
    """HEAD at generation time: a provenance label only; parity is anchored on file content (see ``--check``)."""
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        out = ""
    return out or "uncommitted"


def recorded_revision(notebook_path: Path) -> str | None:
    """`generated_from.revision` of an existing notebook, or None."""
    if not notebook_path.exists():
        return None
    try:
        meta = json.loads(notebook_path.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
        return str(meta["revision"])
    except (KeyError, ValueError, TypeError):
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def carried_files(repo: Path, template: dict[str, Any], revision: str) -> dict[str, str]:
    """`{destination: text}` of every carried file, plus the generated `source.json` provenance record last."""
    files: dict[str, str] = {}
    for dest, source in template["carried"].items():
        path = repo / source
        if not path.is_file():
            raise SystemExit(f"carried source missing: {source}")
        files[dest] = path.read_text(encoding="utf-8")
    for key in (template["stage_runner"], template["lock"]):
        if key not in files:
            raise SystemExit(f"template names {key!r} but does not carry it")
    check_lock(_pins(repo), files[template["lock"]])
    record = {
        "repository": f"kurtvalcorza/{template['repo_name']}",
        "revision": revision,
        "generator": GENERATOR_VERSION,
        "notebook_spec": NOTEBOOK_SPEC,
        "files": {dest: sha256_text(text) for dest, text in files.items()},
        "sources": dict(template["carried"]),
    }
    files[SOURCE_RECORD] = json.dumps(record, indent=2) + "\n"
    return files


def load_context(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    pkg = template["package"]
    revision = revision or _head_revision(repo)
    files = carried_files(repo, template, revision)
    modules = [d for d in template["carried"] if d.startswith(f"src/{pkg}/") and d.endswith(".py")]
    pipeline = files[f"src/{pkg}/pipeline.py"]
    ident: dict[str, str] = {}
    for name in ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY"):
        m = re.search(rf'^{name} = "([^"]+)"$', pipeline, re.M)
        if not m:
            raise SystemExit(f"pipeline.py: {name} not found as a top-level string constant")
        ident[name] = m.group(1)
    manifests = {d: json.loads(t) for d, t in files.items() if d.startswith("weights/") and d.endswith("dimer-base-manifest.json")}
    main_manifest = manifests.get(f"weights/{template['weights_key']}/dimer-base-manifest.json")
    if main_manifest is None:
        raise SystemExit("the main snapshot manifest is not carried (MOD13)")
    if (main_manifest["modelId"], main_manifest["revision"]) != (ident["MODEL_ID"], ident["MODEL_REVISION"]):
        raise SystemExit("manifest identity != module identity")
    if template["weights_key"] != ident["MODEL_KEY"]:
        raise SystemExit("template weights_key != MODEL_KEY")
    return {
        "pkg": pkg,
        "revision": revision,
        "files": files,
        "hashes": {d: sha256_text(t) for d, t in files.items()},
        "modules": modules,
        "module_sha256": hashlib.sha256("".join(files[m] for m in modules).encode("utf-8")).hexdigest(),
        "manifests": manifests,
        "lock_packages": len(lock_packages(files[template["lock"]])),
        "pins": _pins(repo),
        **ident,
    }


def _md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": "", "metadata": {}, "source": source.rstrip("\n")}


def _code(source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "id": "", "metadata": metadata or {}, "outputs": [], "source": source.rstrip("\n")}


# GDL11: collapsed-by-default metadata for infrastructure cells (Colab form view; Jupyter source_hidden).
INFRASTRUCTURE_METADATA: dict[str, Any] = {"cellView": "form", "jupyter": {"source_hidden": True}}
MODES = ("REFERENCE", "GUIDED", "WORKSHOP")


# ---- infrastructure cell sources -------------------------------------------------------------------------------------

CHECK_CELL = """# @title Infrastructure: check the runtime, accelerator and disk; create or keep this session's run directory
NEW_RUN_DIRECTORY = False  # @param {{type:"boolean"}}
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

SESSION_START = time.perf_counter()
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('This notebook needs a Linux x86_64 runtime (Google Colab, Kaggle, or a Linux Jupyter kernel): its locked environment is built for manylinux x86_64 wheels. Windows, macOS and ARM kernels are refused.')
try:
    probe_gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
    GPU = probe_gpu.stdout.strip() if probe_gpu.returncode == 0 and probe_gpu.stdout.strip() else None
except FileNotFoundError:
    GPU = None
STEM = {stem!r}
REQUIRE_GPU = {require_gpu!r}
RUNS = Path.cwd() / 'outputs' / STEM
# Re-running this cell on its own keeps the run directory of this session, so the cells after it keep working; only a
# new kernel (or NEW_RUN_DIRECTORY) starts a fresh one, which then needs Sections 2 and 3 again.
previous_root = globals().get('ROOT')
if not NEW_RUN_DIRECTORY and isinstance(previous_root, Path) and previous_root.parent == RUNS and previous_root.is_dir():
    ROOT = previous_root
    print('Keeping the run directory of this session:', ROOT, '(tick NEW_RUN_DIRECTORY for a fresh one, then run Sections 2 and 3 again).')
else:
    ROOT = RUNS / uuid.uuid4().hex[:12]
    ROOT.mkdir(parents=True)
WEIGHTS = Path.cwd() / 'weights'
WEIGHTS.mkdir(exist_ok=True)
# The isolated environment is keyed on the lock digest, the managed Python and the uv version, not on the run: a later
# Run all in the same runtime reuses a complete environment built from the same lock instead of building another.
ENV_KEY = {env_key!r}
ENV_ROOT = Path(tempfile.gettempdir()) / (STEM + '_env_' + ENV_KEY)
staged_gib = sum(p.stat().st_size for p in WEIGHTS.rglob('*') if p.is_file()) / 1024**3
need = {{'weights': max(0.0, {weights_gib} - staged_gib), 'environment': 0.0 if (ENV_ROOT / 'ready.json').is_file() else {env_gib}}}
free = {{'weights': shutil.disk_usage(WEIGHTS).free / 1024**3, 'environment': shutil.disk_usage(tempfile.gettempdir()).free / 1024**3}}
if os.stat(WEIGHTS).st_dev == os.stat(tempfile.gettempdir()).st_dev:
    short = free['weights'] < need['weights'] + need['environment']
else:
    short = free['weights'] < need['weights'] or free['environment'] < need['environment']
if short:
    raise RuntimeError(f'Not enough free disk: need about {{need}} GiB, free {{free}} GiB. Start a fresh runtime (see Troubleshooting).')
if REQUIRE_GPU and not GPU:
    raise RuntimeError('No GPU is attached. ' + REQUIRE_GPU + ' In Colab choose Runtime -> Change runtime type -> T4 GPU, then Run all; nothing has been downloaded yet.')
print({{'gpu': GPU or 'none: the stages will run on the CPU (slower; see Prerequisites)', 'kernel_python': platform.python_version(), 'run_directory': str(ROOT), 'weights': str(WEIGHTS), 'environment': str(ENV_ROOT), 'free_gib': {{k: round(v, 1) for k, v in free.items()}}}})"""

CARRIER_CELL = """# @title Infrastructure: write and verify the carried package, stage runner, lock and manifests
CARRIED_FILES = {files}
CARRIED_HASHES = {hashes}
for name, text in CARRIED_FILES.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\\n')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:
        raise RuntimeError('Carried file integrity failure: ' + name + '. Do not edit this cell; regenerate the notebook from the repository.')
NOTEBOOK_SOURCE = json.loads((ROOT / {source!r}).read_text(encoding='utf-8'))
print({{'carried_files': len(CARRIED_FILES), 'verified': True, 'repository': NOTEBOOK_SOURCE['repository'], 'revision': NOTEBOOK_SOURCE['revision'], 'generator': NOTEBOOK_SOURCE['generator']}})"""

INSTALL_CELL = """# @title Infrastructure: install the locked runtime into an isolated environment and define the stage runner
import io
import urllib.error
import urllib.request
import zipfile

from IPython.display import Image, display

UV_URL = {uv_url!r}
UV_BYTES = {uv_bytes}
UV_SHA256 = {uv_sha256!r}
# The stage processes get no kernel Python path and no token from the kernel's environment; the only token they see is
# the one the Token cell read from this runtime's secrets (HF_TOKEN_VALUE), used for the gated snapshot alone. The kernel
# may export an inline matplotlib backend that the isolated environment cannot import; stages write figures to files.
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', UV_CACHE_DIR=str(ENV_ROOT / 'cache'), MPLBACKEND='Agg', ATTN_BACKEND='sdpa')
for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
    ENV.pop(name, None)
# Only these stages receive the token: uv, the package install, the probe and every other stage run without it.
TOKEN_STAGES = {token_stages!r}
PYTHON = ENV_ROOT / 'venv' / 'bin' / 'python'
ENV_SPEC = {{'lock_sha256': CARRIED_HASHES[{lock!r}], 'python': {python!r}, 'uv': {uv_version!r}}}
try:
    ENV_REUSED = PYTHON.is_file() and json.loads((ENV_ROOT / 'ready.json').read_text(encoding='utf-8')) == ENV_SPEC
except (OSError, ValueError):
    ENV_REUSED = False
if ENV_REUSED:
    print('Reusing the isolated environment built earlier in this runtime from the same lock:', ENV_ROOT / 'venv')
else:
    (ENV_ROOT / 'ready.json').unlink(missing_ok=True)
    shutil.rmtree(ENV_ROOT / 'venv', ignore_errors=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(UV_URL, timeout=90) as response:
                wheel = response.read(UV_BYTES + 1)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
        raise RuntimeError('uv {uv_version} wheel size/hash mismatch: refusing to run it')
    ENV_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        member = next(n for n in archive.namelist() if n.endswith('.data/scripts/uv'))
        UV = ENV_ROOT / 'uv'
        UV.write_bytes(archive.read(member))
    UV.chmod(0o700)
    subprocess.run([str(UV), 'venv', '--managed-python', '--python', {python!r}, str(ENV_ROOT / 'venv')], env=ENV, check=True)
    subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:', '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / {lock!r})], env=ENV, check=True)
    # Written last: an interrupted build leaves no marker and is rebuilt from scratch by the next run of this cell.
    (ENV_ROOT / 'ready.json').write_text(json.dumps(ENV_SPEC), encoding='utf-8')
probe = subprocess.run([str(PYTHON), '-c', {probe!r}], env=ENV, check=True, capture_output=True, text=True)
RUNTIME = json.loads(probe.stdout.strip().splitlines()[-1])
print({{'notebook_source': NOTEBOOK_SOURCE['revision'], **RUNTIME, 'locked_packages': {n_locked}, 'environment': str(ENV_ROOT / 'venv'), 'environment_reused': ENV_REUSED, 'setup_seconds': round(time.perf_counter() - SESSION_START)}})
if GPU and not RUNTIME['cuda']:
    print('Note: a GPU is attached but the isolated environment cannot use it; the stages will run on the CPU. See Troubleshooting.')


def run_stage(stage, *options):
    \"\"\"Run one stage of the carried runner in its own process with the isolated interpreter; stream its output.\"\"\"
    if not (ROOT / {runner!r}).is_file() or not PYTHON.is_file():
        raise RuntimeError('The run directory ' + str(ROOT) + ' has no carried files, or the isolated environment is gone: run the three Infrastructure cells again in order (Sections 1, 2 and 3), then this cell, or choose Runtime -> Run all.')
    log = ROOT / 'logs' / (stage + '.log')
    log.parent.mkdir(exist_ok=True)
    error_file = ROOT / 'state' / (stage + '.error.json')
    error_file.unlink(missing_ok=True)
    command = [str(PYTHON), '-u', str(ROOT / {runner!r}), '--root', str(ROOT), '--weights', str(WEIGHTS), '--stage', stage, *map(str, options)]
    print('Running stage', repr(stage), 'in the isolated environment; log:', log, flush=True)
    with log.open('w', encoding='utf-8') as output:
        stage_env = dict(ENV, HF_TOKEN=HF_TOKEN_VALUE) if stage in TOKEN_STAGES and globals().get('HF_TOKEN_VALUE') else ENV
        process = subprocess.Popen(command, env=stage_env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output.write(line)
            output.flush()
            if line.strip() and len(line) < 4000:
                print(line.rstrip(), flush=True)
        process.wait()
    if process.returncode:
        detail = 'see the log above'
        if error_file.is_file():
            error = json.loads(error_file.read_text(encoding='utf-8'))
            detail = error['type'] + ': ' + error['message']
        raise RuntimeError(f'Stage {{stage!r}} failed (exit {{process.returncode}}): {{detail}}')


def load_record(name):
    \"\"\"A JSON record a stage wrote to ROOT/outputs.\"\"\"
    return json.loads((ROOT / 'outputs' / name).read_text(encoding='utf-8'))


def show_image(name, caption=None):
    \"\"\"Display an image a stage wrote to ROOT/outputs.\"\"\"
    path = ROOT / 'outputs' / name
    print(caption or name, '->', path)
    display(Image(filename=str(path)))"""

WEIGHTS_CELL = """# @title Infrastructure: stage and digest-verify the pinned snapshots
run_stage('weights')"""

TOKEN_CELL = """# @title Infrastructure: read HF_TOKEN from this runtime's secrets (needed for the gated DINOv3 encoder)
HF_TOKEN_VALUE, HF_TOKEN_SOURCE = None, 'secrets'
try:
    from google.colab import userdata
    try:
        HF_TOKEN_VALUE = userdata.get('HF_TOKEN')
    except Exception as error:  # the secret is missing, this notebook has no access to it, or there is no Colab UI (Colab CLI)
        HF_TOKEN_VALUE, HF_TOKEN_SOURCE = os.environ.get('HF_TOKEN'), 'the environment'
        if not HF_TOKEN_VALUE:
            raise RuntimeError('No HF_TOKEN secret is readable: in Colab open the key icon on the left, add a secret named HF_TOKEN holding a token of a Hugging Face account that has accepted the terms at https://huggingface.co/{gated}, switch on Notebook access, then run this cell again. (' + type(error).__name__ + ')') from None
except ImportError:  # not Colab: Kaggle secrets, then the environment
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN_VALUE = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        HF_TOKEN_VALUE, HF_TOKEN_SOURCE = os.environ.get('HF_TOKEN'), 'the environment'
if not HF_TOKEN_VALUE:
    raise RuntimeError('HF_TOKEN is empty: add a token of a Hugging Face account that has accepted the terms at https://huggingface.co/{gated} as a secret named HF_TOKEN, then run this cell again.')
print({{'hf_token': 'read from ' + HF_TOKEN_SOURCE + ' (' + str(len(HF_TOKEN_VALUE)) + ' characters; not printed, not written to disk)', 'used_for': {gated!r}}})"""


def environment_key(ctx: dict[str, Any], template: dict[str, Any]) -> str:
    """Directory key of the isolated environment: the lock digest, the managed Python and the uv version (not the run)."""
    basis = "\n".join((ctx["hashes"][template["lock"]], template["managed_python"], template["uv"]["version"]))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _probe(modules: list[str]) -> str:
    versions = ", ".join(f"'{m}': {m}.__version__" for m in modules)
    return f"import json, platform, {', '.join(modules)}; print(json.dumps({{'python': platform.python_version(), {versions}, 'cuda': torch.cuda.is_available()}}))"


def _infrastructure_code(key: str, ctx: dict[str, Any], template: dict[str, Any]) -> str:
    if key == "check":
        disk = template["disk_gib"]
        return CHECK_CELL.format(require_gpu=template.get("require_gpu", ""), stem=template["stem"], weights_gib=float(disk["weights"]), env_gib=float(disk["environment"]), env_key=environment_key(ctx, template))
    if key == "carrier":
        return CARRIER_CELL.format(files=repr(ctx["files"]), hashes=repr(ctx["hashes"]), source=SOURCE_RECORD)
    if key == "install":
        uv = template["uv"]
        return INSTALL_CELL.format(
            uv_url=uv["url"],
            uv_bytes=int(uv["bytes"]),
            uv_sha256=uv["sha256"],
            uv_version=uv["version"],
            python=template["managed_python"],
            lock=template["lock"],
            probe=_probe(template["runtime_modules"]),
            n_locked=ctx["lock_packages"],
            runner=template["stage_runner"],
            token_stages=sorted(template.get("token_stages", [])),
        )
    if key == "weights":
        return WEIGHTS_CELL
    if key == "token":
        return TOKEN_CELL.format(gated=template["gated_model"])
    raise SystemExit(f"unknown setup cell {key!r}")


def render(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    ctx = load_context(repo, template, revision)
    mode = template["mode"]
    if mode not in MODES:
        raise SystemExit(f"template mode {mode!r} is not one of {MODES}")
    stem = template["stem"]
    fmt = {"stem": stem, **{k: ctx[k] for k in ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")}, "n_locked": ctx["lock_packages"]}
    cells: list[dict[str, Any]] = []

    def add(cell: dict[str, Any]) -> None:
        cell["id"] = f"{stem}-{len(cells):02d}"
        cells.append(cell)

    badges = " ".join(f"[![{alt}]({img})]({link})" for alt, img, link in template["badges"])
    total_mb = ctx["manifests"][f"weights/{template['weights_key']}/dimer-base-manifest.json"]["totalBytes"] / 1e6  # the main snapshot only
    pinned = ctx["MODEL_REVISION"] != "unpinned"
    revision_text = (
        f"the Hugging Face Hub at the immutable revision `{ctx['MODEL_REVISION']}` (~{total_mb:.0f} MB, digest-verified before loading)"
        if pinned
        else f"the Hugging Face Hub (~{total_mb:.0f} MB) — but only once the snapshot is pinned: this revision of the notebook carries "
        "**no immutable revision yet** (`MODEL_REVISION = \"unpinned\"`), so Section 3 stops with an explanation instead of downloading "
        "unverified weights (see **Pin status** in the Prerequisites)"
    )
    header = (
        f"# {template['title']}\n\n{badges}\n\n"
        f"**Profile:** `{template['profile']}`  \n"
        f"**Mode:** `{mode}`  \n"
        f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)  \n"
        f"**Capability:** {template['capability']}\n\n"
        f"**This notebook is standalone.** Section 2 carries the repository's package ({len(ctx['modules'])} modules under "
        f"`src/{ctx['pkg']}/`, at revision `{ctx['revision'][:12]}`){template.get('carried_extra', '')}, the stage runner, the hash-locked requirements "
        f"({ctx['lock_packages']} packages), the {len(ctx['manifests'])} snapshot manifests and the licence, and verifies every carried file "
        f"against its SHA-256 before use, so the notebook keeps working after export even if the repository changes or disappears. "
        f"Nothing is installed into the notebook kernel: a pinned `uv` (checked by size and SHA-256) builds an isolated Python "
        f"environment from the lock with `--require-hashes`, and every stage runs there in its own process, so the hosted "
        f"runtime's own packages are never replaced and no restart is needed. Its only external dependencies are PyPI, the "
        f"managed CPython build that `uv` downloads, {revision_text}, the further pinned snapshots and the sample source "
        f"named in the Prerequisites. It was generated by "
        f"`tools/build_notebook.py` ({GENERATOR_VERSION}); edit the repository and regenerate rather than editing cells.\n\n"
        f"**Run all:** {template['run_all'].strip()}\n\n"
        f"**Bring Your Own Data:** {template['byod'].strip()}\n\n"
        f"{template['intro'].strip()}\n\n"
        f"**Learning objectives:** {template['learning_objectives'].strip()}\n\n"
        f"**This notebook does not demonstrate:** {template['exclusions'].strip()}"
    )
    add(_md(header))
    for opening in (template.get("guided") or {}).get("opening", []):
        add(_md(opening.format(**fmt)))

    prereq = list(template["prerequisites"]) + [
        f"- **External access:** the Hugging Face Hub, to fetch the `{ctx['MODEL_ID']}` snapshot "
        f"(~{total_mb:.0f} MB) at the revision in the carried manifest (`{ctx['MODEL_REVISION'][:12]}`); {template.get('external_extra', '')}"
        f"PyPI (`files.pythonhosted.org`), for the pinned `uv` wheel and the {ctx['lock_packages']} hash-locked packages; and "
        f"the managed CPython build (python-build-standalone) that `uv` downloads for the isolated environment. No repository "
        + ("clone is required; nothing is installed from this repository. **Credential:** " + template["credential"].strip() if template.get("credential") else "clone and no credentials are required; nothing is installed from this repository.")
    ]
    add(_md("## Prerequisites\n\n" + "\n".join(p.format(**fmt) for p in prereq)))

    for setup in template["setup"]:
        add(_md(setup["md"].format(**fmt)))
        metadata: dict[str, Any] = dict(INFRASTRUCTURE_METADATA)
        if setup["cell"] == "carrier":
            metadata["dimer"] = {"embedded_sources": True, "files": dict(ctx["hashes"])}
        add(_code(_infrastructure_code(setup["cell"], ctx, template), metadata))
        if setup.get("after"):
            add(_md(setup["after"].format(**fmt)))

    for stage in template["cells"]:
        add(_md(stage["md"].format(**fmt)))
        if stage.get("code"):
            add(_code(stage["code"].format(**fmt)))
    add(_md(template["closing"].format(**fmt)))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",  # recommended; the stages fall back to the CPU
            "colab": {"gpuType": "T4", "name": template["notebook_name"], "provenance": []},
            "dimer": {
                "notebook_profile": template["profile"],
                "notebook_kind": template.get("kind", "TUTORIAL"),
                "notebook_mode": mode,
                "notebook_spec": NOTEBOOK_SPEC,
                "standalone": True,
                "requires_dimer_worker": False,
                "environment": "isolated hash-locked uv environment; nothing installed into the kernel",
                "generated_from": {
                    "repository": template["repo_name"],
                    "revision": ctx["revision"],
                    "module": f"src/{ctx['pkg']}/pipeline.py",
                    "modules": ctx["modules"],
                    "module_sha256": ctx["module_sha256"],
                    "files": dict(ctx["hashes"]),
                    "generator": GENERATOR_VERSION,
                },
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def to_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--template", type=Path, default=None, help="default: <repo>/tools/notebook_template.py")
    parser.add_argument("--out", type=Path, default=None, help="default: <repo>/tutorials/<notebook_name>")
    parser.add_argument("--check", action="store_true", help="exit 1 if the existing notebook differs (PAR3)")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    template = load_template(args.template or repo / "tools" / "notebook_template.py")
    out = args.out or repo / "tutorials" / template["notebook_name"]
    if args.check:
        # The recorded revision is a provenance label carried through the check; drift is caught by content — a changed
        # carried file changes the carrier cell and its hashes, so the byte comparison fails regardless of the label.
        rendered = to_bytes(render(repo, template, recorded_revision(out)))
        current = out.read_bytes().replace(b"\r\n", b"\n") if out.exists() else b""
        if current != rendered:
            print(f"STALE: {out} differs from the generator output; run tools/build_notebook.py", file=sys.stderr)
            return 1
        print(f"OK: {out} is up to date ({len(rendered)} bytes)")
        return 0
    rendered = to_bytes(render(repo, template))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rendered)
    print(f"wrote {out} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
