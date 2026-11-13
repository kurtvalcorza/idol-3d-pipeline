# Idol 3D pipeline — one image to a 3D Gaussian splat (DIMER, task inference)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/idol-3d-pipeline/blob/main/tutorials/idol_3d_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-speridlabs%2Fidol-ffcc4d?style=flat)](https://huggingface.co/speridlabs/idol)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Status: Candidate, not yet run on a hosted runtime** — see [`STATUS.md`](STATUS.md).

[Idol](https://huggingface.co/speridlabs/idol) turns one picture of an object into a 3D Gaussian splat, written as a
standard 3DGS `.ply`. This repository packages it for the DIMER fleet with one change: Idol's configuration loads its
VAE from `black-forest-labs/FLUX.2-dev` (FLUX Non-Commercial License); this pipeline loads the same VAE architecture
from `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, ungated) instead, and never contacts FLUX.2-dev on the default
path. DINOv3 ViT-H+/16 is still fetched from its gated origin with the runtime's `HF_TOKEN`.

## What is here

| Path | What it is |
|---|---|
| `src/idol_3d_pipeline/` | the package: snapshot manifests and verification, the input contract, the wrapper that builds Idol from local verified snapshots with both encoders replaced, a NumPy splat renderer, and the checks |
| `src/idol/` | the Idol SDK, carried verbatim from [`speridlabs/idol-sdk@8db7003`](https://github.com/speridlabs/idol-sdk/tree/8db700349a5570aac390e4780fa32e3eed6b587a) (Apache-2.0); hashes in `docs/upstream-sdk.json` |
| `weights/*/dimer-base-manifest.json` | pinned revisions and per-file sizes and SHA-256 for Idol, the klein VAE and DINOv3 (DINOv3 digests pending, see `STATUS.md`) |
| `tools/` | the notebook generator and template, the stage runner the notebook carries, and the pin tool |
| `tutorials/` | the generated standalone notebook and its hash lock |

## Running it

Open the notebook in Colab, choose a T4 GPU, add an `HF_TOKEN` secret from an account that has accepted the DINOv3
terms, and choose **Run all**. Nothing is installed into the Colab kernel: a pinned `uv` builds an isolated
environment from the carried lock and every stage runs there.

## Developing

```bash
python -m pip install numpy==2.5.4 pillow==13.0.0 plyfile==1.1.5 huggingface-hub==2.4.0 pytest==8.4.2 ruff==0.17.4 hypothesis==6.170.15
python -m pip install -e . --no-deps
ruff check src/idol_3d_pipeline tests tools && pytest
python tools/build_notebook.py          # regenerate after changing anything the notebook carries
uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt
```

The renderer's maths (Idol's camera, the EWA footprint, compositing, the `.ply` axes) is also formalised in Lean 4
(`lean/`): 34 machine-checked theorems, each mirrored by a property test in `tests/test_properties.py`, and an
executable Lean reference that `tests/test_lean_reference.py` compares with the Python. Setup, the map from functions
to theorems, and what the formalisation found are in [`docs/LEAN.md`](docs/LEAN.md).

## Licences

This repository's code: Apache-2.0. Idol weights and SDK: Apache-2.0. FLUX.2 [klein] 4B VAE: Apache-2.0. DINOv3:
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) — Built with DINOv3. The
weights are not redistributed here; the notebook fetches them from their origins at pinned revisions.
