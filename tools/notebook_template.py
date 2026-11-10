"""Per-repository template for tools/build_notebook.py /3.1-token (NOTEBOOK_SPEC 3.1 §4 standalone, §25.14 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, token, isolated install + stage runner, snapshot
staging) from repository files; this template holds the learner-facing prose, the list of carried files and the learner
cells. Every learner cell calls ``run_stage(...)``: the carried ``tutorial_stages.py`` (``tools/`` in the repository)
runs one stage per process in an isolated, hash-locked environment, so nothing is installed into the notebook kernel.

TASK-INFERENCE workflow: three pinned snapshots are staged and digest-verified (Idol, the Apache-2.0 FLUX.2 [klein]
VAE that replaces Idol's FLUX.2-dev VAE, and the gated DINOv3 encoder fetched with the runtime's HF_TOKEN), two pinned
sample images from the Idol SDK are validated, Idol turns each into a 3D Gaussian splat written as a standard 3DGS
``.ply``, and every splat is checked (structure, ``.ply`` read-back, input-view silhouette agreement) and previewed.
Two optional branches follow: the klein-vs-dev VAE equivalence check, and Bring Your Own Data.

Every string below goes through ``str.format``: literal braces in prose and code are doubled.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "idol-3d-pipeline"
NOTEBOOK = "idol_3d_colab.ipynb"
GATED = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

BADGES = [
    ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", f"https://github.com/kurtvalcorza/{REPO}"),
    ("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg", f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/{NOTEBOOK}"),
    ("Hugging Face", "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-speridlabs%2Fidol-ffcc4d?style=flat", "https://huggingface.co/speridlabs/idol"),
    ("Upstream", "https://img.shields.io/badge/Upstream-speridlabs%2Fidol--sdk-181717?style=flat&logo=github&logoColor=white", "https://github.com/speridlabs/idol-sdk"),
    ("License", "https://img.shields.io/badge/License-Apache--2.0-blue.svg", "https://www.apache.org/licenses/LICENSE-2.0"),
]

SDK_FILES = [
    "__init__.py",
    "camera.py",
    "conditioning.py",
    "device.py",
    "gaussian.py",
    "pipeline.py",
    "preprocess.py",
    "sampler.py",
    "models/__init__.py",
    "models/birefnet.py",
    "models/decoder.py",
    "models/dit.py",
    "models/layers.py",
]
PACKAGE_FILES = ["__init__.py", "pipeline.py", "render.py", "metrics.py", "samples.py"]

CARRIED = {f"src/idol_3d_pipeline/{name}": f"src/idol_3d_pipeline/{name}" for name in PACKAGE_FILES}
CARRIED.update({f"src/idol/{name}": f"src/idol/{name}" for name in SDK_FILES})
CARRIED.update(
    {
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/idol/dimer-base-manifest.json": "weights/idol/dimer-base-manifest.json",
        "weights/flux2-vae-klein/dimer-base-manifest.json": "weights/flux2-vae-klein/dimer-base-manifest.json",
        "weights/dinov3-vith16plus/dimer-base-manifest.json": "weights/dinov3-vith16plus/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    }
)

TEMPLATE = {
    "package": "idol_3d_pipeline",
    "repo_name": REPO,
    "stem": "idol_3d",
    "notebook_name": NOTEBOOK,
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "gated_model": GATED,
    "credential": (
        f"a Hugging Face token, read from this runtime's secrets as `HF_TOKEN` by the Section 2 Token cell, of an account that has accepted the DINOv3 terms at https://huggingface.co/{GATED} (approval is manual and can take a while). "
        "It is passed only to the `weights` stage (for that gated snapshot) and, if you switch it on, the optional `vae_check` stage (for FLUX.2-dev); uv, the package install and every other stage run without it, and it is never printed or written to disk. "
        "Because the default path needs it, this notebook does not meet NOTEBOOK_SPEC RUN13/MOD4 (nor RUN14 or REL6, which follow from them) and stays a Candidate (see STATUS.md)."
    ),
    "token_stages": ["weights", "vae_check"],
    "kind": "TUTORIAL",
    "require_gpu": "Idol's generation stage needs a CUDA GPU (on a CPU it would take an estimated tens of minutes per image), so this notebook stops here rather than after building the environment and downloading about 6.6 GB.",
    "carried_extra": ", the Idol SDK carried verbatim (13 modules under `src/idol/`, upstream commit `8db7003`)",
    "external_extra": "the `black-forest-labs/FLUX.2-klein-4B` VAE (`vae/` only, 168 MB) and the gated `facebook/dinov3-vith16plus-pretrain-lvd1689m` encoder (3.36 GB) at the revisions in their manifests, with the token below; `raw.githubusercontent.com` and `inaturalist-open-data.s3.amazonaws.com` for the three sample images; ",
    "run_all": (
        "Selecting **Run all** in a fresh Linux runtime with a CUDA GPU (a Colab T4 is the documented target) and an `HF_TOKEN` secret (see **Credential** in the Prerequisites) builds an isolated Python "
        "environment from the carried hash-locked requirements (torch, transformers, diffusers, tensordict and their dependencies) without touching the notebook kernel's own packages, then runs each "
        "stage below in its own process: it stages and digest-verifies three pinned snapshots (Idol's 3.08 GB of weights, the 0.17 GB FLUX.2 [klein] VAE and the 3.36 GB DINOv3 ViT-H+/16 encoder; 6.6 GB in all), fetches three "
        "pinned sample images (two from the Idol SDK, one CC0 photograph), validates them, turns each into a 3D Gaussian splat of 262,144 Gaussians written as a standard 3DGS `.ply`, and checks and previews every splat. "
        "The default path needs no repository clone, no DIMER worker or service, no upload dialog, no configuration edit and no runtime restart; it does need the token above. No hosted run of this "
        "notebook has been recorded yet (see `docs/release-verification.md`)."
    ),
    "byod": (
        "After the default path completes, Section 9 runs the same validation, generation and checks on your own images: set `USE_BYOD = True`, choose `BYOD_MODE` — `'real'` for photographs "
        "(a looser crop) or `'synthetic'` for renders and generated images — and either set `BYOD_PATH` to an image, folder or zip already in the runtime or leave it empty to get an upload dialog. "
        "Up to four images are used; each takes about as long as one sample in Section 5. Uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "weights_key": "idol",
    "carried": CARRIED,
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 7.5, "environment": 9},
    "runtime_modules": ["torch", "transformers", "diffusers", "numpy", "PIL"],
    "title": "Idol — DIMER Guided Notebook: one image to a 3D Gaussian splat (task inference, standalone)",
    "badges": BADGES,
    "capability": "single-image to 3D Gaussian splat generation with Idol (a 0.37 B-parameter flow-matching DiT with octree and Gaussian decoders), on Apache-2.0 FLUX.2 [klein] VAE conditioning in place of the non-commercial FLUX.2 [dev] VAE, with structural, file and input-view silhouette checks",
    "intro": (
        "Idol [1] turns one picture of an object into a 3D Gaussian splat: up to a million small, coloured, semi-transparent 3D blobs that together form the object, including the "
        "sides the picture never showed. Six networks run in order. **BiRefNet** [5] cuts the object out and the image is framed on black at 1024 × 1024. Two image encoders read "
        "it: **DINOv3** [4] for what the object is, and the **FLUX.2 VAE** [7] for how its pixels look. A **diffusion transformer** trained with flow matching [3] "
        "turns noise into an 8,192 × 16 object latent and a predicted camera in 50 Euler steps with classifier-free guidance. An **octree decoder** places anchor points on a 256³ grid, and a "
        "**Gaussian decoder** turns each anchor into 32 Gaussians [2], written as a `.ply` any splat viewer opens.\n\n"
        "One change sets this notebook apart from the upstream SDK. Idol's own configuration loads the VAE from `black-forest-labs/FLUX.2-dev`, which is under the FLUX Non-Commercial License. "
        "This notebook loads the same VAE architecture from `black-forest-labs/FLUX.2-klein-4B`, released under Apache-2.0; its configuration names FLUX.2-dev as its origin, and the optional "
        "Section 8 tests whether the weights are the same. Nothing is trained here: this is a pretrained model used as published, which is why the profile is task inference."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** the input → output contract of single-image 3D generation and what each of Idol's five networks contributes (Sections 4–5).\n"
        "2. **Identify** an invalid request from a validation refusal before any model runs (Section 4).\n"
        "3. **Interpret** a Gaussian splat's structure: count, bounds, opacity and scale (Section 6).\n"
        "4. **Compute and interpret** the silhouette agreement between a splat rendered from the predicted camera and the object mask, and **state** what that check cannot establish about the unseen sides (Section 6).\n"
        "5. **Explain** how a licence constraint was met by replacing one component, and what evidence would show the replacement changes nothing (Sections 3 and 8).\n"
        "6. **Write** an evidence-based conclusion that names the checks run and their limits (Interpretation and conclusion)."
    ),
    "exclusions": (
        "fine-tuning or training (Idol publishes no training recipe), scene or multi-object reconstruction, text-to-3D, mesh extraction, the 3D Gaussian Splatting CUDA rasterizer (previews use an "
        "approximate NumPy renderer), any quality score for the unseen sides (there is no ground-truth 3D here), and any use of FLUX.2-dev on the default path."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime with a **CUDA GPU**; a Colab T4 (16 GB) is the documented target. Idol's card gives a 7.2 GB VRAM peak per request. The generation stage refuses a CPU-only runtime with a message (tens of minutes per image otherwise). The kernel's own Python version does not matter: the notebook installs nothing into it and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages. The Section 1 check asks for about 9 GB of free disk for that environment and 5.5 GB for the weights and samples.",
        "- **Time:** not yet measured for this notebook; the figures here are estimates. Idol's card reports 54.6 s per image on an L4 with compilation on; this notebook runs eagerly by default (`OPTIMIZE = False` in Section 5), which is slower but avoids a compile step on the T4. Downloading about 6.6 GB of weights takes several minutes.",
        "- **Precision, seeds and variation:** DINOv3 and the FLUX.2 VAE run in bf16, BiRefNet in fp16 on CUDA, the denoiser and decoders as the SDK sets them (fp16 transformer blocks with fp32 embeddings and norms). Each sample uses its preset seed (42 for the backpack, 7 for the armchair, 42 for the porcini photograph); BYOD uses seed 42. GPU kernels are not bitwise deterministic, so splats and the scores in Section 6 can differ slightly between runs and GPUs.",
        "- **Knowledge:** you can run notebook cells and read short Python. No prior experience with 3D reconstruction is assumed; the glossary defines the terms.",
        "- **Weights:** only `.safetensors` files are loaded, each checked by size and SHA-256 against the carried manifests before any model code runs. The Idol SDK is carried verbatim at commit `8db7003` (Apache-2.0) and verified like every carried file; no Hub-hosted code is executed. Idol's weights and the klein VAE are Apache-2.0; DINOv3 is under the DINOv3 License.",
        "- **Pin status:** all three snapshots are pinned by commit and per-file size and SHA-256 (DINOv3's recorded with access to the gated repository on 2026-11-10); any mismatch stops Section 3 before a model loads.",
        "- **Data contract:** one RGB or RGBA image of one object, 64..4096 px per side. Without a real alpha channel BiRefNet predicts the object mask. Output: a 3DGS `.ply` (z-up object turned y-up for viewers), the 1024 × 1024 image Idol saw, its object mask, and the predicted camera (yaw, pitch, radius, field of view).",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing, and BYOD files stay in this runtime.",
        "- **External access (samples):** two pinned images (70 KB) from `raw.githubusercontent.com/speridlabs/idol-sdk` at commit `8db7003` (an Apache-2.0 repository; the SDK does not state how they were made) and one CC0 photograph (379 KB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com`, each digest-verified before decoding.",
    ],
    "guided": {
        "opening": [
            (
                "## Use notice\n\n"
                "Provided for research, training and evaluation purposes only. The model weights are fetched unmodified from their upstream locations under their own licences (Idol and the FLUX.2 [klein] VAE: Apache-2.0; DINOv3: the DINOv3 License — Built with DINOv3), which control your use; this notebook's code is Apache-2.0. Everything is supplied as is, without warranty, and has not been validated for production or safety-critical use. Running it downloads third-party weights and uses compute on your own account. The splats are predictions: the sides the picture does not show are generated, not measured."
            ),
            (
                "## Problem statement\n\n"
                "A single photograph of an object fixes only one side of it. Turning it into a 3D object a viewer can rotate means predicting the rest. This notebook asks what a learner can check about such a prediction "
                "without any 3D ground truth, and shows how a licence constraint on one component (the FLUX.2-dev VAE) was met without retraining the model.\n\n"
                "## Data context\n\n"
                "Three default inputs. Two are example images from the Idol SDK's own demo (`demo/examples/` in [speridlabs/idol-sdk](https://github.com/speridlabs/idol-sdk/tree/8db700349a5570aac390e4780fa32e3eed6b587a/demo/examples) "
                "at commit `8db7003`, Apache-2.0), run with the settings in its `presets.json`. The SDK does not say how they were made; their style suggests renders or generated images, the `synthetic` "
                "input kind Idol's default mode is for. The third is a CC0 photograph of one porcini mushroom from iNaturalist (observation [187524159](https://www.inaturalist.org/observations/187524159), photo 327959642, by moorebuesing), "
                "run in `real` mode, so both crop settings and a natural background are on the default path. Idol's training data is not described upstream, so the demo images may resemble, or come from the same source as, what the model was trained on: treat "
                "every result here as tutorial evidence about these three images, not as an estimate of how Idol does on new objects. Your own photographs go through Section 9 in `real` mode."
            ),
            (
                "## How to use this notebook\n\n"
                "**Running it.** In Colab choose *Runtime → Change runtime type → T4 GPU*, add the `HF_TOKEN` secret (key icon on the left; switch on notebook access), then *Runtime → Run all*. "
                "Sections 1–3 build the isolated environment and download about 5 GB of weights before any model runs; read ahead while they finish.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls `run_stage('…')`, which runs one stage of the carried stage runner "
                "in its own process with the isolated environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it fails.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–9) are the workflow; *Infrastructure cells* (Sections 1–3) are collapsed and titled **Infrastructure**. You may run them without "
                "studying them: they exist for reproducibility and provenance.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an expectation; a collapsed **Check your reasoning** answer follows."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Framing** | one image of one object | BiRefNet mask (unless the image has alpha) → 1 px erosion → square crop at 1.2× (synthetic) or 1.7× (real) the object → 1024 px on black | the model input and its object mask |\n"
                "| **Generation** | the model input | DINOv3 tokens + FLUX.2 VAE latents → flow-matching DiT, 50 steps, guidance 3.0 → octree decoder → Gaussian decoder | 262,144 Gaussians and a predicted camera |\n"
                "| **Checks** | the Gaussians, the `.ply`, the mask, the camera | structure and bounds; `.ply` read-back; render from the predicted camera and compare silhouettes | a record per sample and preview images |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens |\n"
                "|---|---|---|\n"
                "| 1–3 | [Engineering] | runtime check; carried code, token and isolated environment; three snapshots staged and verified |\n"
                "| 4 | [Evaluation practice] | three sample images (two synthetic, one photograph) and refusal probes |\n"
                "| 5 | [Concept] | Idol generates a splat for each sample |\n"
                "| 6 | [Evaluation practice] | structure, file and silhouette checks; previews |\n"
                "| 7 | [Engineering] | the files you can take away; 7b optional activity: change one thing |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion |\n"
                "| 8 | [Evaluation practice] | optional: is the klein VAE the dev VAE? (needs FLUX.2-dev access) |\n"
                "| 9 | [Evaluation practice] | optional: Bring Your Own Data |"
            ),
            (
                "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **3D Gaussian splat** | A 3D object represented as many anisotropic Gaussians, each with a position, scale, rotation, opacity and colour, drawn by blending their 2D footprints. |\n"
                "| **3DGS `.ply`** | The file layout of Kerbl et al. (2023): one vertex per Gaussian with fields `x … rot_3`; splat viewers read it. |\n"
                "| **Flow matching** | Training a network to predict the velocity that carries noise to data; sampling integrates that velocity in steps. |\n"
                "| **Classifier-free guidance** | Each step also runs with the image conditioning zeroed and moves away from that result; strength 3.0 here. |\n"
                "| **VAE** | Variational autoencoder; here only its encoder is used, to turn the image into a 128-channel latent grid. |\n"
                "| **Octree** | A cube split recursively into eight children; Idol's decoder decides how many points each child gets, down to a 256³ grid. |\n"
                "| **Silhouette IoU** | Intersection over union of two binary shapes: the rendered splat's coverage and the object mask, 1.0 when they coincide. |\n"
                "| **Gated model** | A Hub repository that serves files only to accounts that accepted its terms. |\n"
                "</details>"
            ),
        ]
    },
    "setup": [
        {"md": "## 1. Check the runtime [Engineering]\n\nThis cell refuses anything but Linux x86_64, reports the GPU, checks free disk, and creates this session's run directory.", "cell": "check"},
        {"md": "## 2. Carry the code, read the token, install the runtime [Engineering]\n\nThe next cell writes the repository's package, the vendored Idol SDK, the stage runner, the hash lock and the three snapshot manifests into the run directory, and verifies each against its SHA-256.", "cell": "carrier"},
        {"md": "The Token cell reads `HF_TOKEN` from this runtime's secrets. Only the gated DINOv3 snapshot needs it; the stage processes receive it, the kernel does not print it, and nothing writes it to disk.", "cell": "token"},
        {"md": "The install cell downloads a pinned `uv` (checked by size and SHA-256), builds an isolated CPython 3.12.12 environment, installs the {n_locked} locked packages with `--require-hashes`, and defines `run_stage`.", "cell": "install"},
        {
            "md": "## 3. Pin, stage and verify the snapshots [Engineering]\n\nThree snapshots are fetched at their pinned commits and every file is checked by size and SHA-256: Idol (`speridlabs/idol`, Apache-2.0), the VAE of FLUX.2 [klein] 4B (Apache-2.0, ungated; only `vae/` is fetched), and DINOv3 ViT-H+/16 (gated, DINOv3 License). The stage also reads Idol's `config.json` and confirms it names exactly the two encoders this package replaces.",
            "cell": "weights",
            "after": "**What to notice.** Every snapshot reports `verified: True`, and the last line says the non-commercial FLUX.2-dev origin is not contacted: Idol's config still names it, but the local klein snapshot takes its place before any model code runs.",
        },
    ],
    "cells": [
        {
            "md": "## 4. Sample inputs and validation [Evaluation practice]\n\nTwo example images ship with the Idol SDK, each with the settings its own demo uses (`synthetic` mode); the third is a CC0 photograph of a porcini mushroom on a forest floor (`real` mode). The stage fetches them by pinned URL, checks their digests, validates each request, and then shows that three invalid requests are refused with a message before any model runs.\n\n**Predict before running.** None of the three has an alpha channel. Which network will produce their object masks, and which of the three do you expect it to find hardest to cut out?",
            "code": "run_stage('prepare')\nshow_image('inputs.png', 'Sample inputs: two Idol SDK demo images and one CC0 iNaturalist photograph')",
        },
        {
            "md": "<details>\n<summary><strong>Check your reasoning</strong></summary>\n\nBiRefNet: an image with no real alpha channel gets a predicted mask (`background_removal` says so). An RGBA image with transparent pixels would skip it and use its own alpha. The photograph is the hardest case: moss, needles and leaves around the stem are the kind of clutter a mask can include or bite into, which Section 6's mask column lets you check. The refusal probes show the contract is enforced before the GPU is touched.\n</details>\n\n## 5. Generate the splats [Concept]\n\nIdol is loaded from the verified snapshots with both encoders replaced, then each sample runs with its upstream preset: 50 flow-matching steps, guidance 3.0, 262,144 Gaussians. `OPTIMIZE` turns on `torch.compile` and batched guidance: faster after a slow first image, but the compile step is the riskiest part on a T4, so it is off by default.\n\n**Predict before running.** Idol's card quotes 7.2 GB of VRAM and under a minute per image on an L4. Will a T4 running eagerly be faster or slower, and by roughly how much?",
            "code": "OPTIMIZE = False  # @param {{type:\"boolean\"}}\nrun_stage('generate', *(['--optimize'] if OPTIMIZE else []))\ngeneration = load_record('generation.json')\nfor sample in generation['samples']:\n    print({{k: sample[k] for k in ('id', 'gaussians', 'seconds', 'camera', 'mask_replica_matches_sdk', 'ply_bytes')}})",
        },
        {
            "md": "<details>\n<summary><strong>Check your reasoning</strong></summary>\n\nSlower. A T4 has roughly half an L4's fp16 throughput, and eager mode gives up the card's ~1.7× speed-up from compilation and batched guidance, so several times the L4 figure is a reasonable expectation. The measured `seconds` below replace that estimate for this runtime.\n</details>\n\n**What to notice.** Each sample reports 262,144 Gaussians, a predicted camera, and `mask_replica_matches_sdk: True`, meaning the mask this package rebuilt comes from exactly the image Idol saw. The `seconds` are this runtime's measurement; compare them with your prediction.\n\n## 6. Check and preview each splat [Evaluation practice]\n\nThere is no ground-truth 3D for these images, so the checks are the ones that need none. **Structure:** the Gaussians are finite, their centres within ±0.55 of the origin (Idol's [-0.5, 0.5]³ box plus a 0.05 margin for the decoder's offsets), with unit rotations and opacities in [0, 1]. **File:** the `.ply` reads back with the same count, the standard 3DGS fields and the same positions to within 10⁻⁵. If a structure or file check fails, or the rebuilt mask does not match the SDK's input, the stage stops after writing its record. **Input view:** the splat is rendered from the camera Idol predicted, and its silhouette is compared with the object mask (IoU). Idol's SDK does not document which way its camera direction points or the image's handedness, so the stage scores all four readings and reports each; the reported IoU is the best of the four, so it is an upper bound for the reading it picks.\n\n**Predict before running.** If the camera and the renderer agree with Idol's, should the silhouette IoU at the input view be near 0.5, near 0.9, or near 1.0? Why not exactly 1.0?",
            "code": "run_stage('render')\nshow_image('previews.png', 'Model input, mask, render from the predicted camera, and three turned views')\nevaluation = load_record('evaluation.json')\nfor sample in evaluation['samples']:\n    print({{'id': sample['id'], 'structure': sample['structure']['verdict'], 'ply_verified': sample['ply_roundtrip']['verified'], 'silhouette_iou': sample['silhouette_iou'], 'by_convention': sample['silhouette_iou_by_convention']}})\nprint({{'convention': evaluation['conventions_chosen'], 'consistent': evaluation['consistent_convention']}})",
        },
        {
            "md": "<details>\n<summary><strong>Check your reasoning</strong></summary>\n\nHigh, but not 1.0. The previews use an approximate NumPy renderer (footprints cut at 3σ and 24 px, colour from the degree-0 harmonic only), the mask was eroded by 1 px, and Idol predicts its camera rather than being told it. One convention should score far above the others; if all four are low, either the camera reading or the splat is off, and the row says which numbers to look at. A high IoU says the visible side lines up. It says nothing about the back of the object.\n</details>\n\n## 7. Outputs [Engineering]\n\nEach sample's folder under `outputs/` holds the `.ply`, the model input, its mask, the predicted camera and the input-view render; `generation.json` records the settings, timings, digests and snapshots; `evaluation.json` records every check. Open a `.ply` in any Gaussian-splat viewer for a faithful view.",
            "code": "for path in sorted((ROOT / 'outputs').rglob('*')):\n    if path.is_file():\n        print(path.relative_to(ROOT), path.stat().st_size, 'bytes')",
        },
        {
            "md": "## 7b. Optional activity: change one thing [Concept]\n\nOff by default. **Predict → Change one thing → Run → Observe → Explain.** Pick one change for the first sample (the backpack): fewer flow-matching steps (`ACTIVITY_STEPS`, e.g. 25) or fewer Gaussians (`ACTIVITY_GAUSSIANS`, e.g. 32768). Before you tick `RUN_ACTIVITY`, write down what you expect to happen to the time, the `.ply` size and the silhouette IoU. The stage regenerates the backpack with your change, scores it with the same camera reading as the default run, and prints (default, changed) pairs.",
            "code": "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\nACTIVITY_STEPS = 25  # @param {{type:\"integer\"}}\nACTIVITY_GAUSSIANS = 262144  # @param [\"32768\", \"65536\", \"131072\", \"262144\", \"524288\"] {{type:\"raw\"}}\nif RUN_ACTIVITY:\n    run_stage('activity', '--steps', ACTIVITY_STEPS, '--num-gaussians', ACTIVITY_GAUSSIANS)\n    show_image('activity_backpack/activity.png', 'Your change, scored the same way as the default')\nelse:\n    print('Skipped: tick RUN_ACTIVITY to try one change on the backpack.')",
        },
        {
            "md": "<details>\n<summary><strong>Check your reasoning</strong></summary>\n\nFewer steps cut the denoiser's time almost proportionally (each step is two transformer passes) but leave the decoders' time unchanged; the shape usually survives and fine detail goes first, which a silhouette barely registers. Fewer Gaussians shrink the `.ply` in proportion and speed up the decoders, not the denoiser; the silhouette can stay close because each Gaussian grows to cover the same surface. If the IoU moves by a few hundredths, remember it is one image and one seed.\n</details>\n\n## What did not work\n\n- **Loading the VAE from FLUX.2-dev**, as Idol's own configuration does: its non-commercial licence rules it out for this fleet. The FLUX.2 [klein] 4B copy replaced it after a bit-for-bit comparison of all 251 tensors (in bf16, the precision Idol uses).\n- **`gsplat`, the usual Gaussian-splat rasterizer**: it has no binary wheel that a hash-locked, binary-only install can use, so the previews use the small NumPy renderer in `render.py` instead.\n- **Assuming one camera convention**: Idol's SDK does not document the direction or handedness of its predicted camera, so Section 6 scores four readings rather than guessing one.\n- **Installing the Idol SDK from GitHub**: it is not on PyPI, and a source build would need tools outside the lock, so the notebook carries the SDK's files verbatim instead.",
        },
        {
            "md": (
                "## Interpretation and conclusion [Evaluation practice]\n\n"
                "**What this run shows.** Idol ran from digest-verified local snapshots: DINOv3 is the same gated model Idol names, and the VAE is the Apache-2.0 FLUX.2 [klein] copy in place of FLUX.2-dev. It produced well-formed splats whose `.ply` files read back with the same count, fields and positions. The silhouette IoU says how well the visible side lines up with the input, through an approximate renderer [6].\n\n"
                "**What it does not show.** That the unseen sides are right: there is no reference for them here. That the klein VAE equals the dev VAE: Section 8 tests that and needs FLUX.2-dev access. That three images represent anything beyond themselves: two come from the model's own demo and may resemble its undisclosed training data, so this is tutorial evidence, not a performance estimate.\n\n"
                "**Write your conclusion** in three sentences: what was generated, which checks passed with which numbers, and the one limit you consider most important.\n\n"
                "## 8. Optional: is the klein VAE the FLUX.2-dev VAE? [Evaluation practice]\n\n"
                "Off by default. With `RUN_VAE_CHECK` ticked and an `HF_TOKEN` whose account has accepted the FLUX.2-dev terms, the `vae_check` stage downloads the dev VAE (336 MB) for this check only, compares every tensor cast to bf16 with the klein VAE bit for bit, and runs Idol's own VAE encoding with each on the same image beside a dev-vs-dev rerun that measures this GPU's noise floor. FLUX.2-dev is never used for generation."
            ),
            "code": "RUN_VAE_CHECK = False  # @param {{type:\"boolean\"}}\nif RUN_VAE_CHECK:\n    run_stage('vae_check')\n    print(load_record('vae_check.json')['verdict'])\nelse:\n    print('Skipped: tick RUN_VAE_CHECK to run the klein-vs-dev VAE check (needs FLUX.2-dev access).')",
        },
        {
            "md": "## 9. Optional: Bring Your Own Data [Evaluation practice]\n\nOff by default. Tick `USE_BYOD`, choose `BYOD_MODE` (`'real'` for photographs, `'synthetic'` for renders and generated images), and either set `BYOD_PATH` to an image, folder or zip in this runtime or leave it empty for an upload dialog. Up to four images are used. A refused image stops with one message naming the file and the rule.",
            "code": "USE_BYOD = False  # @param {{type:\"boolean\"}}\nBYOD_MODE = 'real'  # @param [\"real\", \"synthetic\"]\nBYOD_PATH = ''  # @param {{type:\"string\"}}\nif USE_BYOD:\n    if not BYOD_PATH:\n        from google.colab import files\n        upload_dir = ROOT / 'byod_upload'\n        shutil.rmtree(upload_dir, ignore_errors=True)  # a fresh upload replaces the last one\n        upload_dir.mkdir()\n        for name, data in files.upload().items():\n            (upload_dir / name).write_bytes(data)\n        BYOD_PATH = str(upload_dir)\n    try:\n        run_stage('byod', '--byod-path', BYOD_PATH, '--byod-mode', BYOD_MODE)\n    except RuntimeError as failure:\n        if not str(failure).split('): ', 1)[-1].startswith('InputRefused:'):\n            raise  # a genuine failure, not a refused input\n        print('BYOD refused your input:', str(failure).split(': ', 1)[-1])\n        print('Fix the input named above, then run this cell again; the default results above are unchanged.')\n    else:\n        show_image('idol_3d_byod/previews.png', 'Your images')\nelse:\n    print('Skipped: tick USE_BYOD to run your own images.')",
        },
    ],
    "closing": (
        "## Troubleshooting\n\n"
        "- **`No HF_TOKEN secret is readable`** — add the secret, switch on notebook access, rerun the Token cell, then the cells after it.\n"
        "- **`403` or `gated` while staging DINOv3** — the token's account has not been approved for DINOv3 yet; approval is manual.\n"
        "- **`has no recorded size/SHA-256`** — the DINOv3 manifest is not pinned yet; the maintainer runs `tools/pin_snapshot.py dinov3-vith16plus` and regenerates the notebook.\n"
        "- **`No GPU is attached` (Section 1) or `needs a CUDA GPU`** — switch the runtime to a T4 GPU and Run all again.\n"
        "- **A GPU is attached but `needs a CUDA GPU` still appears** — the isolated environment's torch is the CUDA 13 build; a runtime whose NVIDIA driver is older cannot run it. Start a fresh Colab runtime (Colab's current images carry a newer driver) and report the driver version (`nvidia-smi`) if it persists.\n"
        "- **`Not enough free disk`** — start a fresh runtime; the weights (about 6.6 GB) and the environment (about 9 GB) need room.\n"
        "- **`differs from the manifest this notebook carries`** — the weights folder came from another notebook revision: delete the folder named in the message, or start a fresh runtime.\n"
        "- **Out of GPU memory** — restart the runtime (*Runtime → Restart session*: the isolated environment and the weights on disk are kept and reused) and keep `OPTIMIZE = False`.\n\n"
        "## References\n\n"
        "[1] Sperid Labs. 2026. Idol: a single image to a 3D Gaussian splat. Model card and SDK. https://huggingface.co/speridlabs/idol ; https://github.com/speridlabs/idol-sdk\n\n"
        "[2] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4, Article 139. https://doi.org/10.1145/3592433\n\n"
        "[3] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2023. Flow Matching for Generative Modeling. In ICLR 2023. https://arxiv.org/abs/2210.02747\n\n"
        "[4] Oriane Siméoni et al. 2025. DINOv3. arXiv:2508.10104. https://arxiv.org/abs/2508.10104\n\n"
        "[5] Peng Zheng et al. 2024. Bilateral Reference for High-Resolution Dichotomous Image Segmentation. CAAI Artificial Intelligence Research 3. https://doi.org/10.26599/AIR.2024.9150038\n\n"
        "[6] Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and Markus Gross. 2002. EWA Splatting. IEEE Trans. Vis. Comput. Graph. 8, 3, 223–238. https://doi.org/10.1109/TVCG.2002.1021576\n\n"
        "[7] Black Forest Labs. 2026. FLUX.2 [klein] 4B. https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
    ),
}
