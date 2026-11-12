# Release verification

Two clean-runtime executions of `tutorials/idol_3d_colab.ipynb` are recorded, both the default path on a fresh Colab
T4 executed cell by cell by the Colab CLI (2026-11-12, below): at `71564c0`, and at `1338a4a` with the re-fitted
camera calibration. Neither is a browser `Run all`: forms were not rendered and the notebook was not opened from GitHub. The token came from the environment (`colab exec --env HF_TOKEN=…`), not
from the Colab secret, because Colab secrets cannot be read without the Colab UI.

| Date | Runtime | Notebook blob | Source revision | Scope | Result | Evidence |
|---|---|---|---|---|---|---|
| 2026-11-12 (02:26 UTC start) | **fresh Google Colab, Tesla T4** (15,360 MiB), kernel Python 3.13.16; executor **Colab CLI 0.7.5**, driven by `.agent/scripts/colab-serial-test-suite.py` | `5ef5f39421be679296488c202939dbbe4a7f0fd8`, fetched from `1338a4a` and checked before the VM was allocated; generated at `169da75` (the label commit `1338a4a` changes only the label) | `1338a4a` | **Default path with the re-fitted calibration (× 0.725, yaw − 5°), all 12 code cells, 0 errors, 1075 s wall.** Token read from the environment (37 characters); its value occurs in none of the run's files. Environment built in 93 s (73 packages, torch 2.14.0+cu130, CUDA available); `weights` ok in 67.8 s; Idol loaded on CUDA in 74.0 s; `generate` ok in 821.1 s at the full presets (backpack 233.5 s, armchair 252.2 s, porcini 251.8 s), every rebuilt mask equal to the SDK's model input; `render` ok in 62.5 s, all three splats well-formed, every `.ply` read back. Calibrated silhouette IoU **0.7755 (backpack), 0.7586 (armchair), 0.9567 (porcini)**: equal, digit for digit, to the local render-stage re-run in the calibration section below, and every uncalibrated reading equals both earlier T4 runs, so generation reproduced a third time. Activity, `vae_check` and BYOD cells printed their "Skipped" lines. VM released and its absence checked (`stop_verified`). | **pass** for the default path on a hosted T4, first hosted run of the new calibration; Candidate (RUN13/MOD4); not a browser `Run all`; REL12 BYOD and `vae_check` not run | `docs/execution-evidence/2026-11-12-colab-t4-calibrated/idol_3d_colab_1338a4a_colab-cli-t4.ipynb` (1,821,614 bytes, SHA-256 `7920fff6d2163c2db4a927e79282ed5fd486273b72d53342842a7ccce07ca254`) |
| 2026-11-12 | **fresh Google Colab, Tesla T4** (15,360 MiB), kernel Python 3.13.16; executor **Colab CLI 0.7.5** (`colab exec`, cells in order in one kernel), driven by `.agent/scripts/colab-serial-test-suite.py` | `052aa0ed1b983374e5ebfa862b7e4d8adf1bc964`, fetched from `71564c0` and checked before the VM was allocated; generated at `c220621` (the label commit `71564c0` changes only the label) | `71564c0` | **Default path, all 12 code cells, 0 errors, 1109 s wall.** Runtime check: T4 detected. Token cell: `read from the environment (37 characters; not printed, not written to disk)`, so the new fallback is exercised; the token value occurs in none of the run's 5 files (executed notebook, `exec.log`, summary, source copies). Install cell built the locked environment in 130 s (CPython 3.12.12, 73 packages, torch 2.14.0+cu130, CUDA available). `weights` staged and verified all snapshots in 141.3 s, **including DINOv3, the 401 of attempt 1**. Idol loaded on CUDA in 74.9 s and generated all three samples at their full presets (**50 steps, 262,144 Gaussians**): backpack 215.5 s, armchair 222.2 s, porcini 222.1 s (`generate` 744.2 s); every rebuilt mask equals the SDK's model input. `render` (70.2 s): all three splats well-formed, every `.ply` read back. Calibrated silhouette IoU **0.4781 (backpack), 0.7336 (armchair), 0.6847 (porcini)**. The activity, `vae_check` and BYOD cells printed their "Skipped" lines (defaults). VM released and its absence checked (`stop_verified`). | **pass** for the default path on a hosted T4; Candidate (RUN13/MOD4); not a browser `Run all`; REL12 BYOD and `vae_check` not run | `docs/execution-evidence/2026-11-12-colab-t4-attempt-2/idol_3d_colab_71564c0_colab-cli-t4.ipynb` (2,090,916 bytes, SHA-256 `cecdb641253bab929c007366d1bf8a3161d9eb678eeea317590d28e5c628f54c`) |
| 2026-11-11 | **Google Colab, Tesla T4** (15,360 MiB), kernel Python 3.13.16 | executed copy saved by the owner from Colab (`8d6acc9`), generated at `7017778` | `7017778` | Sections 1–3 by the owner: runtime check passed (T4 detected), 25 carried files verified, token cell read the `HF_TOKEN` secret, **install cell built the locked environment in 125 s with CUDA available** (`torch 2.14.0+cu130`, `cuda: True`). The `weights` stage then failed: `GatedRepoError 401 … You must have access to it and be authenticated` on the DINOv3 `config.json`. A 401 means the Hub treated the request as anonymous: huggingface-hub 2.4.0 does send an explicitly passed token (checked), so the secret's token was not accepted. The same account's other token works from the cloud session. Fix in this commit: the `weights` stage now checks the token (`whoami`) and the account's access (`auth_check`) before any download and names the problem. | fail (token), environment pass on T4 | `docs/execution-evidence/2026-11-11-colab-t4-attempt-1/idol_3d_colab_executed.ipynb` |
| 2026-11-10 | cloud session, Linux x86_64, 4 vCPU, 15 GB RAM, **no GPU** | generated at `fc4608a` (infrastructure cells and `weights` stage), stage runner and package at the working tree that became this commit (`render`) | `fc4608a` → this commit | **ENV14 — the locked environment, end to end on CPU.** The notebook's own install cell built the isolated environment from the lock (CPython 3.12.12, 73 packages, torch 2.14.0+cu130, transformers 5.22.2, diffusers 0.42.1) in 381 s; the `weights` stage staged and verified all three snapshots (Idol 2.87 GiB, klein VAE 0.16 GiB, DINOv3 3.13 GiB; FLUX.2-dev in no manifest) in 43.8 s. Idol then loaded from the verified snapshots on CPU (12.9–18.2 s) and generated two splats with the stage runner's own `_generate_one`, reduced for CPU: **12 steps, 65,536 Gaussians** — backpack (`synthetic`, 600 s) and porcini (`real`, 661 s); both rebuilt masks equal the SDK's model input. The `render` stage passed: both splats well-formed, both `.ply` read back. The run exposed that Idol's camera, read as documented, renders the object front-on at half size (silhouette IoU 0.33 / 0.33); the fixed calibration now in `render.py` (field of view × 0.6, yaw + 25°, fitted on these two splats) gives **0.866 (backpack) and 0.874 (porcini)**. | pass for ENV14 and the full code path; **not** REL1 (no GPU, reduced steps and Gaussians, calibration fitted on the same two splats it is scored on) | `docs/execution-evidence/2026-11-10-local-cpu/` (previews.png, evaluation.json, logs) |
| 2026-11-10 | cloud session, NumPy only (no torch) | — | `9b53f47` | **Gate 1, weights: klein VAE vs FLUX.2-dev VAE.** `black-forest-labs/FLUX.2-dev@26afe3a` `vae/diffusion_pytorch_model.safetensors` (336,213,556 bytes, SHA-256 `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5`, F32 + one I64) against `FLUX.2-klein-4B@e7b7dc2` (168,120,878 bytes, SHA-256 `ca70d220…0f04`, BF16 + one I64): same 251 tensor names and shapes; every F32 tensor rounded to bfloat16 (round to nearest, ties to even, as `torch.Tensor.to(torch.bfloat16)`) is bit-for-bit equal to klein's, and the I64 counter is equal; `vae/config.json` differs only in `_diffusers_version` and klein's `_name_or_path`. Idol loads the VAE with `dtype=torch.bfloat16`, so both load to identical parameters. | **PASS** (251/251 bitwise equal, max abs diff 0) | session log; the notebook's `vae_check` stage repeats it with torch and adds Idol's own `feature2` |
| 2026-11-10 | local CPython 3.13, no GPU | generated | uncommitted | Sections 1–2 (runtime check, carrier) executed from the notebook; carried `prepare` stage run with the repository's NumPy/Pillow | pass (not REL1: no isolated environment, no model) | session log only |

## Comparison: 2026-11-12 T4 against the 2026-11-10 CPU run

The two runs differ in settings, so only the parts that do not depend on them can match:

| | CPU, 2026-11-10 | T4, 2026-11-12 |
|---|---|---|
| Locked environment | 73 packages, torch 2.14.0+cu130, built in 381 s | same 73 packages and versions, built in 130 s |
| Snapshots | all verified, 43.8 s | all verified, 141.3 s (downloaded on a fresh VM) |
| Settings | reduced: 12 steps, 65,536 Gaussians, 2 samples | presets: 50 steps, 262,144 Gaussians, 3 samples |
| Generation per sample | 600 s / 661 s | 215.5 s / 222.2 s / 222.1 s |
| Mask replica equals the SDK input | yes (2/2) | yes (3/3) |
| Structure, `.ply` read-back | pass (2/2) | pass (3/3) |
| Calibrated silhouette IoU | backpack 0.866, porcini 0.874 | backpack 0.4781, armchair 0.7336, porcini 0.6847 |

**Open observation: the camera calibration does not transfer.** The fixed calibration in `render.py` (field of view ×
0.6, yaw + 25°) was fitted on the two CPU splats, and it scored 0.866 and 0.874 on them. On the full-setting T4 splats
of the same two images it scores 0.4781 and 0.6847. For the backpack this is below the uncalibrated `toward_camera`
reading (0.5202). The calibration is evidently tied to the reduced-setting splats it was fitted on. It is a diagnostic
on an approximate renderer, not a pass gate (the `render` verdict rests on structure, read-back and the mask replica),
so the run passes, but the IoU numbers should not be quoted as a quality measure until the calibration is re-fitted
on full-setting splats or replaced.

## Camera calibration re-fit, 2026-11-12 (closes the observation above)

**Splats.** The three default-path splats were regenerated on a second fresh Colab T4 (Colab CLI 0.7.5, the same
notebook blob `052aa0ed…` at `71564c0`, 12/12 cells, 0 errors) and downloaded before the VM was released. Every
silhouette IoU of that run, calibrated and uncalibrated, equals the recorded run's above, and re-scoring the
downloaded `state/*.npz` locally reproduces 0.4781 / 0.7336 / 0.6847, so the local scoring is the stage's own.

**Fit.** `tools/fit_camera_calibration.py`, leave-one-out: each splat is scored with the calibration that fits the
other two best. Coarse grid (field-of-view scale 0.4–1.2 by 0.1, yaw ±60° by 10°, pitch ±10° by 10°), then a fine grid
(0.65–0.75 by 0.025, yaw −40° to +20° by 5°, pitch ±5° by 5°); `toward_camera` convention, 256 px, as the stage.

| | uncalibrated | 2026-11-10 calibration (× 0.6, + 25°) | **held out** (fine grid) | new calibration, in-sample |
|---|---|---|---|---|
| backpack | 0.5202 | 0.4781 | **0.7755** | 0.7755 |
| armchair | 0.4503 | 0.7336 | **0.6812** | 0.7586 |
| porcini | 0.5121 | 0.6847 | **0.8682** | 0.9567 |
| mean | 0.494 | 0.632 | **0.775** | 0.830 |

Every held-out score beats its uncalibrated reading, the acceptance rule, so the calibration fitted on all three is
now in `render.py`: **field of view × 0.725, yaw − 5°, pitch unchanged**. The field-of-view scale is stable across
folds (0.7–0.725); the yaw offset is not (the best yaw per object is about −30° for the backpack, +10° for the armchair
and −10° to 0° for the porcini), so the yaw is a compromise and the held-out mean (0.775) is the figure to quote, not
the in-sample one. The `render` stage itself, re-run locally on the downloaded splats with the new calibration, gives
0.7755 / 0.7586 / 0.9567 with the same structure and read-back verdicts. That re-run is local, not a hosted run of the
new notebook; three objects and one seed each are a small sample. Records:
`docs/execution-evidence/2026-11-12-calibration-fit/` (`fit_coarse.json`, `fit_fine.json` with every IoU in
`fit_fine.tables.json`, and the local render stage's `evaluation_rerender.json` and `previews_rerender.png`).

A hosted Colab T4 `Run all` is required before any Release-grade claim; the RUN13/MOD4 deviation (see `STATUS.md`)
caps the notebook at Candidate regardless.
