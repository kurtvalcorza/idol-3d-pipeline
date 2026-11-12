# Status — Candidate (default path passed on a hosted Colab T4 via the Colab CLI, 2026-11-12, at `1338a4a`)

**2026-11-10.** Built on branch `ccr-deb2a494-n7dsf1`, reviewed twice (a torch-path review and a NOTEBOOK_SPEC 3.1
review; their findings are fixed in `791b564` and after). CI-level checks pass locally (ruff; 46 tests without torch;
notebook parity). From the generated notebook, the runtime-check and carrier cells and the carried `prepare` and
`weights` stages were run locally: 25 carried files verified; all three samples fetched and digest-checked; the
refusal probes refused; the `weights` stage installed the carried manifests and stopped on the unpinned DINOv3
before downloading anything. **No model has run yet:** torch and the model stages have never executed here.

## Gate 1 (klein VAE = FLUX.2-dev VAE): passed at the weight level

On 2026-11-10 every one of the 251 FLUX.2-dev VAE tensors, rounded to bf16 as torch does, was bit-for-bit equal to
the klein VAE (record in `docs/release-verification.md`). Idol loads the VAE in bf16, so the parameters it uses are
identical on this route. The notebook's optional `vae_check` stage repeats this with torch and adds Idol's own
conditioning on a GPU.

## Pins: complete

DINOv3 access was granted on 2026-11-10 and its manifest is pinned (`model.safetensors` 3,362,432,800 bytes, SHA-256
`3e1d4d18…1a4d`; `config.json`; `LICENSE.md`), recorded by `tools/pin_snapshot.py` from the Hub's records at
`c807c9e`. Nothing blocks a hosted run any more; it needs a Colab T4 and an `HF_TOKEN` secret of an account with
DINOv3 access.

## Conformance boundary

- **RUN13 / MOD4 — deviation, by decision (Kurt, 2026-11-10):** the default path needs `HF_TOKEN` for the gated
  DINOv3 encoder. The same deviation fails RUN14, §3.1, REL6 and §31(4). This caps the notebook at Candidate. Re-hosting DINOv3 (its licence permits redistribution with the
  licence text and "Built with DINOv3") would remove the deviation.
- **NOTEBOOK_SPEC 3.1** is declared. Closed after review: the Predict → Change → Run activity (Section 7b, GDL10), a
  use-notice cell, a "What did not work" section, and the Model Card's short sections (MODEL_CARD_SPEC G13). ENV14 is done: on
  2026-11-10 the notebook's install cell built the locked environment, every snapshot verified, and Idol generated and
  passed the checks on two samples on CPU (12 steps, 65,536 Gaussians; `docs/release-verification.md`).
- **Samples:** two Idol SDK demo images (Apache-2.0 repository; how they were made is not stated upstream) and one
  CC0 iNaturalist photograph (porcini, observation 187524159) so the `real` mode runs on the default path.
- **MODEL_CARD.md** is a first draft against MODEL_CARD_SPEC 1.2 and has not been reviewed.
- **Evaluation** has no ground-truth 3D: structure, `.ply` read-back and an input-view silhouette IoU through an
  approximate renderer. Idol's camera is read with a fixed calibration (field of view × 0.725, yaw − 5°), re-fitted
  on 2026-11-12 by leave-one-out on the three T4 splats (`tools/fit_camera_calibration.py`): held-out IoU 0.776,
  0.681, 0.868 (mean 0.775; uncalibrated mean 0.494). It replaced the 2026-11-10 CPU fit (× 0.6, + 25°), which
  scored 0.48–0.73 on the T4 splats. The best yaw differs by object, so one fixed yaw is a compromise.
- **Hosted runs (2026-11-12):** the default path passed on a fresh Colab T4 twice, driven by the Colab CLI with
  `HF_TOKEN` from the environment (`docs/release-verification.md`): at `71564c0` (12/12 cells, 1109 s) and at
  `1338a4a` with the re-fitted calibration (12/12 cells, 1075 s; IoU 0.7755 / 0.7586 / 0.9567). Neither is a
  browser `Run all`.

## Next steps

1. A browser `Run all` on Colab with the `HF_TOKEN` secret (the CLI runs cannot exercise the secret path or forms).
2. Optionally run Section 8 (`vae_check`, needs FLUX.2-dev access) and record the verdict in the ml-worker matrix row.
3. REL12 BYOD journey on a hosted runtime.

Design and the calls made while building: `docs/superpowers/specs/2026-11-10-idol-3d-pipeline-design.md`.
