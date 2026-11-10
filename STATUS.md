# Status — Candidate (not yet run on a hosted runtime)

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

## Blocking the first hosted run

1. **DINOv3 digests are not recorded.** `weights/dinov3-vith16plus/dimer-base-manifest.json` pins the commit
   (`c807c9e`) but not the file sizes and SHA-256, because the Hub hides them without access. Until they are recorded
   the notebook stops at Section 3, before any download.

   A Hugging Face token for account `kurtvalcorza` is saved on the cloud environment (huggingface.co network secret),
   and FLUX.2-dev access works. The DINOv3 access request is **awaiting review by Meta** (Hub message, 2026-11-10);
   approval is manual.

   Once access works, the maintainer session runs (no download of the weights; the Hub's LFS record gives their size
   and SHA-256):

   ```bash
   HF_TOKEN=proxy-injected python tools/pin_snapshot.py dinov3-vith16plus && python tools/build_notebook.py && pytest
   ```

   The same token in Colab (`HF_TOKEN` secret) is what the notebook's `weights` stage uses.

## Conformance boundary

- **RUN13 / MOD4 — deviation, by decision (Kurt, 2026-11-10):** the default path needs `HF_TOKEN` for the gated
  DINOv3 encoder. The same deviation fails RUN14, §3.1, REL6 and §31(4). This caps the notebook at Candidate. Re-hosting DINOv3 (its licence permits redistribution with the
  licence text and "Built with DINOv3") would remove the deviation.
- **NOTEBOOK_SPEC 3.1** is declared. Open gaps found by review and not yet closed: no Predict → Change → Run
  activity (GDL10), no separate use-notice cell, no "What did not work" section, and ENV14 (instantiating the paths
  from the lock alone) is not done; the Model Card's short sections need expanding (MODEL_CARD_SPEC G13).
- **Samples:** two Idol SDK demo images (Apache-2.0 repository; how they were made is not stated upstream) and one
  CC0 iNaturalist photograph (porcini, observation 187524159) so the `real` mode runs on the default path.
- **MODEL_CARD.md** is a first draft against MODEL_CARD_SPEC 1.2 and has not been reviewed.
- **Evaluation** has no ground-truth 3D: structure, `.ply` read-back and an input-view silhouette IoU through an
  approximate renderer. The camera convention is reported under four readings, not assumed.

## Next steps

1. Record the DINOv3 digests (above).
2. Colab T4 `Run all`; record the executed copy under `docs/execution-evidence/` and the run in
   `docs/release-verification.md`.
3. Optionally run Section 8 (`vae_check`, needs FLUX.2-dev access) and record the verdict in the ml-worker matrix row.
4. REL12 BYOD journey on a hosted runtime.

Design and the calls made while building: `docs/superpowers/specs/2026-11-10-idol-3d-pipeline-design.md`.
