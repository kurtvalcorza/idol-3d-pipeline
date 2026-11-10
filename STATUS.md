# Status — Candidate (not yet run on a hosted runtime)

**2026-11-10.** Built on branch `ccr-deb2a494-n7dsf1`. CI-level checks pass locally (ruff; 42 tests without torch;
notebook parity). The notebook's runtime-check and carrier cells and its carried `prepare` stage were executed locally
from the generated notebook (25 carried files verified; both samples fetched and digest-checked; three refusal probes
refused). **No model has run yet:** torch and the model stages have never executed in this repository.

## Blocking the first hosted run

1. **DINOv3 digests are not recorded.** `weights/dinov3-vith16plus/dimer-base-manifest.json` pins the commit
   (`c807c9e`) but not the file sizes and SHA-256, because the Hub hides them without access. Until they are recorded
   the notebook stops at Section 3 with that message. Either run, with an `HF_TOKEN` whose account has DINOv3 access:

   ```bash
   python tools/pin_snapshot.py dinov3-vith16plus && python tools/build_notebook.py && pytest
   ```

   or paste the output of this cell (run in any Colab with the `HF_TOKEN` secret) back to the maintainer session:

   ```python
   from google.colab import userdata
   from huggingface_hub import HfApi
   api = HfApi(token=userdata.get('HF_TOKEN'))
   for i in api.get_paths_info('facebook/dinov3-vith16plus-pretrain-lvd1689m', ['config.json', 'model.safetensors', 'LICENSE.md'],
                               revision='c807c9eeea853df70aec4069e6f56b28ddc82acc', expand=True):
       print(i.path, i.size, i.blob_id, i.lfs.sha256 if i.lfs else None)
   ```

   (Small files need their SHA-256 computed from the bytes; the pin tool does that and checks the git blob id.)

## Conformance boundary

- **RUN13 / MOD4 — deviation, by decision (Kurt, 2026-11-10):** the default path needs `HF_TOKEN` for the gated
  DINOv3 encoder. This caps the notebook at Candidate. Re-hosting DINOv3 (its licence permits redistribution with the
  licence text and "Built with DINOv3") would remove the deviation.
- **NOTEBOOK_SPEC 3.1** is declared; the guided layer is partial (no Predict → Change → Run activity yet, no
  separate use notice cell), and the write-up layer (DOC1–DOC21) has not been reviewed against this notebook.
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
