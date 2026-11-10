# Idol 3D pipeline — design (2026-11-10)

## Intent

Put Idol (`speridlabs/idol`, single image → 3D Gaussian splat, Apache-2.0) into the DIMER fleet as a
TASK-INFERENCE pipeline repository with a standalone Colab notebook, without depending on anything under a
non-commercial licence. Idol's published configuration loads two gated encoders: DINOv3 ViT-H+/16 (DINOv3
License) and the VAE of `black-forest-labs/FLUX.2-dev` (FLUX Non-Commercial License). The FLUX.2-dev VAE is the
licence blocker; DINOv3's licence permits use and redistribution, its gate is an access step.

## Decisions (with who made them)

- **VAE route (Kurt, 2026-11-10):** load the FLUX.2 VAE from `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0,
  ungated). Its `vae/config.json` names FLUX.2-dev as its origin; same `AutoencoderKLFlux2` architecture; half the
  bytes (bf16). Equivalence with dev is presumed until the optional `vae_check` stage passes (gate 1 of the
  ml-worker matrix row).
- **DINOv3 route (Kurt, 2026-11-10): option (b), token.** The notebook reads `HF_TOKEN` from the runtime's
  secrets and fetches DINOv3 from its gated origin. Consequence: the default path needs a credential, which
  NOTEBOOK_SPEC RUN13 / MOD4 forbid, so the carrier stays **Candidate** (FLUX.1-schnell precedent). Re-hosting
  DINOv3 (option a) remains the route to Release-grade.
- **Repository (Kurt):** `kurtvalcorza/idol-3d-pipeline`, separate from ml-worker.
- **Environment (Kurt):** the fleet's uv pattern (NOTEBOOK_SPEC §8.1, §25.14): pinned uv verified by size and
  SHA-256, `uv venv --managed-python` 3.12.12, a `uv pip compile --generate-hashes` lock carried byte-identical in
  the notebook, stages in child processes.

## Approaches considered

1. **Wrap the published SDK, vendored, with encoder overrides (chosen).** The SDK's `IdolPipeline` accepts
   `image_encoders={'dinov3': folder, 'flux2_vae': folder}`; local folders skip its gated-access check. The SDK is
   not on PyPI, so it is carried verbatim at commit `8db7003` and verified by SHA-256 (RAFT-Stereo precedent),
   rather than installed from git (ENV11 would need a source build).
2. Re-implement Idol's networks in this package — rejected: duplicates upstream code and drifts from it.
3. Install the SDK from a pinned git archive with `--no-deps` — rejected: needs a build backend outside the lock.

## Builder's calls (Claude, overnight 2026-11-10; for review)

- **Previews and the input-view check use a NumPy renderer** (`render.py`: EWA projection, 3σ / 24 px footprint
  cut, exact per-pixel depth sort, degree-0 colour). `gsplat` was not used: it has no lock-compatible binary wheel
  under `--only-binary :all:` from PyPI.
- **Camera convention is reported, not assumed.** The SDK does not document the direction or handedness of its
  predicted camera, so the `render` stage scores the silhouette IoU under four readings and uses the best.
- **Eager by default.** `optimize=False` (no `torch.compile`); the T4 path is the documented target and compile is
  the riskiest step there. An `OPTIMIZE` toggle turns it on.
- **Samples:** two images from the SDK's own `demo/examples/` (Apache-2.0 repository) with their upstream presets.
  How they were made is not stated upstream; recorded as such.
- **Checks without ground truth:** structure (finite, bounds, unit quaternions, opacity range), `.ply` read-back
  (count, standard fields, positions), silhouette IoU at the predicted camera. No score for the unseen sides.
- **Pins:** transformers 5.22.2 / diffusers 0.42.1 / huggingface-hub 2.4.0 (diffusers ≥ 0.40 is needed for
  `AutoencoderKLFlux2` and requires hub ≥ 1.0, which transformers 4.57.x forbids); torch 2.14.0 (the fleet's T4-proven
  pin).

## Non-goals

Fine-tuning (no upstream training recipe), FLUX.2-dev on the default path, mesh extraction, the CUDA 3DGS
rasterizer, Release-grade status while the token route stands.

## Open items (owner)

1. Pin DINOv3's per-file digests (needs an account with access): `tools/pin_snapshot.py dinov3-vith16plus`.
2. Hosted Colab T4 `Run all` and its evidence in `docs/release-verification.md`.
3. Run the `vae_check` branch once with FLUX.2-dev access; record the verdict in the ml-worker row.
4. DIMER acceptance of `.ply` output and a CUDA-only runtime (matrix gate 3).
