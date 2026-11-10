# Release verification

No clean-runtime execution of `tutorials/idol_3d_colab.ipynb` has been recorded yet.

| Date | Runtime | Notebook blob | Source revision | Scope | Result | Evidence |
|---|---|---|---|---|---|---|
| 2026-11-10 | cloud session, NumPy only (no torch) | — | `9b53f47` | **Gate 1, weights: klein VAE vs FLUX.2-dev VAE.** `black-forest-labs/FLUX.2-dev@26afe3a` `vae/diffusion_pytorch_model.safetensors` (336,213,556 bytes, SHA-256 `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5`, F32 + one I64) against `FLUX.2-klein-4B@e7b7dc2` (168,120,878 bytes, SHA-256 `ca70d220…0f04`, BF16 + one I64): same 251 tensor names and shapes; every F32 tensor rounded to bfloat16 (round to nearest, ties to even, as `torch.Tensor.to(torch.bfloat16)`) is bit-for-bit equal to klein's, and the I64 counter is equal; `vae/config.json` differs only in `_diffusers_version` and klein's `_name_or_path`. Idol loads the VAE with `dtype=torch.bfloat16`, so both load to identical parameters. | **PASS** (251/251 bitwise equal, max abs diff 0) | session log; the notebook's `vae_check` stage repeats it with torch and adds Idol's own `feature2` |
| 2026-11-10 | local CPython 3.13, no GPU | generated | uncommitted | Sections 1–2 (runtime check, carrier) executed from the notebook; carried `prepare` stage run with the repository's NumPy/Pillow | pass (not REL1: no isolated environment, no model) | session log only |

A hosted Colab T4 `Run all` is required before any Release-grade claim; the RUN13/MOD4 deviation (see `STATUS.md`)
caps the notebook at Candidate regardless.
