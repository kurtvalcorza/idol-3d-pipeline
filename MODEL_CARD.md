---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: image-to-3d
task: "Others - Image-to-3D (Gaussian Splatting)"
base_model: speridlabs/idol
date_published: "2026-10-28"
date_published_source: "Creation time of the Hugging Face repository speridlabs/idol (Hub API createdAt 2026-10-28T15:55:17Z, read 2026-11-10); the pinned revision 2731c22 was last modified 2026-10-29."
---

# Idol — Single Image to 3D Gaussian Splat (Task Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-speridlabs%2Fidol-ffcc4d?style=flat)](https://huggingface.co/speridlabs/idol)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-speridlabs%2Fidol--sdk-181717?style=flat&logo=github&logoColor=white)](https://github.com/speridlabs/idol-sdk)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are fetched unmodified from their upstream locations under their own licences, which control your use; the accompanying code and notebooks are released under this repository's licence. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production or safety-critical use. Running the notebook downloads third-party weights governed by their own licences and consumes compute on your own Colab account. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> **Draft card, Candidate pipeline.** No result in this card was produced by running the model in this repository yet. The DINOv3 snapshot is pinned by commit only; its per-file digests are pending (see `STATUS.md`).

---

## Interactive Colab Tutorials

- **Task-inference Guided Notebook**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/idol-3d-pipeline/blob/main/tutorials/idol_3d_colab.ipynb) [`idol_3d_colab.ipynb`](tutorials/idol_3d_colab.ipynb)
  *Three pinned sample images (two Idol SDK demo images, one CC0 photograph) turned into 3D Gaussian splats on a T4, with structural, `.ply` read-back and input-view silhouette checks, rendered previews, an optional klein-vs-dev VAE equivalence check, and Bring Your Own Data. Needs an `HF_TOKEN` secret for the gated DINOv3 encoder.*

#### Description

Idol generates up to 1,048,576 3D Gaussians of one object from one image, including the sides the image does not show. BiRefNet removes the background; DINOv3 ViT-H+/16 tokens and FLUX.2 VAE latents condition a 370.6 M-parameter flow-matching diffusion transformer that produces an 8,192 × 16 object latent and a predicted camera in 50 guided Euler steps; a 50.8 M-parameter octree decoder places anchors on a 256³ grid and a 237.3 M-parameter Gaussian decoder turns each anchor into 32 Gaussians (parameter counts from the upstream card). **Packaging change:** Idol's configuration names `black-forest-labs/FLUX.2-dev` for the VAE (FLUX Non-Commercial License). This pipeline replaces it with the VAE of `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0), which declares FLUX.2-dev as its origin and has the same architecture in bf16; Idol runs the VAE in bf16 either way. Their equality is presumed, not yet verified (the notebook's optional Section 8 tests it).

#### Intended Use and Limitations

###### Primary Intended Uses

- Teaching and evaluating single-image 3D generation on a hosted GPU runtime.
- Producing 3DGS `.ply` files of single objects from renders, generated images or photographs for inspection in a splat viewer.

###### Primary Intended Users

- Learners and researchers using the DIMER platform's tutorials.

###### Out-of-scope use cases

- Scenes, multiple objects, or measurement-grade geometry; the unseen sides are a prediction.
- Fine-tuning (no upstream training recipe is published).
- Any use requiring FLUX.2-dev; this pipeline does not load it on the default path.

#### Factors

###### Groups

Not assessed. The samples are two object images from the upstream SDK and one CC0 photograph of a mushroom.

###### Instrumentation

Input framing depends on BiRefNet's mask and the `synthetic` (1.2×) or `real` (1.7×) crop margin.

###### Environment

A CUDA GPU is required (documented target: Colab T4). The upstream card reports 7.2 GB peak VRAM.

#### Metrics

###### Performance Measures

No ground-truth 3D is used here. The notebook reports per sample: structural validity (finite values, Gaussian centres inside [-0.55, 0.55]³, unit quaternions, opacity in [0, 1]); `.ply` read-back (vertex count, standard 3DGS fields, positions); and the silhouette IoU between the splat rendered from Idol's predicted camera (approximate NumPy renderer) and the object mask of the input. The upstream card's benchmark (PSNR, SSIM, LPIPS, IoU at the input view; DINOv2 and CLIP at seven new views, versus TripoSplat, TRELLIS and LGM on H100) is the authors' and is **not re-run here**.

###### Decision thresholds

None. The silhouette IoU is reported, not thresholded.

###### Approaches to uncertainty and variability

One seed per sample (the upstream preset). Variation across seeds is not measured.

#### Ethical considerations and biases

###### Data

The training data of Idol is not described in the upstream card.

###### Human Life

Not intended for decisions affecting people.

###### Mitigations

Digest-verified weights; no Hub-hosted code; BYOD files stay in the runtime.

###### Risks and harms

Generated geometry may be mistaken for measured geometry. Images of identifiable people should not be uploaded to a hosted runtime without authorisation.

###### Use cases

See Primary Intended Uses.

## Immutable provenance

| Component | Source | Revision | Licence | Files |
|---|---|---|---|---|
| Idol | `speridlabs/idol` | `2731c2291fc65a797c5cb627a5e5526fc1490902` | Apache-2.0 | 4 safetensors (3,078,979,896 bytes) + `config.json` |
| FLUX.2 VAE | `black-forest-labs/FLUX.2-klein-4B` (`vae/` only) | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | Apache-2.0 | `vae/diffusion_pytorch_model.safetensors` (168,120,878 bytes, SHA-256 `ca70d220…0f04`), `vae/config.json` |
| DINOv3 ViT-H+/16 | `facebook/dinov3-vith16plus-pretrain-lvd1689m` (gated) | `c807c9eeea853df70aec4069e6f56b28ddc82acc` | DINOv3 License | digests pending |
| Idol SDK | `speridlabs/idol-sdk` (carried in `src/idol`) | `8db700349a5570aac390e4780fa32e3eed6b587a` | Apache-2.0 | 13 files, SHA-256 in `docs/upstream-sdk.json` |

Built with DINOv3.

## Input/output contract

Input: one PIL image of one object, 64..4096 px per side, mode `synthetic` or `real`, 1..100 steps, 32,768..1,048,576 Gaussians (multiple of 32). Output: a 3DGS `.ply` (z-up object turned y-up), the 1024 × 1024 model input and its mask, the predicted camera (yaw, pitch in radians; radius; field of view in degrees).

## Runtime and verification records

See `docs/release-verification.md`. No hosted run recorded yet.

## References

[1] Sperid Labs. 2026. Idol. https://huggingface.co/speridlabs/idol

[2] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4, Article 139. https://doi.org/10.1145/3592433

[3] Oriane Siméoni et al. 2025. DINOv3. arXiv:2508.10104. https://arxiv.org/abs/2508.10104
