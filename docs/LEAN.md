# Lean formalisation of the renderer's maths

`lean/` is a [Lean 4](https://lean-lang.org/) project, built on [Mathlib](https://leanprover-community.github.io/mathlib4_docs/),
that proves the mathematical properties the NumPy renderer and the checks rely on: how Idol's camera is decoded
and placed, that each Gaussian's footprint covers its 3σ ellipse, that front-to-back compositing is exact, and what
the `.ply` axis change does. Lean's kernel checks every proof; nothing is assumed beyond Lean's three standard
axioms (`propext`, `Classical.choice`, `Quot.sound`). The same approach is used in
[`insect-3d-reconstruction`](https://github.com/kurtvalcorza/insect-3d-reconstruction/blob/main/docs/LEAN.md).

| Layer | Where | What it shows |
|---|---|---|
| Proofs over ℝ | `lean/Idol3d/*.lean` | the mathematics is right, for all inputs |
| Property tests | `tests/test_properties.py` | the Python, in floating point, has each proved property (Hypothesis, 150 random cases per Hypothesis test) |
| Executable reference | `lean/Idol3d/Reference/Float.lean`, `tests/test_lean_reference.py` | a second, independent `Float` implementation gives the Python's numbers |

Nothing here checks Idol, the encoders, the VAE or any measured number.

## Building and running

```bash
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o elan-init.sh
sh elan-init.sh -y --default-toolchain none     # installs elan (the Lean version manager) into ~/.elan
cd lean
lake exe cache get          # clones Mathlib at the commit in lake-manifest.json, downloads its prebuilt cache (~7 GB)
lake build                  # checks every proof
lake build idol3d-ref       # the executable reference used by tests/test_lean_reference.py
cd ..
python -m pip install hypothesis==6.170.15
pytest tests/test_properties.py tests/test_lean_reference.py
```

Use `lake exe cache get`, not `lake update`, which moves Mathlib to its latest commit and rewrites the manifest.
`tests/test_lean_reference.py` skips when the executable is not built; `tests/test_properties.py` skips without
Hypothesis. CI's `lean` job builds the proofs with Mathlib's cache, runs `axiom-audit` (which fails on `sorry`,
`admit`, `native_decide` or any non-standard axiom), builds `idol3d-ref` and runs the cross-check.

## What is proved

| Python | Lean | Theorems | Property test |
|---|---|---|---|
| `decode_camera` (SDK) then `camera_pose`'s `d` | `sdkYaw`, `sdkPitch`, `direction` | `direction_decode`, `direction_unit` | `test_renderer_rebuilds_the_sdk_direction` |
| `camera_pose`, all four `CONVENTIONS` | `centre`, `forward`, `right`, `cameraAxes` | `len_centre`, `forward_eq`, `cameraAxes_eq`, `cameraAxes_orthogonal`, `det_cameraAxes`, `up_z_plain` | `test_camera_axes` |
| the `*_mirrored` conventions | `cameraAxes` | `cameraAxes_mirrored` | `test_mirrored_is_upside_down` |
| `orbit` | `centre` | `centre_orbit` | `test_orbit_turns_clockwise` |
| the field of view (`decode_camera`, `render`'s focal length) | — | `halfWidth_to_edge` | `test_half_width_reaches_the_image_edge` |
| `quaternion_to_matrix` | `quatMatrix`, `normalise` | `quaternion_to_matrix_rotation`, `quatMatrix_eq_H` | `test_quaternion_to_matrix_is_a_rotation` |
| the EWA footprint in `render` | `lamMax` | `cov2_gram`, `gram2_psd`, `det_lowpass_ge`, `valid_always`, `inverse_correct`, `lamMax_ge`, `lamMax_char`, `three_sigma_cover`, `pixel_enumerated` | `test_footprint_covers_the_three_sigma_ellipse` |
| front-to-back compositing in `render` | `transmittance` | `sum_weights`, `exp_sum_log`, `transmittance_mem`, `accumulated_mem`, `image_mem` | `test_front_to_back_compositing` |
| `save_ply`'s axis change (SDK), `ply_roundtrip` | `plyAxes` | `plyAxes_rotation` | `test_ply_axes_are_a_rotation_that_puts_up_along_minus_y` |
| `silhouette_iou` | `silhouetteIoU` | `silhouetteIoU_mem`, `silhouetteIoU_comm`, `silhouetteIoU_empty` | `test_silhouette_iou` |

Files: `Idol3d/Basic.lean`, `Camera.lean`, `Quaternion.lean`, `Ewa.lean`, `Composite.lean`, `Metrics.lean`. 34 theorems.
Theorems not in the table (`quatMatrixH_mul_transpose`, `det_quatMatrixH`, `dot_self_nonneg`) are proof steps.

## What the formalisation found

1. **The `*_mirrored` conventions are upside down, not mirrored.** `camera_pose` negates `right` and then computes
   `up = cross(right, forward)`, so `up` flips too. The result is the plain camera turned 180° about its view axis
   (`cameraAxes_mirrored`); every convention has the same handedness (`det_cameraAxes`). So the 2026-11-10 search over
   "mirroring" (tutorial, "What did not work") searched upside-down readings, not mirror images; a mirror image was
   never one of the candidates. The fitted calibration uses the plain `toward_camera` reading, so it is unaffected.
2. **The `.ply` puts the object's up along −y.** `save_ply` applies `(x, y, z) ↦ (x, −z, y)`, the +90° rotation about
   x that its comment names. That is a proper rotation (not a mirror image), but it sends +z to −y
   (`plyAxes_rotation`), while the comment, and `ply_roundtrip`'s docstring, call the target "the y-up convention
   viewers use". A y-up viewer shows the object upside down; a y-down one (the OpenCV convention of many 3DGS tools)
   shows it upright. The map to y-up would be `(x, z, −y)`. The code is the vendored SDK, carried verbatim and pinned
   by hash, so this is reported rather than changed.
3. **`valid = det > 1e-12` never fails in exact arithmetic.** The projected covariance is a Gram matrix, so with the
   0.3 px² low-pass its determinant is at least 0.09 (`valid_always`). The check only matters for non-finite input.
4. **The 3σ cut is exact as described.** Every pixel whose centre is within 3σ is enumerated (`three_sigma_cover`,
   `pixel_enumerated`), as long as the radius is not cut at `MAX_RADIUS`; the property test confirms those pixels get
   exactly the Gaussian's alpha.
5. **The two clips in `render` are no-ops in exact arithmetic.** The weights telescope to `1 − Π(1 − α)`, so the
   accumulated opacity and the image are already in [0, 1] (`accumulated_mem`, `image_mem`), and the log form of
   the transmittance is exact while `α < 1` (`exp_sum_log`; `ALPHA_MAX = 0.99`).
6. **The SDK's field of view is the object's half-width.** With `fov = 2 atan(w / radius)`, a point `w` to the
   side at the object's distance lands exactly on the image edge (`halfWidth_to_edge`); the calibration's × 0.725
   scales that.
7. **`silhouette_iou` returns NaN for two empty silhouettes**, a third convention beside insect-3d-reconstruction's
   `None` and `0`.

## What is not covered

- Idol itself (the denoiser, decoders, encoders, VAE) and the leave-one-out calibration fit
  (`tools/fit_camera_calibration.py`), which is a search, not a formula.
- The EWA Jacobian as an approximation of the true projection: the proofs are about the formulas the renderer uses.
- `splat_summary`'s bounds and quantiles, the mask resize in `silhouette_iou`, and the snapshot and notebook tooling.
- Floating point: the theorems are about real numbers; the tests connect them to the code.

## Keeping it in step with the code

When a formalised function changes, change its Lean definition (each file's header names the Python), run
`lake build`, update `Reference/Float.lean` and rebuild `idol3d-ref`, then run both test files. To move to a newer
Mathlib on purpose, set its commit in `lakefile.toml`, copy its `lean-toolchain`, then `lake update`,
`lake exe cache get` and `lake build`.
