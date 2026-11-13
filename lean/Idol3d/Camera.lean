/-
Idol's predicted camera, as the SDK decodes it and the renderer places it.

* `sdkYaw`, `sdkPitch` — `src/idol/camera.py::decode_camera` (vendored SDK): `yaw = atan2(dx, dy)`,
  `pitch = asin(dz)` of the unit direction `d`
* `direction`           — the `d` that `src/idol_3d_pipeline/render.py::camera_pose` rebuilds from yaw and pitch
* `cameraAxes`          — the rows `right, up, forward` of `camera_pose`, for its four conventions
-/
import Idol3d.Basic

namespace Idol3d

open Real Matrix

/-- `math.atan2(dx, dy)` as the argument of `dy + dx i`. -/
noncomputable def sdkYaw (d : Fin 3 → ℝ) : ℝ := Complex.arg ⟨d 1, d 0⟩

/-- `math.asin(clip(dz, -1, 1))`; Mathlib's `arcsin` already clamps. -/
noncomputable def sdkPitch (d : Fin 3 → ℝ) : ℝ := arcsin (d 2)

/-- `camera_pose`'s `d = (sin yaw cos pitch, cos yaw cos pitch, sin pitch)`. -/
noncomputable def direction (yaw pitch : ℝ) : Fin 3 → ℝ :=
  ![sin yaw * cos pitch, cos yaw * cos pitch, sin pitch]

/-- **The renderer rebuilds exactly the direction the SDK decoded.** For the unit direction `d` that
`decode_camera` normalises, `direction (atan2 dx dy) (asin dz) = d`, including straight up or down. -/
theorem direction_decode {d : Fin 3 → ℝ} (hd : d ⬝ᵥ d = 1) : direction (sdkYaw d) (sdkPitch d) = d := by
  have hd' : d 0 * d 0 + d 1 * d 1 + d 2 * d 2 = 1 := by simpa [dotProduct, Fin.sum_univ_three] using hd
  have h2 : d 2 ∈ Set.Icc (-1 : ℝ) 1 := by
    constructor <;> nlinarith [mul_self_nonneg (d 0), mul_self_nonneg (d 1)]
  have hcos : cos (sdkPitch d) = √(d 1 ^ 2 + d 0 ^ 2) := by
    rw [sdkPitch, cos_arcsin]; congr 1; linear_combination -hd'
  set z : ℂ := ⟨d 1, d 0⟩
  have hz : ‖z‖ = √(d 1 ^ 2 + d 0 ^ 2) := by
    rw [Complex.norm_def, Complex.normSq_mk]; congr 1; ring
  ext i
  fin_cases i
  · show sin (Complex.arg z) * cos (sdkPitch d) = d 0
    rcases eq_or_ne z 0 with h | h
    · have : d 0 = 0 := by simpa [z] using congrArg Complex.im h
      simp [h, this]
    · rw [Complex.sin_arg, hcos, ← hz]
      field_simp [norm_ne_zero_iff.2 h]
      rfl
  · show cos (Complex.arg z) * cos (sdkPitch d) = d 1
    rcases eq_or_ne z 0 with h | h
    · have h0 : d 0 = 0 := by simpa [z] using congrArg Complex.im h
      have h1 : d 1 = 0 := by simpa [z] using congrArg Complex.re h
      simp [hcos, h0, h1]
    · rw [Complex.cos_arg h, hcos, ← hz]
      field_simp [norm_ne_zero_iff.2 h]
      rfl
  · show sin (sdkPitch d) = d 2
    exact sin_arcsin h2.1 h2.2

theorem direction_unit (yaw pitch : ℝ) : direction yaw pitch ⬝ᵥ direction yaw pitch = 1 := by
  simp only [direction, dotProduct, Fin.sum_univ_three]
  simp
  linear_combination (cos pitch) ^ 2 * sin_sq_add_cos_sq yaw + sin_sq_add_cos_sq pitch

/-! ### `camera_pose` -/

/-- The four conventions of `CONVENTIONS`: on which side of the object the camera sits, and whether
`right` is negated. -/
structure Convention where
  towardCamera : Bool
  mirrored : Bool

/-- `centre = radius * (d if toward_camera else -d)`. -/
noncomputable def centre (c : Convention) (yaw pitch radius : ℝ) : Fin 3 → ℝ :=
  (if c.towardCamera then radius else -radius) • direction yaw pitch

/-- `forward = -centre / |centre|`. -/
noncomputable def forward (c : Convention) (yaw pitch radius : ℝ) : Fin 3 → ℝ :=
  (1 / len (centre c yaw pitch radius)) • -centre c yaw pitch radius

/-- `right = cross(forward, z)` (or `x` near vertical), normalised, then negated if mirrored. -/
noncomputable def right (c : Convention) (yaw pitch radius : ℝ) : Fin 3 → ℝ :=
  let f := forward c yaw pitch radius
  let r := cross f ![0, 0, 1]
  let r := if len r < 1 / 1000000 then ![1, 0, 0] else r
  let r := (1 / len r) • r
  if c.mirrored then -r else r

/-- `pose["axes"] = stack([right, up, forward])` with `up = cross(right, forward)`. -/
noncomputable def cameraAxes (c : Convention) (yaw pitch radius : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let f := forward c yaw pitch radius
  let r := right c yaw pitch radius
  ofRows r (cross r f) f

theorem len_centre (c : Convention) (yaw pitch : ℝ) {radius : ℝ} (hr : 0 < radius) :
    len (centre c yaw pitch radius) = radius := by
  unfold len centre
  rw [smul_dotProduct, dotProduct_smul, direction_unit]
  split_ifs <;> simp <;> rw [← sq, sqrt_sq hr.le]

/-- **The camera sits at distance `radius` and looks at the origin**: `forward = −d` when the camera is on
the `d` side (`toward_camera`), `+d` otherwise. -/
theorem forward_eq (c : Convention) (yaw pitch : ℝ) {radius : ℝ} (hr : 0 < radius) :
    forward c yaw pitch radius = (if c.towardCamera then -1 else 1 : ℝ) • direction yaw pitch := by
  unfold forward
  rw [len_centre c yaw pitch hr]
  unfold centre
  ext i
  split_ifs <;> simp <;> field_simp

/-- Away from straight overhead, the axes in closed form. `s = ±1` is the side, `m = ±1` the mirroring. -/
theorem cameraAxes_eq (c : Convention) {yaw pitch radius : ℝ} (hr : 0 < radius)
    (hp : 1 / 1000000 < cos pitch) :
    let s : ℝ := if c.towardCamera then 1 else -1
    let m : ℝ := if c.mirrored then -1 else 1
    cameraAxes c yaw pitch radius =
      !![-(m * s) * cos yaw, m * s * sin yaw, 0;
         -(m * s * s) * (sin yaw * sin pitch), -(m * s * s) * (cos yaw * sin pitch), m * s * s * cos pitch;
         -s * (sin yaw * cos pitch), -s * (cos yaw * cos pitch), -s * sin pitch] := by
  intro s m
  have hc : 0 < cos pitch := lt_trans (by norm_num) hp
  have hs : s * s = 1 := by simp only [s]; split_ifs <;> norm_num
  have hm : m * m = 1 := by simp only [m]; split_ifs <;> norm_num
  have hf : forward c yaw pitch radius = (-s) • direction yaw pitch := by
    rw [forward_eq c yaw pitch hr]; simp only [s]; split_ifs <;> simp
  have hcr : cross (forward c yaw pitch radius) ![0, 0, 1] = ![-s * cos yaw * cos pitch, s * sin yaw * cos pitch, 0] := by
    rw [hf]; ext i; fin_cases i <;> simp [cross, direction] <;> ring
  have hlen : len ![-s * cos yaw * cos pitch, s * sin yaw * cos pitch, 0] = cos pitch := by
    rw [len, show (![-s * cos yaw * cos pitch, s * sin yaw * cos pitch, 0] : Fin 3 → ℝ) ⬝ᵥ
        ![-s * cos yaw * cos pitch, s * sin yaw * cos pitch, 0] = cos pitch ^ 2 by
      simp [dotProduct, Fin.sum_univ_three]
      linear_combination (cos pitch) ^ 2 * (s * s) * sin_sq_add_cos_sq yaw + (cos pitch) ^ 2 * hs]
    exact sqrt_sq hc.le
  have hr' : right c yaw pitch radius = ![-(m * s) * cos yaw, m * s * sin yaw, 0] := by
    simp only [right, hcr, hlen, show ¬ cos pitch < 1 / 1000000 from not_lt.2 hp.le, ite_false]
    ext i
    simp only [m]
    split_ifs <;> fin_cases i <;> simp <;> field_simp
  simp only [cameraAxes, hr', hf]
  ext i j
  fin_cases i <;> fin_cases j <;> simp [ofRows, cross, direction] <;>
    first | ring1 | linear_combination (m * s * s * cos pitch) * sin_sq_add_cos_sq yaw

section closed
variable (c : Convention) {yaw pitch radius : ℝ} (hr : 0 < radius) (hp : 1 / 1000000 < cos pitch)
include hr hp

/-- **The axes are orthonormal**, in every convention. -/
theorem cameraAxes_orthogonal :
    cameraAxes c yaw pitch radius * (cameraAxes c yaw pitch radius)ᵀ = 1 := by
  rw [cameraAxes_eq c hr hp]
  have hy := sin_sq_add_cos_sq yaw
  have hq := sin_sq_add_cos_sq pitch
  have hs : (if c.towardCamera then (1 : ℝ) else -1) ^ 2 = 1 := by split_ifs <;> norm_num
  have hm : (if c.mirrored then (-1 : ℝ) else 1) ^ 2 = 1 := by split_ifs <;> norm_num
  generalize (if c.towardCamera then (1 : ℝ) else -1) = s at hs
  generalize (if c.mirrored then (-1 : ℝ) else 1) = m at hm
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_three] <;>
    first
      | ring1
      | linear_combination (m ^ 2 * s ^ 2) * hy + m ^ 2 * hs + hm
      | linear_combination (m ^ 2 * s ^ 4 * sin pitch ^ 2) * hy + m ^ 2 * s ^ 4 * hq + m ^ 2 * (s ^ 2 + 1) * hs + hm
      | linear_combination (s ^ 2 * cos pitch ^ 2) * hy + s ^ 2 * hq + hs
      | linear_combination (m * s ^ 3 * sin pitch * cos pitch) * hy

/-- **Every convention is left-handed** (`det = −1`): x right, y up, z forward, as a non-mirrored
picture needs; with the renderer's `v = size/2 − f·y/z` the image axes (u right, v down, depth) are
right-handed. So none of the four readings is a mirror image. -/
theorem det_cameraAxes : (cameraAxes c yaw pitch radius).det = -1 := by
  rw [cameraAxes_eq c hr hp]
  have hy := sin_sq_add_cos_sq yaw
  have hq := sin_sq_add_cos_sq pitch
  have hs : (if c.towardCamera then (1 : ℝ) else -1) ^ 2 = 1 := by split_ifs <;> norm_num
  have hm : (if c.mirrored then (-1 : ℝ) else 1) ^ 2 = 1 := by split_ifs <;> norm_num
  generalize (if c.towardCamera then (1 : ℝ) else -1) = s at hs
  generalize (if c.mirrored then (-1 : ℝ) else 1) = m at hm
  simp [Matrix.det_fin_three]
  have hs4 : s ^ 4 = 1 := by nlinarith [hs]
  linear_combination (-(m ^ 2 * s ^ 4)) * ((cos pitch) ^ 2 + (sin pitch) ^ 2) * hy
    - (m ^ 2 * s ^ 4) * hq - s ^ 4 * hm - hs4 + 0 * hs

/-- **"Mirrored" is upside down, not mirrored**: the `*_mirrored` axes are the plain ones turned by 180°
about the view axis (`diag(−1, −1, 1)`), so the rendered image is rotated by 180°. Its "up" points down. -/
theorem cameraAxes_mirrored (t : Bool) :
    cameraAxes ⟨t, true⟩ yaw pitch radius =
      Matrix.diagonal ![-1, -1, 1] * cameraAxes ⟨t, false⟩ yaw pitch radius := by
  cases t <;> rw [cameraAxes_eq _ hr hp, cameraAxes_eq _ hr hp] <;> ext i j <;>
    fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Matrix.diagonal]

/-- The plain conventions are upright: "up" has positive world z (`cos pitch > 0`). -/
theorem up_z_plain (t : Bool) : cameraAxes ⟨t, false⟩ yaw pitch radius 1 2 = cos pitch := by
  rw [cameraAxes_eq _ hr hp]; cases t <;> simp

end closed

/-! ### `orbit` and the field of view -/

/-- **`orbit(camera, θ)` moves the camera clockwise by `θ` seen from above**: its centre is the old one turned
by `−θ` about z. -/
theorem centre_orbit (c : Convention) (yaw θ pitch radius : ℝ) :
    centre c (yaw + θ) pitch radius = rotZ (-θ) *ᵥ centre c yaw pitch radius := by
  unfold centre
  ext i
  fin_cases i <;> split_ifs <;>
    simp [direction, rotZ, Matrix.mulVec, dotProduct, Fin.sum_univ_three, sin_add, cos_add] <;> ring

/-- **What the SDK's field of view means.** `decode_camera` sets `fov = 2·atan(w·k)` from the half-width `w`
and inverse distance `k = 1/radius`; the renderer's focal length `(size/2)/tan(fov/2)` then sends a point at
the object's distance (depth `1/k`), `w` to the side, exactly to the image edge `u = size`. -/
theorem halfWidth_to_edge {w k size : ℝ} (hw : 0 < w) (hk : 0 < k) (hs : 0 < size) :
    let fov := 2 * arctan (w * k)
    let focal := size / 2 / tan (fov / 2)
    size / 2 + focal * w / (1 / k) = size := by
  intro fov focal
  have : tan (fov / 2) = w * k := by
    simp only [fov]; rw [show 2 * arctan (w * k) / 2 = arctan (w * k) by ring, tan_arctan]
  simp only [focal, this]
  field_simp
  ring

end Idol3d
