/-
The EWA footprint in `src/idol_3d_pipeline/render.py::render`.

For each Gaussian: `cov_world = (R S)(R S)ᵀ`, `cov_cam = A cov_world Aᵀ` (A = camera axes),
`cov2 = J cov_cam Jᵀ` (J = 2×3 Jacobian), then `a, b, c = cov2[0,0] + 0.3, cov2[0,1], cov2[1,1] + 0.3`,
`det = a c − b²`, `valid = det > 1e-12`, the inverse `(c, −b, a) / det`, the largest eigenvalue
`lam = (a+c)/2 + √(((a+c)/2)² − det)`, and the footprint radius `ceil(3 √lam)`.
-/
import Idol3d.Basic

namespace Idol3d

open Real Matrix

/-- `cov2` is a Gram matrix: `J (A (M Mᵀ) Aᵀ) Jᵀ = N Nᵀ` with `N = J A M`, whatever `J`, `A`, `M`. -/
theorem cov2_gram (J : Matrix (Fin 2) (Fin 3) ℝ) (A M : Matrix (Fin 3) (Fin 3) ℝ) :
    J * (A * (M * Mᵀ) * Aᵀ) * Jᵀ = (J * A * M) * (J * A * M)ᵀ := by
  simp only [Matrix.transpose_mul, Matrix.mul_assoc]

/-- The entries of a 2×2 Gram matrix: `a ≥ 0`, `c ≥ 0`, `a c ≥ b²` (Cauchy–Schwarz). -/
theorem gram2_psd (N : Matrix (Fin 2) (Fin 3) ℝ) :
    let G := N * Nᵀ
    0 ≤ G 0 0 ∧ 0 ≤ G 1 1 ∧ G 0 1 ^ 2 ≤ G 0 0 * G 1 1 := by
  intro G
  simp only [G, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_three]
  refine ⟨by nlinarith [mul_self_nonneg (N 0 0), mul_self_nonneg (N 0 1), mul_self_nonneg (N 0 2)],
    by nlinarith [mul_self_nonneg (N 1 0), mul_self_nonneg (N 1 1), mul_self_nonneg (N 1 2)], ?_⟩
  nlinarith [sq_nonneg (N 0 0 * N 1 1 - N 0 1 * N 1 0), sq_nonneg (N 0 0 * N 1 2 - N 0 2 * N 1 0),
    sq_nonneg (N 0 1 * N 1 2 - N 0 2 * N 1 1)]

/-- With the low-pass `λ` added to the diagonal, the determinant is at least `λ²`. -/
theorem det_lowpass_ge {a b c lam : ℝ} (ha : 0 ≤ a) (hc : 0 ≤ c) (hb : b ^ 2 ≤ a * c) (hl : 0 ≤ lam) :
    lam ^ 2 ≤ (a + lam) * (c + lam) - b ^ 2 := by
  nlinarith [mul_nonneg ha hl, mul_nonneg hc hl]

/-- **`valid` is always true in exact arithmetic**: with `LOW_PASS = 0.3`, every footprint has
`det ≥ 0.09 > 1e-12`, so the check only matters for non-finite input. -/
theorem valid_always (N : Matrix (Fin 2) (Fin 3) ℝ) :
    let G := N * Nᵀ
    (1 / 10 ^ 12 : ℝ) < (G 0 0 + 3 / 10) * (G 1 1 + 3 / 10) - G 0 1 * G 0 1 := by
  intro G
  obtain ⟨ha, hc, hb⟩ := gram2_psd N
  have := det_lowpass_ge ha hc hb (show (0 : ℝ) ≤ 3 / 10 by norm_num)
  nlinarith

section footprint
variable {a b c : ℝ} (ha : 0 < a) (hc : 0 < c) (hd : 0 < a * c - b ^ 2)

include hd in
/-- `(inv_a, inv_b, inv_c) = (c, −b, a)/det` is the inverse of `[[a, b], [b, c]]`. -/
theorem inverse_correct :
    !![c / (a * c - b ^ 2), -b / (a * c - b ^ 2); -b / (a * c - b ^ 2), a / (a * c - b ^ 2)] *
      !![a, b; b, c] = 1 := by
  have h1 : a * c - b ^ 2 ≠ 0 := hd.ne'
  have h2 : c * a - b ^ 2 ≠ 0 := by rw [mul_comm]; exact h1
  ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply] <;> field_simp <;> ring

/-- `lam`, the larger eigenvalue. -/
noncomputable def lamMax (a b c : ℝ) : ℝ := (a + c) / 2 + √(((a + c) / 2) ^ 2 - (a * c - b ^ 2))

include ha hc in
theorem lamMax_ge : a ≤ lamMax a b c ∧ c ≤ lamMax a b c ∧ 0 < lamMax a b c := by
  have hdisc : ((a + c) / 2) ^ 2 - (a * c - b ^ 2) = ((a - c) / 2) ^ 2 + b ^ 2 := by ring
  have hs : |a - c| / 2 ≤ √(((a + c) / 2) ^ 2 - (a * c - b ^ 2)) := by
    rw [hdisc]
    apply le_sqrt_of_sq_le
    rw [div_pow, sq_abs]; nlinarith [sq_nonneg b]
  unfold lamMax
  refine ⟨?_, ?_, by positivity⟩ <;> cases abs_cases (a - c) <;> linarith

/-- `lam` solves the characteristic equation `λ² − (a+c) λ + det = 0`. -/
theorem lamMax_char : lamMax a b c ^ 2 - (a + c) * lamMax a b c + (a * c - b ^ 2) = 0 := by
  have hdisc : 0 ≤ ((a + c) / 2) ^ 2 - (a * c - b ^ 2) := by nlinarith [sq_nonneg (a - c), sq_nonneg b]
  unfold lamMax
  have := sq_sqrt hdisc
  linear_combination this

include ha hc hd in
/-- **The 3σ footprint covers the 3σ ellipse.** If a pixel centre's offset `(dx, dy)` has Mahalanobis
distance at most 3 (`power ≥ −4.5`), then `|dx|, |dy| ≤ 3 √lam`, inside the radius `ceil(3 √lam)`. -/
theorem three_sigma_cover {dx dy : ℝ}
    (hq : (c * dx ^ 2 - 2 * b * dx * dy + a * dy ^ 2) / (a * c - b ^ 2) ≤ 9) :
    |dx| ≤ 3 * √(lamMax a b c) ∧ |dy| ≤ 3 * √(lamMax a b c) := by
  obtain ⟨hla, hlc, hl⟩ := lamMax_ge ha hc
  have hchar := lamMax_char (a := a) (b := b) (c := c)
  set l := lamMax a b c
  set D := a * c - b ^ 2
  -- l Σ⁻¹ − I is positive semi-definite: diagonals ≥ 0 and determinant 0
  have hp : 0 ≤ l * c - D := by nlinarith [sq_nonneg b]
  have hr : 0 ≤ l * a - D := by nlinarith [sq_nonneg b]
  have hdet0 : (l * c - D) * (l * a - D) - (l * b) ^ 2 = 0 := by
    have : (l * c - D) * (l * a - D) - (l * b) ^ 2 = D * (l ^ 2 - (a + c) * l + D) := by
      simp only [D]; ring
    rw [this, hchar, mul_zero]
  have hform : 0 ≤ (l * c - D) * dx ^ 2 + 2 * (-(l * b)) * dx * dy + (l * a - D) * dy ^ 2 := by
    rcases eq_or_lt_of_le hp with h0 | h0
    · have hb0 : l * b = 0 := by nlinarith [sq_nonneg (l * b)]
      rw [← h0, hb0]; nlinarith [sq_nonneg dy]
    · nlinarith [sq_nonneg ((l * c - D) * dx - l * b * dy), mul_pos h0 h0]
  have hsum : dx ^ 2 + dy ^ 2 ≤ 9 * l := by
    have hq' : c * dx ^ 2 - 2 * b * dx * dy + a * dy ^ 2 ≤ 9 * D := by
      rwa [div_le_iff₀ hd] at hq
    nlinarith
  have hsq : (3 * √l) ^ 2 = 9 * l := by rw [mul_pow, sq_sqrt hl.le]; norm_num
  constructor <;> apply abs_le_of_sq_le_sq' _ (by positivity) |>.elim (fun h1 h2 => abs_le.2 ⟨h1, h2⟩) <;>
    nlinarith [sq_nonneg dx, sq_nonneg dy]

end footprint

/-- **Every pixel of the ellipse is enumerated.** The renderer visits pixels `px = ⌊u⌋ + o` for offsets
`o ∈ [−r, r]` and samples at the pixel centre, `dx = px + 0.5 − u`. A pixel whose centre is within `r` of `u`
is among them: `|dx| ≤ r → |px − ⌊u⌋| ≤ r`. -/
theorem pixel_enumerated {u : ℝ} {px r : ℤ} (h : |(px : ℝ) + 1 / 2 - u| ≤ r) : |px - ⌊u⌋| ≤ r := by
  have hf := Int.floor_le u
  have hf' := Int.lt_floor_add_one u
  rw [abs_le] at h ⊢
  constructor
  · have : ((-r : ℤ) : ℝ) - 1 < px - ⌊u⌋ := by push_cast; linarith [h.1]
    have := Int.cast_lt.1 (by push_cast at this ⊢; linarith : ((-r - 1 : ℤ) : ℝ) < ((px - ⌊u⌋ : ℤ) : ℝ))
    omega
  · have := Int.cast_lt.1 (by push_cast; linarith [h.2] : ((px - ⌊u⌋ : ℤ) : ℝ) < ((r + 1 : ℤ) : ℝ))
    omega

end Idol3d
