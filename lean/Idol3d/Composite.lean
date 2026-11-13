/-
Front-to-back compositing in `src/idol_3d_pipeline/render.py::render`.

Per pixel the contributions are sorted nearest first with opacities `α₀, α₁, …` (each in `[1/255, 0.99]`).
The renderer computes the transmittance through the log, `exp(Σ_{j<k} log1p(−α_j))`, the weights
`w_k = α_k T_k`, `accumulated = Σ w_k` (clipped to [0, 1]), `colour = Σ w_k rgb_k` and
`image = colour + (1 − accumulated) · background` (clipped to [0, 1]).
-/
import Mathlib

namespace Idol3d

open Real Finset

/-- Light that reaches the `k`-th contribution: `Π_{j<k} (1 − α_j)`. -/
def transmittance (α : ℕ → ℝ) (k : ℕ) : ℝ := ∏ j ∈ range k, (1 - α j)

/-- **The log form is exact**: `exp(Σ log(1 − α_j)) = Π (1 − α_j)` whenever every `α_j < 1`
(`ALPHA_MAX = 0.99` guarantees it). -/
theorem exp_sum_log (α : ℕ → ℝ) (k : ℕ) (h : ∀ j, α j < 1) :
    exp (∑ j ∈ range k, log (1 - α j)) = transmittance α k := by
  rw [exp_sum]
  exact Finset.prod_congr rfl fun j _ => exp_log (by linarith [h j])

/-- **The weights telescope**: `Σ_{k<m} α_k T_k = 1 − T_m`. -/
theorem sum_weights (α : ℕ → ℝ) (m : ℕ) :
    ∑ k ∈ range m, α k * transmittance α k = 1 - transmittance α m := by
  induction m with
  | zero => simp [transmittance]
  | succ m ih =>
    rw [Finset.sum_range_succ, ih, transmittance, transmittance, Finset.prod_range_succ]
    ring

theorem transmittance_mem (α : ℕ → ℝ) (h : ∀ j, 0 ≤ α j ∧ α j ≤ 1) (k : ℕ) :
    0 ≤ transmittance α k ∧ transmittance α k ≤ 1 := by
  unfold transmittance
  exact ⟨Finset.prod_nonneg fun j _ => by linarith [(h j).2],
    Finset.prod_le_one₀ (fun j _ => by linarith [(h j).2]) fun j _ => by linarith [(h j).1]⟩

/-- **`accumulated` is already in [0, 1]**: the `clip` is a no-op in exact arithmetic. -/
theorem accumulated_mem (α : ℕ → ℝ) (h : ∀ j, 0 ≤ α j ∧ α j ≤ 1) (m : ℕ) :
    0 ≤ ∑ k ∈ range m, α k * transmittance α k ∧ ∑ k ∈ range m, α k * transmittance α k ≤ 1 := by
  rw [sum_weights]
  have := transmittance_mem α h m
  constructor <;> linarith [this.1, this.2]

/-- **The image is already in [0, 1]** for colours and background in [0, 1]: `colour + (1 − acc) · bg` lies
between 0 and `(1 − T) + T = 1`. -/
theorem image_mem (α rgb : ℕ → ℝ) (h : ∀ j, 0 ≤ α j ∧ α j ≤ 1) (hc : ∀ j, 0 ≤ rgb j ∧ rgb j ≤ 1)
    {bg : ℝ} (hb : 0 ≤ bg ∧ bg ≤ 1) (m : ℕ) :
    let acc := ∑ k ∈ range m, α k * transmittance α k
    let colour := ∑ k ∈ range m, α k * transmittance α k * rgb k
    0 ≤ colour + (1 - acc) * bg ∧ colour + (1 - acc) * bg ≤ 1 := by
  intro acc colour
  have hw : ∀ k, 0 ≤ α k * transmittance α k := fun k => mul_nonneg (h k).1 (transmittance_mem α h k).1
  have hc0 : 0 ≤ colour := Finset.sum_nonneg fun k _ => mul_nonneg (hw k) (hc k).1
  have hc1 : colour ≤ acc := Finset.sum_le_sum fun k _ => by
    nlinarith [hw k, (hc k).2]
  have ha := accumulated_mem α h m
  constructor
  · nlinarith [ha.2, hb.1]
  · nlinarith [ha.1, ha.2, hb.2]

end Idol3d
