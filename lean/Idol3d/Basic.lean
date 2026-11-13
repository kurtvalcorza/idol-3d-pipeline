/-
Vectors and 3×3 matrices as NumPy uses them: `Fin 3 → ℝ`, rows given explicitly.
-/
import Mathlib

namespace Idol3d

open Real Matrix

/-- `np.cross(a, b)`. -/
def cross (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

/-- `np.linalg.norm(v)`. -/
noncomputable def len (v : Fin 3 → ℝ) : ℝ := √(v ⬝ᵥ v)

/-- The matrix with the given rows (`np.stack([r0, r1, r2])`). -/
def ofRows (r₀ r₁ r₂ : Fin 3 → ℝ) : Matrix (Fin 3) (Fin 3) ℝ := Matrix.of ![r₀, r₁, r₂]

/-- Rotation about the object's vertical (z) axis by `φ`, counter-clockwise seen from above. -/
noncomputable def rotZ (φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![cos φ, -sin φ, 0; sin φ, cos φ, 0; 0, 0, 1]

theorem dot_self_nonneg (v : Fin 3 → ℝ) : 0 ≤ v ⬝ᵥ v := by
  simp only [dotProduct, Fin.sum_univ_three]
  nlinarith [mul_self_nonneg (v 0), mul_self_nonneg (v 1), mul_self_nonneg (v 2)]

end Idol3d
