/-
`src/idol_3d_pipeline/render.py::quaternion_to_matrix`: each Gaussian's (w, x, y, z) quaternion, normalised,
to its rotation matrix.
-/
import Idol3d.Basic

namespace Idol3d

open Real Matrix

/-- The matrix `quaternion_to_matrix` builds from an already normalised (w, x, y, z). -/
def quatMatrix (w x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y);
     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x);
     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]

/-- The homogeneous form (quadratic entries), equal to `quatMatrix` on unit quaternions. -/
def quatMatrixH (w x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y);
     2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x);
     2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]

theorem quatMatrix_eq_H {w x y z : ℝ} (h : w * w + x * x + y * y + z * z = 1) :
    quatMatrix w x y z = quatMatrixH w x y z := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [quatMatrix, quatMatrixH] <;> linear_combination -h

theorem quatMatrixH_mul_transpose (w x y z : ℝ) :
    quatMatrixH w x y z * (quatMatrixH w x y z)ᵀ =
      (w * w + x * x + y * y + z * z) ^ 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [quatMatrixH, Matrix.mul_apply, Fin.sum_univ_three] <;> ring

theorem det_quatMatrixH (w x y z : ℝ) :
    (quatMatrixH w x y z).det = (w * w + x * x + y * y + z * z) ^ 3 := by
  simp [quatMatrixH, Matrix.det_fin_three]; ring

/-- `q / |q|` for a non-zero quaternion (the Python's `.clip(1e-12)` only guards `q = 0`). -/
noncomputable def normalise (w x y z : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let n := √(w * w + x * x + y * y + z * z)
  (w / n, x / n, y / n, z / n)

/-- **`quaternion_to_matrix` returns a proper rotation for every non-zero quaternion**, unit or not, because
it normalises first. -/
theorem quaternion_to_matrix_rotation {w x y z : ℝ} (h : w * w + x * x + y * y + z * z ≠ 0) :
    let q := normalise w x y z
    let r := quatMatrix q.1 q.2.1 q.2.2.1 q.2.2.2
    r * rᵀ = 1 ∧ r.det = 1 := by
  intro q r
  have h0 : 0 ≤ w * w + x * x + y * y + z * z := by
    nlinarith [mul_self_nonneg w, mul_self_nonneg x, mul_self_nonneg y, mul_self_nonneg z]
  have hpos : 0 < w * w + x * x + y * y + z * z := lt_of_le_of_ne h0 (Ne.symm h)
  have hn : q.1 * q.1 + q.2.1 * q.2.1 + q.2.2.1 * q.2.2.1 + q.2.2.2 * q.2.2.2 = 1 := by
    simp only [q, normalise]
    set S := w * w + x * x + y * y + z * z
    set n := √S
    have hs : n ^ 2 = S := sq_sqrt hpos.le
    have hs0 : n ≠ 0 := (sqrt_pos.2 hpos).ne'
    have : w / n * (w / n) + x / n * (x / n) + y / n * (y / n) + z / n * (z / n) = S / n ^ 2 := by
      simp only [S]; field_simp
    rw [this, hs, div_self h]
  simp only [r]
  rw [quatMatrix_eq_H hn, quatMatrixH_mul_transpose, det_quatMatrixH, hn]
  simp

end Idol3d
