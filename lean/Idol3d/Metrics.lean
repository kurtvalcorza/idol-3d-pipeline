/-
`src/idol_3d_pipeline/metrics.py`: the `.ply` axis change and the silhouette IoU.
-/
import Idol3d.Basic

namespace Idol3d

open Matrix

/-- `save_ply`'s axis change, `(x, y, z) ↦ (x, −z, y)`: the +90° rotation about x its comment names. -/
def plyAxes : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, -1; 0, 1, 0]

/-- **The `.ply` axis change is a proper rotation** (+90° about x), not a reflection, so the file holds the same
object, not its mirror image. **But it sends the object's up (+z) to −y**, not +y: in a y-up viewer the object
is upside down; in a y-down one (the OpenCV convention many 3DGS tools use) it is upright. The SDK's comment
calls the target "the y-up convention viewers use"; `(x, z, −y)` would be the map to y-up. -/
theorem plyAxes_rotation :
    plyAxes * plyAxesᵀ = 1 ∧ plyAxes.det = 1 ∧ plyAxes *ᵥ ![0, 0, 1] = ![0, -1, 0] := by
  refine ⟨?_, ?_, ?_⟩
  · ext i j; fin_cases i <;> fin_cases j <;> simp [plyAxes, Matrix.mul_apply, Fin.sum_univ_three]
  · simp [plyAxes, Matrix.det_fin_three]
  · ext i; fin_cases i <;> simp [plyAxes, Matrix.mulVec, dotProduct, Fin.sum_univ_three]

variable {α : Type*} [DecidableEq α]

/-- `silhouette_iou`: `NaN` (here `none`) when both silhouettes are empty. -/
def silhouetteIoU (A B : Finset α) : Option ℚ :=
  if (A ∪ B).card = 0 then none else some ((A ∩ B).card / (A ∪ B).card)

theorem silhouetteIoU_mem {A B : Finset α} {v : ℚ} (h : silhouetteIoU A B = some v) : 0 ≤ v ∧ v ≤ 1 := by
  unfold silhouetteIoU at h
  split_ifs at h with h0
  cases h
  refine ⟨by positivity, div_le_one_of_le₀ ?_ (by positivity)⟩
  exact_mod_cast Finset.card_le_card Finset.inter_subset_union

theorem silhouetteIoU_comm (A B : Finset α) : silhouetteIoU A B = silhouetteIoU B A := by
  simp [silhouetteIoU, Finset.union_comm, Finset.inter_comm]

/-- Empty silhouettes give `NaN`, a third convention beside insect-3d-reconstruction's `None` and `0`. -/
theorem silhouetteIoU_empty : silhouetteIoU (∅ : Finset α) ∅ = none := by simp [silhouetteIoU]

end Idol3d
