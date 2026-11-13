/-
Executable `Float` versions of the functions formalised over ℝ in `Idol3d/`, written line by line from
`src/idol_3d_pipeline/render.py` and `src/idol/camera.py`. `tests/test_lean_reference.py` runs them and the
Python on the same inputs. Nothing here is proved: it is a second, independent implementation.
-/
namespace Idol3d.Ref

abbrev V := Array Float

def get (v : V) (i : Nat) : Float := v.getD i 0.0
def dot (a b : V) : Float := get a 0 * get b 0 + get a 1 * get b 1 + get a 2 * get b 2
def norm (a : V) : Float := Float.sqrt (dot a a)
def cross (a b : V) : V :=
  #[get a 1 * get b 2 - get a 2 * get b 1, get a 2 * get b 0 - get a 0 * get b 2,
    get a 0 * get b 1 - get a 1 * get b 0]
def scale (c : Float) (a : V) : V := a.map (c * ·)
def neg (a : V) : V := a.map (- ·)
def pi : Float := 3.141592653589793

/-- `render.camera_pose` for `convention` 0..3 = `CONVENTIONS`; returns centre ++ axes (row-major). -/
def cameraPose (yaw pitch radius : Float) (convention : Nat) : V :=
  let toward := convention == 0 || convention == 1
  let mirrored := convention == 1 || convention == 3
  let d : V := #[Float.sin yaw * Float.cos pitch, Float.cos yaw * Float.cos pitch, Float.sin pitch]
  let centre := scale radius (if toward then d else neg d)
  let forward := scale (1 / norm centre) (neg centre)
  let right := cross forward #[0, 0, 1]
  let right := if norm right < 1e-6 then #[1, 0, 0] else right
  let right := scale (1 / norm right) right
  let right := if mirrored then neg right else right
  let up := cross right forward
  centre ++ right ++ up ++ forward

/-- `render.quaternion_to_matrix` for one (w, x, y, z). -/
def quaternionToMatrix (q : V) : V :=
  let n := Float.sqrt (get q 0 * get q 0 + get q 1 * get q 1 + get q 2 * get q 2 + get q 3 * get q 3)
  let n := if n < 1e-12 then 1e-12 else n
  let w := get q 0 / n; let x := get q 1 / n; let y := get q 2 / n; let z := get q 3 / n
  #[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]

/-- `decode_camera`'s yaw and pitch of a direction, then `camera_pose`'s `d` rebuilt from them. -/
def decodeThenDirection (v : V) : V :=
  let n := norm v
  let u : V := if n > 1e-6 then scale (1 / n) v else #[0, 0, 1]
  let z := get u 2
  let yaw := Float.atan2 (get u 0) (get u 1)
  let pitch := Float.asin (if z < -1 then -1 else if z > 1 then 1 else z)
  #[yaw, pitch, Float.sin yaw * Float.cos pitch, Float.cos yaw * Float.cos pitch, Float.sin pitch]

/-- Front-to-back compositing of one pixel, nearest first: alphas then colours (one channel) then the
background. Returns `[accumulated, colour + (1 - accumulated) · background]` before the final clip. -/
def composite (xs : V) : V := Id.run do
  let k := (xs.size - 1) / 2
  let bg := get xs (xs.size - 1)
  let mut logT := 0.0
  let mut acc := 0.0
  let mut col := 0.0
  for i in [0:k] do
    let a := get xs i
    let w := a * Float.exp logT
    acc := acc + w
    col := col + w * get xs (k + i)
    logT := logT + Float.log (1 - a)
  return #[acc, col + (1 - acc) * bg]

end Idol3d.Ref
