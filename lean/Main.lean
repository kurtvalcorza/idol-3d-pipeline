/-
`lake exe idol3d-ref`: one request per line, `<function> <u64> ...` with each number the IEEE-754 bit pattern
of a double; one line of bit patterns back per request.
-/
import Idol3d.Reference.Float

open Idol3d.Ref

def bits (s : String) : Float := Float.ofBits (String.toNat! s).toUInt64

def run (fn : String) (xs : Array Float) : Except String V :=
  match fn with
  | "camera_pose" => pure (cameraPose xs[0]! xs[1]! xs[2]! xs[3]!.toUInt64.toNat)
  | "quaternion_to_matrix" => pure (quaternionToMatrix xs)
  | "decode_then_direction" => pure (decodeThenDirection xs)
  | "composite" => pure (composite xs)
  | other => throw s!"unknown function {other}"

partial def loop (stdin stdout : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return
  match (line.trimAscii.toString.splitOn " ").filter (· ≠ "") with
  | [] => loop stdin stdout
  | fn :: args =>
    match run fn (args.toArray.map bits) with
    | .ok v => stdout.putStrLn (" ".intercalate (v.toList.map fun x => toString x.toBits))
    | .error e => stdout.putStrLn s!"error {e}"
    stdout.flush
    loop stdin stdout

def main : IO Unit := do
  loop (← IO.getStdin) (← IO.getStdout)
