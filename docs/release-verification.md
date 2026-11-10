# Release verification

No clean-runtime execution of `tutorials/idol_3d_colab.ipynb` has been recorded yet.

| Date | Runtime | Notebook blob | Source revision | Scope | Result | Evidence |
|---|---|---|---|---|---|---|
| 2026-11-10 | local CPython 3.13, no GPU | generated | uncommitted | Sections 1–2 (runtime check, carrier) executed from the notebook; carried `prepare` stage run with the repository's NumPy/Pillow | pass (not REL1: no isolated environment, no model) | session log only |

A hosted Colab T4 `Run all` is required before any Release-grade claim; the RUN13/MOD4 deviation (see `STATUS.md`)
caps the notebook at Candidate regardless.
