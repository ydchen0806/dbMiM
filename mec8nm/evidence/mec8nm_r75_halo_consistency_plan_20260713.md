# MEC 8 nm R75 Halo-Context Consistency Plan

R75 is the first post-R74 experiment to target crop/full-block context transfer
directly. It starts from the exact PublicEM-R57 checkpoint and otherwise keeps
the complete R58 supervised recipe fixed.

## Controlled hypothesis

For each ordinary `32x160x160` supervised crop, the stopped teacher sees the
unchanged image. The student sees the same image after only the outer
`4x16x16` z/y/x halo has been reconstructed by reflection from the central
`24x128x128` core. That core remains byte-identical between views. A weight
`.05` probability-MSE term matches the three affinity channels only inside the
unchanged core. This tests whether predictions on identical central content
can be stabilized against changes in surrounding window context.

No new images, labels, model parameters, normalization rule, or supervised
loss term are introduced. R72 photometric consistency is disabled, and R75
does not use R73/R74 adapted checkpoints.

## Frozen protocol

- Initialization: exact PublicEM-R57 214-key checkpoint.
- Supervised training: M01--M07/M09 only, exact R58 recipe and 12000 steps.
- Validation: M08/M10 center crop, WaterZ only, exactly 80 finite rows.
- Remote download IDs: M01--M10 only; M11/M12 are absent.
- Exactly one 8-GPU Shanghai `med-model` wave.
- Pass only if mean ARAND, mean VOI, M08 ARAND, and M08 VOI all do not regress
  versus R58. A crop failure forbids full-block work.

## Outcome

Completed and rejected. SiFlow UUID
`1dde395e-27a4-46e8-b5a2-5be9199049ed` succeeded with exactly 80 finite
M08/M10 WaterZ rows and no failures, but the paired analyzer found zero strict
R58 gate passes. Best mean ARAND/VOI was
`0.036896625527/0.387556128221`, with M08
`0.040070609288/0.377633169630`. No full-block task was submitted.
