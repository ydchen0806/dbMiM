# MEC 8 nm R76 R58/R71 Checkpoint-Soup Plan

R76 tests whether the aggregate improvement of the R71 EMA mechanism can be
borrowed without crossing the strict M08 guard. It performs no training and
introduces no new data, labels, architecture, loss, or calibration family.

## Audited endpoints

- R58 final SHA256:
  `4715c6679808eac5e073159202be1cca77307ab90202bd683f8ab97cacc55e09`.
- R71 EMA SHA256:
  `9c82d7ac4d4196a19966451343fcc033b7ea279a428b5913a54d9f4075790a45`.
- Both checkpoints record seed1789, step12000, `unetr_aniso_em`, and identical
  214-key shapes. All non-floating states agree.
- R71-to-R58 relative L2 distance is `1.8140%` overall, `0.1749%` in the 77
  encoder tensors, and `2.5505%` in the 137 decoder/head tensors.

## Controlled candidates

For each floating tensor, construct
`theta = (1-alpha) * theta_R58 + alpha * theta_R71_EMA`. Preserve non-floating
R58 state exactly. Evaluate only alpha `.010`, `.025`, `.050`, `.100`, and
`.200`. Alpha zero is the existing R58 anchor; larger interpolation is excluded
because R71's EMA endpoint already severely fails M08.

Each candidate is independently evaluated on M08/M10 center crops with the
frozen 40-setting WaterZ grid and must return exactly 80 finite rows with an
empty failure ledger. M11/M12 remain absent. Promote only a nonzero alpha whose
mean ARAND, mean VOI, M08 ARAND, and M08 VOI all do not regress versus R58.
Otherwise reject this checkpoint-space family before full-block work.

## Outcome

Completed and rejected. Successful SiFlow UUID
`51a8f0f0-3873-4676-9e45-17371f9d1077` returned five independently audited
80-row M08/M10 WaterZ results. All five alphas have zero strict R58 gate
passes. The least-regressed alpha `.050` has mean ARAND/VOI
`0.030034391192/0.338540905377` and M08
`0.035741292818/0.342101735986`. No full-block task was submitted.
