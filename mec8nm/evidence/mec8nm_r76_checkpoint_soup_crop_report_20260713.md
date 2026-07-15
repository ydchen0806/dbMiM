# MEC8nm R76 R58/R71 Checkpoint-Soup Crop Report

Canonical versionable report:

`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_r76_checkpoint_soup_crop_report_20260713.md`

R76 tested a bounded checkpoint-space interpolation between R58 final and the
R71 EMA mechanism signal. Both sources record seed1789, step12000,
`unetr_aniso_em`, identical 214-key shapes, and equal non-floating states. The
R71-to-R58 relative L2 distance is `1.8140%` overall, `0.1749%` in the 77
encoder tensors, and `2.5505%` in the 137 decoder/head tensors. Five nonzero
R71 EMA fractions were evaluated: `.010`, `.025`, `.050`, `.100`, and `.200`.

Two launcher-only attempts failed before inference. UUID
`8597f63a-4517-4100-9fce-17eef686fbf9` stopped on a readonly shell assignment
after successfully constructing all five byte-verified soups; UUID
`b507e2de-241c-4a1c-ae6c-37c35a474e05` stopped because the eval-only bundle
omitted an evaluator import dependency. Successful UUID
`51a8f0f0-3873-4676-9e45-17371f9d1077`
(`mec8nm-r76-checkpoint-soup-r3-8db4ebe2`) completed in 26m01s. Every alpha
independently returned exactly 80 finite M08/M10 WaterZ rows and an empty
failure ledger. M11/M12 were absent.

No alpha has a strict R58 gate pass. The least-regressed point is alpha `.050`
at zero bias, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.030034391192/0.338540905377`; M08
`0.035741292818/0.342101735986`; M10
`0.024327489565/0.334980074768`. Relative to R58, mean ARAND/VOI regress
`33.2138%/8.7914%`, while M08 regresses `139.1004%/33.5667%`.

Even alpha `.010` produces mean `0.061763191963/0.409615369963`, showing that
the independently trained R58 and R71 endpoints are not linearly connected in
a usable low-loss basin despite their small Euclidean distance. Reject R76
before full-block validation, M11/M12, deployment, or 2PB inference. Do not
expand the alpha range or repeat cross-run linear model soup. A same-run R58
intermediate-checkpoint screen remains a distinct, admissible early-stopping
test because it does not mix independently diverged trajectories.
