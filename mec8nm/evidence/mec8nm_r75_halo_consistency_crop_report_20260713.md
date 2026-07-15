# MEC8nm R75 Halo-Context Consistency Crop Report

Canonical versionable report:

`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_r75_halo_consistency_crop_report_20260713.md`

R75 started from the exact PublicEM-R57 214-key checkpoint and otherwise held
the R58 supervised recipe fixed. For each ordinary `32x160x160` crop, a
stopped teacher saw the unchanged image. The student saw a view in which only
the outer `4x16x16` z/y/x halo was reflect-reconstructed from the central
`24x128x128` core. That core remained byte-identical, and a weight-`.05`
probability-MSE term matched the three affinity channels only within it.

SiFlow UUID `1dde395e-27a4-46e8-b5a2-5be9199049ed`
(`mec8nm-r75-halo-consistency-e68c2276`) succeeded in 42m25s on one 8-GPU
Shanghai `med-model` pod. Startup verified CUDA 8, WaterZ, train IDs
M01--M07/M09, eight paths, 214 loaded R57 keys, the unchanged R58 loss, and
the exact halo configuration. Training reached step 12000 in 2059 seconds.
The final task assertion and independent TOS fetch confirm exactly 80 finite
WaterZ rows, 40 each for M08/M10, with an empty failure ledger and no M11/M12
access.

The paired analyzer found zero strict R58 gate passes. R75's only Pareto and
best-balance point is bias `(-.20,-.45,-.45)`, temperature `1.00`, threshold
`.50`: mean ARAND/VOI `0.036896625527/0.387556128221`; M08
`0.040070609288/0.377633169630`; M10
`0.033722641766/0.397479086812`. Relative to R58, mean ARAND/VOI regress
`63.6504%/24.5427%`, balance regresses `42.7636%`, and M08 ARAND/VOI regress
`168.0624%/47.4393%`.

Reject R75 before full-block validation, M11/M12, deployment, or 2PB
inference. Do not sweep its weight, halo width, or reflection mode. The result
shows that forcing center predictions to ignore strongly altered window
context degrades both blocks rather than repairing crop/full-block transfer.
