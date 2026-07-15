# MEC8nm R68 Multiscale-Affinity Crop Report

## Runtime Contract

SiFlow UUID `b8939e9c-ffa2-48e5-bed9-a7b173bda89f`
(`mec8nm-r68-msaff-b43f40ff`) completed successfully on one 8-GPU Shanghai
`med-model` pod. Runtime logs confirm CUDA 8, WaterZ, 214 inherited R57 keys,
the fixed MEC01--07/MEC09 training set, six declared affinity offsets, and
`bcar_channels=3`. Training reached 12000 steps without a reported exception.

The hardened evaluator used only M08/M10 and WaterZ. The downloaded metrics
contain exactly 80 raw rows; 16 exact `pred` / zero-bias, temperature-1.0
duplicates collapse to 64 rows and 32 paired settings. No M11/M12 or
full-block evaluation was executed.

## Strict Crop Gate

No R68 point passes the strict R58 gate. The best-balance point uses zero bias,
temperature `1.00`, and threshold `.50`:

- mean ARAND/VOI: `0.029857502936/0.358199176768`;
- balance: `0.103416309023`;
- worst-block balance: `0.160015143163`;
- M08: `0.052169534998/0.490800733464`, `n_pred=428`;
- M10: `0.007545470874/0.225597620072`, `n_pred=165`.

Relative to the R58 anchor, mean ARAND/VOI regress
`32.4292%/15.1087%`, balance regresses `23.4656%`, and M08 ARAND/VOI
regress `249.0012%/91.6232%`. M10 improves, but the required M08 guard and
both aggregate metrics fail.

## Mechanism Check

At the uncalibrated `pred` output, affinity Dice drops from R58 to R68 from
`.979281` to `.968275` on M08 and from `.989033` to `.986410` on M10. All
three near-neighbor channel means and fifth percentiles decrease on both
blocks. The fixed two-voxel auxiliary objective therefore shifts the shared
decoder toward lower near-neighbor affinities globally: this happens to reduce
M10 merges but substantially increases M08 fragmentation. It is not an
isolated z/y/x calibration defect.

## Decision

Reject R68 before full-block validation. Do not run M11/M12, deployment, or
2PB inference from this checkpoint. Do not launch a blind long-range-loss
weight sweep: the observed effect is block-dependent and violates the fixed
M08 guard. Machine-readable paired evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r68_multiscale_affinity_20260711/paired_analysis_vs_r58.json`.
