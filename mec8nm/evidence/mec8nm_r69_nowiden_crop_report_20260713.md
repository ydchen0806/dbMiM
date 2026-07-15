# MEC8nm R69 No-Widen Crop Report

## Runtime Contract

SiFlow UUID `ae98ae06-08b2-4fac-b5bf-936196b79b0f`
(`mec8nm-r69-nowiden-30543597`) completed successfully on one 8-GPU Shanghai
`med-model` pod. Runtime logs confirm CUDA 8, WaterZ, 214 loaded R57 keys,
MEC01--07/MEC09 training only, 32768 training samples, 12000 steps, and the
unchanged R58 loss configuration. No exception was reported.

R69 differs from R58 only in `data.widen_border: true -> false` and its output
directory. Its evaluator was hard locked to M08/M10 validation and WaterZ. The
downloaded metrics contain exactly 80 raw rows; duplicate collapse leaves 64
rows and 32 paired settings. M11/M12 and full-block inference were not used.

## Strict Crop Gate

No point passes the strict R58 gate. The best-balance point uses bias
`(-.20,-.45,-.45)`, temperature `1.00`, and threshold `.50`:

- mean ARAND/VOI: `0.036164090089/0.396930915975`;
- balance: `0.119810873482`;
- worst-block balance: `0.143499603675`;
- M08: `0.045930476087/0.448332741337`, `n_pred=234`;
- M10: `0.026397704092/0.345529090612`, `n_pred=106`.

Relative to R58, mean ARAND/VOI regress `60.4013%/27.5553%`, balance regresses
`43.0386%`, and M08 ARAND/VOI regress `207.2635%/75.0425%`. M10 improves
slightly, but all aggregate and M08 guard conditions fail.

## Decision

Reject R69 before full-block validation. Removing the artificial zero-label
border band alone does not improve cross-block generalization: it helps M10 but
substantially harms M08. Do not run a border-radius sweep, M11/M12, deployment,
or 2PB inference from this checkpoint. Machine-readable evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r69_nowiden_20260713/paired_analysis_vs_r58.json`.
