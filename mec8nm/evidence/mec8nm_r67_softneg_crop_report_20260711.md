# MEC8nm R67 Soft-Negative Crop Gate

## Scope

- Run: `finetune_mec8nm_r67_seed1789_softneg_stage1`.
- SiFlow UUID: `94db3a0b-6084-4085-a6ad-1c05f3df07f0`.
- Training: MEC01--07 and MEC09 only, 12000 steps, eight GPUs.
- Validation: M08/M10 only, WaterZ only, 80 raw crop rows.
- No full-block, M11/M12, deployment, or 2PB inference path was run.

## Controlled Change

R67 is exactly the R58 configuration except
`train.loss.weight_alpha: 1.0 -> 0.5` under `weighted_mse`. This leaves every
other training setting fixed: seed, public R57 initialization, data IDs, model,
crop geometry, schedule, augmentation, boundary Dice, BCAR, membrane weighting,
optimizer, and evaluation grid.

The preceding boundary audit showed widened training affinity-positive fractions
of `.788879/.852044/.853948` (z/y/x), compared with raw M08/M10 values
`.960546/.969814/.968969`. At `weight_alpha=1.0`, rare training negatives have
equal aggregate loss mass to positives. R67 halves that negative aggregate mass
to reduce the observed full-block fragmentation tendency.

## Runtime Evidence

The task completed successfully after 12000 steps in `1162` seconds. Remote
startup verified WaterZ, CUDA 8, 214 loaded R57 keys, the fixed training set,
and `weighted_mse` with `weight_alpha=.5`. The evaluator then asserted exactly
80 WaterZ rows for M08/M10 with finite ARAND/VOI and no failures.

Downloaded validation evidence:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r67_softneg_20260711/eval_val_crop64x512/eval_val_crop64x512/eval_finetune_mec8nm_r67_seed1789_softneg_stage1_val_crop64x512/`

The fixed paired analyzer confirms 80 raw rows, 16 exact `pred` / zero-bias,
temperature-1.0 duplicates, 64 retained rows, 32 paired configurations, and
zero strict crop-gate passes. The machine-readable analysis is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r67_softneg_20260711/paired_analysis_vs_r58.json`.

## Selection Result

The R58 anchor is zero bias, temperature `1.15`, threshold `.50`:

- mean ARAND/VOI: `0.022546003667/0.311183450751`;
- balance: `0.083761227436`;
- M08: `0.014948239287/0.256127966754`, `n_pred=242`;
- M10: `0.030143768046/0.366238934749`, `n_pred=120`.

R67 has two Pareto points. Its best-balance point is bias
`(-.20,-.45,-.45)`, temperature `1.00`, threshold `.50`:

- mean ARAND/VOI: `0.052563021514/0.399776827409`;
- balance: `0.144960263451`, worst-block balance `0.210299685945`;
- M08: `0.016472016743/0.300693537996`, `n_pred=154`;
- M10: `0.088654026284/0.498860116822`, `n_pred=81`.

R67 reduces predicted instances but creates harmful M10 merges. Relative to
R58 it is `133.1368%` worse in mean ARAND, `28.4698%` worse in mean VOI, and
`73.0637%` worse in balance. M08 regresses `10.1937%/17.3997%` in ARAND/VOI;
M10 regresses `194.1040%/36.2117%`.

## Decision

Reject R67 before full-block validation. The reduced negative-loss pressure
does not solve the train/validation topology mismatch: it trades the historical
fragmentation tendency for severe M10 merging while still regressing M08.
Do not run this checkpoint on full M08/M10 blocks, M11/M12, deployment, or 2PB
inference.
