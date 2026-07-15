# MEC8nm R72 Photometric-Consistency Crop Report

## Contract and execution

- SiFlow UUID: `ff377784-06ce-4b4e-a4f0-bb7a7b8d0a2f`
- Task: `mec8nm-r72-consistency-90145185`
- Resource: one 8-GPU `cn-shanghai/changliu/med-model` `sci.g21-3` pod
- TOS output:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r72_seed1789_consistency_stage1`
- Training: 12000 steps, `train_done code=0`, 2058 seconds
- Startup checks: CUDA 8, WaterZ available, 214 R57 keys, train IDs
  `[1,2,3,4,5,6,7,9]`, dataset size 32768
- R72 delta: stopped-teacher photometric affinity consistency, weight `0.05`
- Final logged main/consistency loss: `0.05802399/0.00308658`

The task evaluated only M08/M10. The remote assertion and independent fetch
both confirm exactly 80 finite WaterZ rows, the exact two validation samples,
and an empty failure ledger. No M11/M12 path was evaluated.

## Frozen paired analysis

The standard analyzer compared the independently fetched R72 CSV with the
frozen R58 CSV. After collapsing 16 exact `pred`/zero-bias duplicates, the 64
rows form 32 paired settings. There are zero strict crop-gate passes.

The only Pareto and best-balance point is bias `(-0.20,-0.45,-0.45)`,
temperature `1.15`, threshold `0.50`:

- mean ARAND/VOI: `0.028005094678/0.369409206168`;
- balance: `0.101712043503`;
- worst-block balance: `0.141638016551`;
- M08: ARAND/VOI `0.042332990281/0.473893471718`, `n_pred=226`;
- M10: ARAND/VOI `0.013677199074/0.264924940619`, `n_pred=106`.

Relative to R58, R72 regresses mean ARAND by `24.2131%`, mean VOI by
`18.7111%`, and balance by `21.4309%`. The M08 protection metrics regress much
more: ARAND by `183.1972%` and VOI by `85.0222%`. M10 improves, but that
block-asymmetric tradeoff is exactly what the strict M08 guard forbids.

## Decision

Reject R72 before full-block validation. Do not run M11/M12, deployment, or
2PB inference from this checkpoint. Do not sweep the R72 consistency weight or
photometric ranges: the mechanism repeats the established pattern of helping
M10 while materially fragmenting M08. The selected anchors remain R58 for crop
screening and R57 for M08/M10 full-block validation.

Evidence:

- `mec8nm_remote_results/r72_consistency_20260713/paired_analysis_vs_r58.json`
- `mec8nm_remote_results/r72_consistency_20260713/eval_val_crop64x512/`
