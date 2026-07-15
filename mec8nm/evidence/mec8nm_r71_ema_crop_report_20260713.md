# MEC8nm R71 EMA Crop Report

R71 UUID `41bc1e3b-6b6e-4fdf-9de9-6af7ab993381` completed 12000 steps on one
8-GPU Shanghai `med-model` pod. Runtime confirms CUDA 8, WaterZ, 214 R57 keys,
the fixed MEC01--07/MEC09 training set, and `ema_decay=.999` with EMA weights
used by the checkpoint. The M08/M10 evaluator returned exactly 80 finite rows.

The best aggregate point is uncalibrated temperature `1.00`, threshold `.50`:
mean ARAND/VOI `0.020969504209/0.299190914044`, improving R58 by
`6.9924%/3.8538%`. M10 improves strongly to
`0.004363798372/0.165133660030`, but M08 regresses to
`0.037575210047/0.433248168057` (`151.3688%/69.1530%` worse). No setting passes
the strict R58 gate.

EMA is a positive aggregate mechanism signal but not a selectable checkpoint.
Reject R71 before full-block, M11/M12, deployment, or 2PB inference. Evidence:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r71_ema_20260713/paired_analysis_vs_r58.json`.
