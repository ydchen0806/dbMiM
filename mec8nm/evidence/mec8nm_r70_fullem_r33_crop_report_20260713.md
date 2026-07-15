# MEC8nm R70 FullEM-R33 Crop Report

R70 UUID `d49b444f-fcd6-44c3-8250-7a2eb176f799` completed 12000 steps on one
8-GPU Shanghai `med-model` pod. Runtime confirmed WaterZ, CUDA 8, 214 loaded
R33 keys, the fixed MEC01--07/MEC09 supervision, and unchanged R58 losses. The
M08/M10-only evaluator returned exactly 80 finite rows without errors.

No setting passes the R58 crop gate. Best balance is zero bias, temperature
`1.15`, threshold `.50`: mean ARAND/VOI
`0.053447533042/0.487475984541`; M08 is
`0.041557721757/0.460674361248` (`n_pred=340`) and M10 is
`0.065337344326/0.514277607833` (`n_pred=168`). Relative to R58, mean
ARAND/VOI regress `137.0599%/56.6523%` and M08 regresses
`178.0108%/79.8610%`.

Reject R70 before full-block, M11/M12, deployment, or 2PB inference. Broader
FullEM self-supervised exposure through the R33 initializer does not transfer
automatically to MEC; data scale alone is insufficient without domain-aligned
pretraining. Evidence:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r70_fullem_r33_20260713/paired_analysis_vs_r58.json`.
