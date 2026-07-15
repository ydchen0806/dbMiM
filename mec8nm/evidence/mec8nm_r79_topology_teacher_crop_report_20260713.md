# MEC8nm R79 Topology-Routed Dual-Teacher Crop Report

R79 tested whether a single student could retain R58's simple-volume behavior
while borrowing R71 EMA's complex-volume signal. It kept the full R58 line
fixed: PublicEM-R57 student initialization, seed 1789, M01--07/M09 sampling,
augmentation, `unetr_aniso_em`, supervised loss, optimizer, and 12000 steps.
The only training delta was a weight-`.10` probability-MSE term from frozen
R58 and R71-EMA teachers.

The simple-teacher route used raw training-label crop statistics before
boundary widening and augmentation:
`sigmoid((fg-.95)/.02) * sigmoid((.05-boundary)/.02) *
sigmoid((5.5-unique)/2)`. It changed neither crop sampling nor the main loss.
A 1024-crop training-only audit found route mean `.110561` and 139 crops at
route >= `.5`. Both teachers loaded all 214 architecture-compatible keys and
were frozen; final inference used only the student.

The first local bundle upload timed out through inherited proxy
`192.168.32.28:18000` before any SiFlow task was created. Reusing the exact
tar (SHA256 `20a5f63e2baa38e85eb38d68c3b9a150bc6257946c223afef183dfe150cb8cff`)
with proxy variables cleared succeeded. SiFlow UUID
`4c50adc6-ee6c-4ccf-bd28-e2341af5e770`
(`mec8nm-r79-topology-teacher-1a465cfa`) ran on one 8-GPU Shanghai
`med-model` pod. Startup verified CUDA 8, WaterZ, R57/R58/R71 hashes, all
three 214-key loads, M01--07/M09 training paths, dataset size 32768, and the
declared route. Training reached step 12000 in 1755 seconds with no runtime
errors.

The remote assertion and independent TOS fetch contain exactly 80 finite
WaterZ rows, samples exactly M08/M10, and an empty failure ledger. The paired
analyzer finds zero strict R58 gate passes. Best balance is zero bias,
temperature `1.00`, threshold `.50`: mean ARAND/VOI
`0.062021381189/0.412646966283`; M08
`0.035591466186/0.337509064870`; M10
`0.088451296192/0.487784867696`. Relative to R58, mean ARAND/VOI regress
`175.0881%/32.6057%`, while M08 regresses `138.0980%/31.7736%`.

At R58's exact zero-bias, temperature-`1.15`, threshold-`.50` point, R79 is
also negative: mean `0.059614581022/0.441990110322`, M08
`0.033324479787/0.380752739921`, and M10
`0.085904682258/0.503227480722`. The two Pareto points are both zero-bias
temperature variants and neither approaches the anchor.

Reject R79 before full-block validation, M11/M12, deployment, or 2PB
inference. Do not sweep the distillation weight, topology thresholds/widths,
or teacher pair. R58 remains the crop anchor and R57 remains the M08/M10
full-block anchor. Crop-level routing between two already biased full-network
teachers is too coarse: the next method must not reuse this teacher-routing
family without a materially different spatial or representation-level
mechanism.

Evidence hashes:

- metrics CSV: `06dcae9bd6ea6a5c198895be0f82b896e9bac9d32fe97d1a376300d597a021af`
- summary JSON: `bf75325a67af4555783e6fb2ecb4bec61a1b82269f4b6d876aa342cc7902b036`
- paired analysis: `de6ef4fbdee823a6f274fe50ade9484840ff160357b9a607c460f8424a6b0abd`
