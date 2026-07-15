# MEC 8 nm R73 Decoder-Aware Domain Pretraining Plan

R73 tests one new family after R66--R72 were rejected: domain-aligned,
label-free pretraining on the frozen MEC training blocks. It does not tune a
rejected loss, calibration, EMA, or consistency branch.

## Controlled hypothesis

Initialize the exact R58 UNETR-EM encoder, convolutional decoder, and
three-channel affinity head from PublicEM-R57. Adapt those shared modules for
4000 steps using only raw crops from M01--M07 and M09, with 75% masked image
reconstruction plus a weak (`0.1`) raw-gradient affinity proxy. Then discard
the reconstruction-only parameters and run the unchanged R58 supervised
recipe for 12000 steps.

The short `2e-5` adaptation stage is intentionally conservative: it gives the
random reconstruction decoder time to learn MEC appearance while limiting
drift in R57's already useful segmentation representation. The learned policy
module is disabled.

## Frozen protocol

- Unlabeled pretraining raw blocks: M01--M07 and M09 only.
- Supervised blocks and labels: M01--M07 and M09 only.
- Validation: M08/M10 center crop only, WaterZ only.
- Frozen and absent from the remote job: M11/M12.
- R57 initializer SHA256:
  `9b67e337aba837ee6c26e5c4b80627ab3745fdb3fe152190d0164c6e9a2fe2ba`.
- Exactly one 8-GPU job may run.
- Evaluation must contain exactly 80 finite rows and no backend failures.

## Decision gate

Compare the best common WaterZ operating point with the frozen R58 crop
anchor. R73 passes only if all four quantities do not regress: mean ARAND,
mean VOI, M08 ARAND, and M08 VOI. A crop pass permits a later M08/M10
full-block validation; it does not permit M11/M12 access by itself.
