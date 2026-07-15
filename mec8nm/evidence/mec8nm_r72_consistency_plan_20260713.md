# MEC8nm R72 Photometric-Consistency Plan

## Hypothesis

R58 performs well on fixed center crops but later supervised candidates fail to
transfer consistently across M08 and M10 and into full-block inference. R72
tests whether explicitly matching affinity probabilities for two
coordinate-aligned photometric views improves invariance to the normalization
and intensity changes encountered by larger inference windows.

This is an augmentation-consistency experiment, not another gamma-dose or
calibration sweep. The supervised R58 view and loss are unchanged. A second
view is used only by a stopped-teacher consistency term.

## Strict controlled delta

R72 is byte-equivalent to the R58 config after removing these two fields:

- `output_dir` changes from R58 to R72;
- `train.augmentation_consistency` is added with weight `0.05`.

The second view keeps voxel coordinates fixed and samples per-volume gamma
`0.85--1.20`, gain `0.90--1.10`, bias `-0.05--0.05`, and Gaussian noise
standard deviation `0.02`. It is clamped to `[0,1]`. The loss is MSE between
the second-view affinity probabilities and detached first-view probabilities
for the three nearest-neighbor channels.

Everything else remains R58: seed `1789`, PublicEM R57 initialization, M01--07
and M09 supervision, 12k steps, anisotropic UNETR, weighted MSE, boundary Dice,
BCAR, membrane weighting, augmentation, optimizer, and checkpoint schedule.

## Evaluation contract

- One 8-GPU crop-screen task only.
- Validation blocks M08/M10 only; M11/M12 paths are forbidden.
- WaterZ only, exactly 80 finite raw rows, empty failure ledger.
- Analyze against the frozen R58 CSV.
- Pass only if mean ARAND, mean VOI, M08 ARAND, and M08 VOI all do not regress.
- No full-block task after any crop-gate failure.

## Local verification

- Four unit tests pass: bounded/aligned view, nonzero perturbation, zero
  identical-logit loss, stopped reference gradient, and invalid-range failure.
- A one-step CPU training smoke completed and logged both consistency loss and
  view L1 distance.
- A real M01 train crop retained shape `1x1x32x160x160`; its sampled view was
  bounded in `[0,1]` with mean absolute perturbation `0.0431056`.
- Config-delta audit confirms only `output_dir` and
  `train.augmentation_consistency` differ from R58.
- The hardened launcher rejects `EVAL_SPLITS=test`, fixes eight train IDs,
  WaterZ, 80 rows, and eight GPUs.
- MEC workspace verifier, Python compilation, and shell syntax checks pass.
