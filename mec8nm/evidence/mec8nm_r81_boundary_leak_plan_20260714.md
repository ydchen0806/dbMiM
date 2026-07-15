# MEC8nm R81 Instance-Pair Boundary-Leak Plan

## Hypothesis

R80 improves M10 and aggregate crop metrics but reduces M08 predicted instances
from 242 to 198 while increasing M08 VOI merge from `.081985` to `.213364`.
The failure is consistent with a small number of high-affinity leaks across
true instance interfaces. Ordinary voxel-balanced MSE can dilute such leaks
inside the large negative edge population.

## Controlled Delta

- Keep the complete R58 seed1789/PublicEM-R57 line: M01--07/M09 random
  sampling, augmentation, widened labels for the ordinary supervised target,
  `unetr_aniso_em`, three nearest-neighbor affinities, loss, optimizer, and
  12000 steps.
- Return an additional default-off copy of the original instance labels before
  boundary widening. Replay the identical geometric augmentation without
  consuming additional training RNG. Use this tensor only to identify true
  adjacent nonzero instance pairs.
- For each crop, collect predicted nearest-neighbor affinities across every
  true adjacent instance pair. For each pair, take its highest 10% affinities
  and penalize squared excess above `.50`. Require at least four interface
  edges and retain at most the 64 largest interfaces.
- Average within each instance pair, then across pairs, then across crops. This
  gives a simple crop's one or two critical boundaries the same crop-level
  scale as a complex crop with many boundary voxels.
- Add the parameter-free loss at weight `.05`. Add no output channel, model
  parameter, crop selection, teacher, EMA, consistency view, DPP, LSD, or
  inference-time operation.

## Distinction From Rejected Families

R61/R62 changed which crops were sampled. R67 globally changed all negative
voxel weights. R68 predicted fixed two-voxel affinities. R79 routed whole
crops between teachers. R80 globally organized decoder features by instance
identity. R81 changes none of those: it applies a supervised high-tail
cannot-link guard only at the spatial interface of each real adjacent instance
pair, normalized per pair rather than per voxel.

## Scale Audit

Eight deterministic real training crops, one from each allowed block, loaded
all 214 PublicEM-R57 keys. At `max_affinity=.50`, raw boundary-leak loss ranged
from `0` for a one-instance crop to `.149964`; its mean was about `.067`. Weight
`.05` therefore contributes about `.0034`, approximately `.8%` of the mean
ordinary R58 loss in this audit. A three-instance crop contributed about
`.0020`; the single-instance crop correctly contributed zero.

## Gate

Run exactly one 8-GPU Shanghai crop-screen wave. Require exactly 80 finite
M08/M10 WaterZ rows and an empty failure ledger. Pass only if mean ARAND, mean
VOI, M08 ARAND, and M08 VOI all do not regress versus R58. A crop failure
forbids full-block work and all access to M11/M12. Do not sweep loss weight,
tail fraction, maximum affinity, minimum edges, or pair cap after a failure.

## Outcome

SiFlow UUID `76ac0302-57de-4868-abb7-25315cf1c41d` completed 12000 steps and
returned exactly 80 finite M08/M10 WaterZ rows with no failures. The paired
analyzer found zero strict R58 gate passes. Best balance has mean ARAND/VOI
`0.028377941932/0.352231169545`, M08
`0.032977813947/0.375969227906`, and M10
`0.023778069917/0.328493111184`. Reject R81 and do not run full-block or
M11/M12. Do not sweep boundary-leak hyperparameters. Full evidence is in
`outputs/mec8nm_r81_boundary_leak_crop_report_20260714.md`.
