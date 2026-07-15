# MEC8nm R80 Decoder-Feature Instance Embedding Plan

## Hypothesis

R58's local affinity supervision does not explicitly organize the shared
decoder representation by instance identity. A weak discriminative embedding
loss at the final decoder feature map may preserve long interior continuity
while separating adjacent instances before the unchanged affinity head. This
is a representation-level topology constraint rather than another crop-level
teacher blend or long-range affinity output.

## Controlled Delta

- Keep R58 seed 1789, PublicEM-R57 initialization, M01--07/M09 sampling and
  augmentation, `unetr_aniso_em`, three-channel affinity head, supervised loss,
  optimizer, and 12000-step schedule unchanged.
- Optionally return the existing 32-channel full-resolution feature map just
  before the affinity head. This changes no parameter or checkpoint key and
  default inference logits remain bit-identical.
- Deterministically sample the feature map at z/y/x stride `2/4/4`. Ignore
  label zero and instances with fewer than eight sampled voxels; cap at 64
  largest instances per crop.
- L2-normalize features. Penalize within-instance feature distance beyond
  `.20` and centroid distances below `1.0`, with equal component weights.
  Add the resulting loss at fixed weight `.02`.
- Do not add an embedding head, teacher, affinity channel, crop selection, DPP,
  LSD, EMA, or consistency loss. Final inference is exactly the ordinary
  three-affinity model.

## Distinction From Rejected Families

R59 used a learnable affinity postprocessor plus two-hop probability products.
R68 added three explicit two-voxel affinity outputs. R79 blended R58/R71
teacher probabilities using one route per crop. R80 instead supervises the
existing decoder representation with whole-instance prototypes; it neither
predicts a fixed-offset target nor mixes checkpoints/logits/teachers.

## Scale Audit

One real R57-initialized training crop produced 30 eligible instances,
sampled foreground fraction `.881914`, variance/distance losses
`.247217/.435592`, and total embedding loss `.682809`. Weight `.02` contributes
`.013656`, meaningful but below the typical `.1--.2` supervised main loss.

## Gate

Run exactly one 8-GPU Shanghai crop-screen wave. Require exactly 80 finite
M08/M10 WaterZ rows and an empty failure ledger. Pass only if mean ARAND, mean
VOI, M08 ARAND, and M08 VOI all do not regress versus R58. A crop failure
forbids full-block work and all access to M11/M12. Do not sweep embedding
weight, feature stride, or margins after a failure.

## Outcome

SiFlow UUID `c4ced6cd-7387-4d3f-baf4-7ca2bf3504aa` completed 12000 steps and
returned exactly 80 finite M08/M10 WaterZ rows with no failures. The paired
analyzer found zero strict R58 gate passes. Best balance improves mean
ARAND/VOI to `0.018971582484/0.264526010140`, but M08 regresses to
`0.033522118658/0.371451721207`; M10 is
`0.004421046309/0.157600299074`. Reject R80 and do not run full-block or
M11/M12. Do not sweep the embedding weight, stride, margins, instance filters,
or normalization. Full evidence is in
`outputs/mec8nm_r80_instance_embedding_crop_report_20260714.md`.
