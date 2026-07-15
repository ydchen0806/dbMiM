# MEC8nm R78 Real Paired-Overlap Consistency Plan

## Hypothesis

R58 may be sensitive to real sliding-window context even when the same voxel is
visible in two windows. R75 could not test this because it fabricated context
with reflected padding. R78 reads two real, coordinate-linked crops from the
same supervised MEC HDF5 block and matches affinities only in their physical
overlap.

## Controlled Delta

- Keep the R58 seed1789, PublicEM-R57 initialization, M01--07/M09 supervision,
  model, main supervised loss, optimizer, augmentation ranges, and 12000-step
  schedule unchanged.
- Draw the second crop at a signed `8x32x32` offset from the primary crop.
  This yields a `24x128x128` overlap, equal to the R75 comparison core.
- Apply one shared rotation/flip and one shared intensity transform to both
  crops. Transform the signed offset under the same geometry.
- Keep each crop's historical percentile normalization independent so the
  loss tests the normalization/context instability seen at inference.
- Use the R72/R75 comparison weight `.05`. Stop gradients through the primary
  supervised prediction and update the paired view only where both affinity
  targets are valid and exactly agree after coordinate alignment.

## Gate

Run exactly one 8-GPU Shanghai crop-screen wave. Require exactly 80 finite
WaterZ rows, M08/M10 only, and an empty failure ledger. Pass only if mean
ARAND, mean VOI, M08 ARAND, and M08 VOI all do not regress versus R58. A crop
failure forbids full-block work and any access to M11/M12. Do not sweep shift
or weight after a failure.

## Outcome

Launcher-only UUID `cf5a12ba-a390-4f0e-9082-3743dcb5b812` failed before data
or training on a readonly-variable assignment. Repaired UUID
`5b34b36a-1b10-416f-a744-76eed2a1874f` succeeded with 12000 training steps and
exactly 80 valid M08/M10 WaterZ rows. Best mean ARAND/VOI improved to
`0.021307430975/0.254415142584`, but M08 regressed to
`0.037685313511/0.346308913501`; zero strict gate passes exist. Reject R78 and
do not run full-block, frozen splits, or shift/weight follow-ups.
