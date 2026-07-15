# MEC8nm R79 Topology-Routed Dual-Teacher Distillation Plan

## Hypothesis

R71 EMA is the strongest complex-block teacher observed so far, especially on
M10, while R58 remains the only accepted anchor because it protects M08. A
single checkpoint may learn both behaviors if the training label topology
chooses the teacher target per crop: simple, low-boundary crops route toward
R58 and complex crops route toward R71 EMA.

## Controlled Delta

- Keep the complete R58 training line fixed: seed 1789, PublicEM-R57 student
  initialization, M01--07/M09 data, crop sampling, augmentation, model,
  supervised loss, optimizer, and 12000-step schedule.
- Load frozen R58 (`4715c667...`) as the simple teacher and frozen R71 EMA
  (`9c82d7ac...`) as the complex teacher. Both have exactly 214 compatible
  `unetr_aniso_em` keys. Teacher construction is RNG-isolated.
- Compute foreground fraction, boundary fraction, and nonzero instance count
  from the raw training label crop before boundary widening and augmentation.
- Use the soft simple-teacher route
  `sigmoid((fg-.95)/.02) * sigmoid((.05-boundary)/.02) *
  sigmoid((5.5-unique)/2)`. Distill affinity probabilities to the routed
  teacher blend with fixed weight `.10` and stopped teacher gradients.
- Do not change sampling or reweight the main supervised loss. Do not combine
  R79 with R72 augmentation, R75 halo, or R78 paired-overlap consistency.
  Final inference uses only the single student network.

## Training-Only Topology Audit

A deterministic audit of 1024 ordinary R58 training crops (128 per training
block) found mean foreground/boundary/unique values
`0.8826566/0.2322310/22.8467`. The proposed soft route mean was `0.110561`,
with 139/1024 crops at route >= .5. Hard-simple crops were concentrated in
M01 (15), M07 (69), and M09 (69), with one in M03. This uses training labels
only and does not access M08/M10 labels for routing or selection.

## Gate

Run exactly one 8-GPU Shanghai crop-screen wave. Require exactly 80 finite
WaterZ rows, M08/M10 only, and an empty failure ledger. Pass only if mean
ARAND, mean VOI, M08 ARAND, and M08 VOI all do not regress versus R58
(`0.022546003667/0.311183450751` mean and
`0.014948239287/0.256127966754` M08). A crop failure forbids full-block work
and all access to M11/M12. Do not sweep the route thresholds or weight after a
failure.

## Outcome

SiFlow UUID `4c50adc6-ee6c-4ccf-bd28-e2341af5e770`
(`mec8nm-r79-topology-teacher-1a465cfa`) succeeded on one 8-GPU Shanghai
`med-model` pod. Training reached step 12000 in 1755 seconds; the remote gate
then asserted exactly 80 finite M08/M10 WaterZ rows and an empty failure
ledger. The paired analyzer found zero strict R58 gate passes.

Best balance was zero bias, temperature `1.00`, threshold `.50`: mean
ARAND/VOI `0.062021381189/0.412646966283`, M08
`0.035591466186/0.337509064870`, and M10
`0.088451296192/0.487784867696`. Mean metrics regressed
`175.0881%/32.6057%`, while M08 regressed `138.0980%/31.7736%`. Reject R79
before full-block and frozen splits. Do not sweep the distillation weight,
route thresholds, or teacher pair.
