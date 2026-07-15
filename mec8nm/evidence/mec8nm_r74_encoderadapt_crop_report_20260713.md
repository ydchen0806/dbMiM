# MEC8nm R74 Encoder-Only Domain-Adaptation Crop Report

Canonical versionable report:

`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_r74_encoderadapt_crop_report_20260713.md`

R74 tested whether MEC raw-domain adaptation remained useful when it could not
overwrite the PublicEM-R57 convolutional decoder or affinity head. A standard
DBMIM masked autoencoder loaded exactly the 77 shared encoder tensors and ran
4000 label-free steps on raw M01--M07/M09 crops. The audited hybrid exporter
then replaced exactly those 77 encoder tensors in R57 while preserving all 137
decoder/head tensors bit-identically. The hybrid SHA256 was
`9195891cabe131f555575c3231c7e02f481d5f984daca832cb7f40a88315fca0`.
The supervised stage loaded all 214 hybrid tensors and otherwise matched R58
for 12000 steps.

SiFlow UUID `9d0d3cb6-113a-46da-a3d6-81b1c7d8fa78`
(`mec8nm-r74-encoderadapt-4741d965`) succeeded in 26m45s on one 8-GPU
Shanghai `med-model` pod. The remote assertion and independent TOS fetch
confirm exactly 80 finite WaterZ rows, 40 each for M08 and M10, and an empty
failure ledger. M11/M12 were absent from the download and evaluation paths.

The paired analyzer found zero strict R58 gate passes. R74's only Pareto and
best-balance point is zero bias, temperature `1.15`, threshold `.50`: mean
ARAND/VOI `0.035560839373/0.359082802791`; M08
`0.037953939399/0.356534733043`; M10
`0.033167739347/0.361630872539`. Relative to R58, mean ARAND/VOI regress
`57.7257%/15.3926%`, while M08 regresses `153.9024%/39.2018%`.

The best M08-specific setting, bias `(-.20,-.45,-.45)`, temperature `1.15`,
threshold `.50`, improves M08 ARAND/VOI by `1.4179%/6.5262%`, but the same
setting makes M10 `0.066314355406/0.510689381445`, regressions of
`119.9936%/39.4416%`. This block-specific tradeoff is not a valid promotion.

Reject R74 before full-block validation, M11/M12, deployment, or 2PB
inference. Do not sweep raw-adaptation steps or learning rate around this
checkpoint family. R73 and R74 together show that short MEC masked
reconstruction, whether it overwrites the whole segmentation network or only
the encoder, does not preserve the R58 cross-block operating point.
