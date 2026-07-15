# MEC8nm R81 Instance-Pair Boundary-Leak Crop Report

R81 tested whether sparse high-affinity leaks across true instance interfaces
explain the recurring M08 merge failure. It preserved the complete R58 line:
PublicEM-R57 initialization, seed 1789, random M01--07/M09 sampling,
augmentation, widened-label supervised targets, `unetr_aniso_em`, three
nearest-neighbor affinities, the original loss/optimizer, and 12000 steps.

The only training delta was a parameter-free instance-pair boundary-leak loss.
The dataset returned an additional original label tensor before boundary
widening and replayed the identical geometry without changing the historical
RNG stream. For each true adjacent nonzero instance pair, R81 selected the
highest 10% predicted interface affinities and penalized squared excess above
`.50`. Interfaces needed four edges; at most 64 largest interfaces were used.
Losses were averaged per pair and per crop, then added at weight `.05`.

This differs from R61/R62 crop sampling, R67 global negative weighting, R68
long-range affinity outputs, R79 teacher routing, and R80 global feature
embedding. R81 added no model parameter, output, teacher, EMA, consistency
view, sampling change, or inference operation.

Eight focused tests and all 45 MEC tests passed. Tests verified real-interface
selection, gradients, threshold inactivity, equal pair rather than edge
weighting, aligned raw/widened augmentation, and an unchanged default RNG
stream. A one-step CPU CLI smoke run exercised the full DataLoader, loss,
backward, and checkpoint path. An eight-block real R57 audit measured mean raw
leak loss about `.067`; weight `.05` contributed about `.0034`, roughly `.8%`
of the audited ordinary loss.

SiFlow UUID `76ac0302-57de-4868-abb7-25315cf1c41d`
(`mec8nm-r81-boundary-leak-295cbbc2`) ran as the only MEC wave on one 8-GPU
Shanghai `med-model` pod. Startup verified the static R81 contract, CUDA 8,
WaterZ, 214/214 R57 keys, dataset size 32768, world size 8, and the exact loss
configuration. Training reached step 12000 in 1559 logged seconds with no
runtime errors. The final main/leak losses were `.052955/.007593`.

The remote assertion and independent TOS fetch contain exactly 80 finite
WaterZ rows, 40 each for M08/M10, and an empty failure list. The paired
analyzer finds zero strict R58 gate passes. Best balance is bias
`(-.20,-.45,-.45)`, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.028377941932/0.352231169545`; M08
`0.032977813947/0.375969227906`; M10
`0.023778069917/0.328493111184`. Relative to R58, mean metrics regress
`25.8668%/13.1908%` and M08 regresses `120.6134%/46.7896%`, although M10
improves `21.1178%/10.3063%`.

At R58's exact zero-bias, temperature-`1.15`, threshold-`.50` point, R81 is
worse on both blocks: mean ARAND/VOI `0.049605838090/0.432148409087`, M08
`0.033004519579/0.377507958388`, and M10
`0.066207156601/0.486788859786`.

Reject R81 before full-block validation, M11/M12, deployment, or 2PB
inference. Do not sweep leak weight, tail fraction, maximum affinity,
minimum-edge count, or pair cap. M08 predicted instances remain low at 202 and
VOI merge remains high at `.207795`, versus R58's 242 and `.081985`.
Penalizing raw voxel-adjacent cannot-link leaks does not control WaterZ's
region-level merge ordering. R58 remains the crop anchor and R57 remains the
M08/M10 full-block anchor.

Reproducibility:

- task duration: `32m15s`
- TOS prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r81_seed1789_boundary_leak_stage1`
- final checkpoint SHA256:
  `b6c020a515d1eb9f93b064b296fb447997984ca8da712365bf84f1c9b14903f6`
- metrics CSV SHA256:
  `ff6343cba44626e28824ab10b9aa592b4b634891761566a82fe894d2f942fa71`
- summary JSON SHA256:
  `38eac1f2ab71d57c7adb3dc0b852039f5b902f5f38fd923150e0383a62a63626`
- paired analysis SHA256:
  `136f9aa727b29c424be79983b408f3c5fff249dda5b5aa009df19fa50da4287e`
