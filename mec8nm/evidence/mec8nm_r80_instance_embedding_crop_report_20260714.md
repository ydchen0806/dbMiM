# MEC8nm R80 Decoder-Feature Instance Embedding Crop Report

R80 tested a representation-level instance objective while preserving the
complete R58 affinity-training line. The student used the exact PublicEM-R57
initializer, seed 1789, M01--07/M09 sampling and augmentation,
`unetr_aniso_em`, three output affinities, supervised objective, optimizer,
and 12000-step schedule. The only training delta was a weight-`.02`
discriminative loss on the existing 32-channel final decoder feature map.

Features were L2-normalized and deterministically sampled at z/y/x stride
`2/4/4`. Background and instances with fewer than eight sampled voxels were
ignored, with at most 64 largest instances per crop. The loss equally combined
a within-instance variance penalty beyond `.20` and an instance-centroid
distance penalty below `1.0`. It added no parameters, checkpoint keys, teacher,
affinity channel, or inference-time operation; default logits remained
bit-identical in the unit test.

Six dedicated tests and all 37 MEC tests passed. A one-step CPU CLI smoke test
also exercised the configured loss and checkpoint path. Before submission, a
real R57-initialized crop produced 30 eligible instances and embedding loss
`.682809`; at weight `.02`, its `.013656` contribution was below the ordinary
supervised loss scale.

SiFlow UUID `c4ced6cd-7387-4d3f-baf4-7ca2bf3504aa`
(`mec8nm-r80-instance-embedding-9edb4c3d`) ran as the only MEC wave on one
8-GPU Shanghai `med-model` pod. Startup verified CUDA 8, WaterZ, all 214 R57
keys, dataset size 32768, the eight allowed training paths, and the exact
embedding configuration. Training reached step 12000 in 1344 logged seconds
with no runtime errors. The final embedding loss was `.161571`, split into
variance/distance `.070016/.091554`; final inference used the unchanged
three-affinity network.

The remote assertion and independent TOS fetch contain exactly 80 finite
WaterZ rows: 40 each for M08/M10, five recorded affinity variants, eight
thresholds, and an empty failure list. The paired analyzer finds zero strict
R58 gate passes. Best balance is bias `(-.20,-.45,-.45)`, temperature `1.15`,
threshold `.50`: mean ARAND/VOI `0.018971582484/0.264526010140`; M08
`0.033522118658/0.371451721207`; M10
`0.004421046309/0.157600299074`. Relative to R58, the means improve
`15.8539%/14.9935%` and M10 improves `85.3335%/56.9679%`, but M08 regresses
`124.2546%/45.0258%`.

At R58's exact zero-bias, temperature-`1.15`, threshold-`.50` operating point,
R80 is worse on both blocks: mean ARAND/VOI
`0.062431049101/0.418866062355`, M08
`0.035969733580/0.348657906045`, and M10
`0.088892364622/0.489074218665`.

Reject R80 before full-block validation, M11/M12, deployment, or 2PB
inference. Do not sweep embedding weight, feature stride, margins, minimum
instance size, instance cap, or normalization. The aggregate/M10 improvement
is a mechanism signal, but it repeats the R71/R78 failure pattern and cannot
override M08 preservation. R58 remains the crop anchor and R57 remains the
M08/M10 full-block anchor.

Reproducibility:

- task duration: `28m33s`
- TOS prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r80_seed1789_instance_embedding_stage1`
- final checkpoint SHA256:
  `518cad1208486b40ff408a3860a88b6c31e46251d0d21237a198c348e88b0017`
- metrics CSV SHA256:
  `f2d333c0a86c54a85949cdc727a55fc4028bf3b3d8eac2e33bf2e45c60edb91d`
- summary JSON SHA256:
  `010e8f2492197d4db6044466e893649fbeb17babd98fd2d3975d5c88c50df982`
- paired analysis SHA256:
  `cef0206fe6f26b3a3fb30a8be44e5346c474752b1708da8e70ef7585125827ee`
