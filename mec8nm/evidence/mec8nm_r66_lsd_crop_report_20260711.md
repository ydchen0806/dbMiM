# MEC8nm R66 LSD Crop Gate

## Scope

- Run: `finetune_mec8nm_r66_seed777_lsd_stage1`.
- SiFlow UUID: `4f60e8f4-eda4-45b0-a21d-81fbd981ff68`.
- Training blocks: MEC01--07 and MEC09 only.
- Validation blocks: MEC08 and MEC10 only.
- Evaluation: volume-normalized `64x512x512` crops, WaterZ only, eight
  thresholds, two fixed bias triplets, two temperatures, and no test path.
- Runtime assertion: 80 raw WaterZ rows, exactly MEC08/MEC10, finite ARAND and
  VOI, and an empty evaluator failure list.

## Controlled Architecture Change

R66 is the R57 training regime with a seven-channel output head. Channels
0--2 remain the affinity logits consumed by WaterZ. Channels 3--6 are an
auxiliary local-shape-descriptor target: one foreground logit and three
foreground-masked centroid-offset regressions. The auxiliary loss uses
`lsd_weight=0.15`, `lsd_fg_weight=0.25`, and `lsd_offset_weight=1.0`.

When loading the three-channel R57 initialization, the first three values of
`head.channel_bias` and `head.channel_scale` are copied exactly into the larger
head; the new descriptor branch stays independently initialized. Static tests
confirmed the transfer, real MEC HDF5 target/loss construction, and the staged
bundle's M08/M10-only entrypoint before submission.

## Runtime Evidence

The remote task succeeded after 5000 steps. Startup proved CUDA with eight
GPUs, WaterZ availability, `loaded_pretrained_keys=214`, the fixed training
set, and the declared LSD loss configuration. The task emitted the strict
80-row metric assertion and then completed successfully.

Downloaded result evidence:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r66_lsd_20260711/eval_val_crop64x512/eval_val_crop64x512/eval_finetune_mec8nm_r66_seed777_lsd_stage1_val_crop64x512/`

The CSV has 80 raw rows. Removing the 16 exact `pred` / zero-bias,
temperature-1.0 duplicates leaves 64 rows, or 32 paired M08/M10
configurations. Every retained row is `backend=waterz` and the summary failure
list is empty.

## Selection Result

The R58 paired-crop anchor is zero bias, temperature `1.15`, threshold `.50`:

- mean ARAND/VOI: `0.022546003667/0.311183450751`;
- balance: `0.083761227436`;
- MEC08: `0.014948239287/0.256127966754`, `n_pred=242`;
- MEC10: `0.030143768046/0.366238934749`, `n_pred=120`.

R66 has two non-dominated paired crop configurations. Its best balance is bias
`(-.20,-.45,-.45)`, temperature `1.15`, threshold `.22`:

- mean ARAND/VOI: `0.049656091996/0.411782761303`;
- balance: `0.142994834444`, worst-block balance `0.173345275549`;
- MEC08: `0.032558920140/0.373421472399`, `n_pred=809`;
- MEC10: `0.066753263853/0.450144050207`, `n_pred=193`.

Relative to R58, this point is `120.2434%` worse in mean ARAND, `32.3280%`
worse in mean VOI, and `70.7172%` worse in balance. Its MEC08 ARAND and VOI
are `117.8111%` and `45.7949%` worse. The alternate zero-bias Pareto point is
also worse on both aggregate metrics. No R66 configuration satisfies the
strict crop gate of non-regression on aggregate ARAND/VOI and MEC08 ARAND/VOI.

## Decision

Reject R66 before full-block validation. Do not fetch or evaluate its
checkpoint further; do not use MEC11/MEC12, deployment, or 2PB inference. The
seven-channel auxiliary supervision and calibrated head-expansion transfer are
mechanically validated, but this loss formulation does not improve the current
MEC crop target.
