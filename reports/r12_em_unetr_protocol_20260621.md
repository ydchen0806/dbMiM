# R12 EM-UNETR dbMiM Protocol

Date: 2026-06-21

## Metric interpretation fix

The very low VOI values around 0.18-0.21 are not CREMI whole-protocol
headline numbers. They came from official waterz evaluation on sample A only
with post-processing calibration sweep. The sample A crop was not small
(`--crop-size 0 0 0` means the full sample), but selecting the best
post-processing point on a single volume is optimistic and not comparable with
the usual A/B/C full-volume CREMI numbers.

Existing full A/B/C summaries in this repo are in the expected range:

| summary | best VOI | ARAND at best VOI | sample count |
|---|---:|---:|---:|
| `outputs/tos_fetch/official_scratch_r5_abc/cremi_segmentation_summary.json` | 1.121139 | 0.287917 | 3 |
| `outputs/tos_fetch/official_pretrained_r5_abc/cremi_segmentation_summary.json` | 1.250565 | 0.347600 | 3 |

R12 therefore uses A/B/C full-volume aggregation for the main comparison. The
summary selector first averages each exact post-processing parameter setting
across A/B/C, then picks best VOI or best ARAND from those aggregate rows.
`scripts/poll_dbmim_tos_results.py` was updated to use the same aggregation
when reconstructing summaries from SiFlow stdout fallback logs.

## R12 model change

R12 keeps the dbMiM/UNETR pretraining interface intact:

- shared encoder keys: `patch_embed`, `pos_embed`, `encoder_blocks`, `norm`
- same ViT token grid and pretrained checkpoint loading path
- same fine-tuning framework and affinity target generation

The EM-specific change is limited to the segmentation decoder output head:

- `UNETREMAffinityNet` subclasses the migrated anisotropic UNETR backbone
- `decode_features()` exposes the original anisotropic decoder feature map
- `EMAffinityHead3D` adds XY-heavy residual membrane refinement
- z affinity and xy affinities have separate 1x1 prediction heads
- a learnable per-channel logit scale/bias calibrates z/y/x affinities
- z-channel bias is initialized lower (`[-0.2, 0.0, 0.0]`) because anisotropic
  EM stacks should be conservative when linking across sections

This is meant to test whether dbMiM pretraining gives stable gains when the
fine-tuning architecture better matches EM affinity prediction.

## Training pair

Both arms use the same model, data, augmentations, optimizer, loss, and
post-processing. The only intended difference is pretrained encoder
initialization.

| arm | config | checkpoint source |
|---|---|---|
| pretrained | `configs/finetune_cremi_real_unetr_aniso_em_bce_encoderlr_allpretrained_r12.yaml` | `outputs/pretrain_cremi_real_all_dbmim_r6/pretrained_latest.pt` |
| scratch | `configs/finetune_cremi_real_unetr_aniso_em_bce_encoderlr_scratch_r12.yaml` | none |

Key settings:

- architecture: `unetr_aniso_em`
- crop: `32 x 160 x 160`
- max steps: `40000`
- save every: `2000`
- loss: BCE + Dice + boundary Dice
- channel weights: `[1.35, 1.0, 1.0]`
- encoder LR: `1e-5`
- decoder/head LR: `8e-5`
- augmentations: XY rotation, gamma, Gaussian noise, widened boundary

## Main evaluation

The post-train evaluation attached to each R12 finetune task uses:

- full CREMI A/B/C: `--max-samples 0`
- full volume per sample: `--crop-size 0 0 0`
- stride: `16 80 80`
- backend: `waterz`
- waterz scoring: `hist_quantile`
- thresholds: `0.05 0.10 0.20 0.30 0.50`
- calibration biases: `(0,0,0)`, `(-0.25,-0.5,-0.5)`, `(-0.5,-1.0,-1.0)`
- CREMI boundary ignore: xy `1`, z `0`
- metrics: `VOI split/merge/sum` and `adapted_rand_error`

This remains an aggregate post-processing sweep, not a locked test parameter.
If R12 gives a useful gain, the next confirmatory run should lock one common
post-processing setting chosen from the aggregate sweep and report scratch vs
pretrained with that fixed setting.

## Expected outputs

Pretrained arm:

- train output: `outputs/finetune_cremi_real_unetr_aniso_em_bce_encoderlr_allpretrained_r12/`
- eval output: `outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_allpretrained_r12/`

Scratch arm:

- train output: `outputs/finetune_cremi_real_unetr_aniso_em_bce_encoderlr_scratch_r12/`
- eval output: `outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_scratch_r12/`
