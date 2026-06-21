# R13 BCAR Method Plan

Date: 2026-06-21

## Motivation

R12 fixed the CREMI evaluation protocol and confirmed that sample-A-only VOI
was not comparable with A/B/C full-volume VOI. The EM-specific UNETR head gave
stronger absolute A/B/C numbers, but dbMiM pretraining did not produce a clean
gain under BCE-style affinity supervision:

| arm | best VOI | ARAND at best VOI | best ARAND |
|---|---:|---:|---:|
| `unetr_aniso_em` pretrained | 0.830019 | 0.172866 | 0.172465 |
| `unetr_aniso_em` scratch | 0.806745 | 0.179934 | 0.175631 |

This suggests that voxel-wise BCE is not enough to expose a stable pretraining
benefit. Waterz depends on merge-score ordering and z/xy calibration, not only
on independent voxel probabilities.

## Proposed paper method

Boundary-Calibrated Affinity Ranking (BCAR) adds a small agglomeration-aligned
regularizer during dbMiM fine-tuning:

1. Positive affinities should rank above boundary affinities with a margin.
2. z-affinity logits should be calibrated separately from xy-affinity logits.
3. The regularizer is lightweight and does not require differentiating through
   waterz.

For logits `a` and affinity target `y`, BCAR samples valid positive edges
`y >= tau_pos` and boundary edges `y <= tau_neg` per channel:

```text
L_rank = mean_c softplus(margin - mean(a_pos^c) + mean(a_neg^c))
L_calib = (mean(a_z) - mean(a_xy) + gap)^2
L = L_affinity + lambda_rank L_rank + lambda_calib L_calib
```

The default R13 setting uses:

- `lambda_rank = 0.05`
- `lambda_calib = 0.005`
- `margin = 1.0`
- `tau_pos = 0.75`
- `tau_neg = 0.25`
- `gap = -0.2`

## Hypothesis

BCAR should help dbMiM-pretrained UNETR preserve boundary ordering under
fine-tuning, making the pretrained arm improve both VOI and ARAND under a
shared A/B/C waterz protocol. If only scratch improves, then BCAR is a useful
segmentation loss but not evidence for pretraining. If only pretrained improves,
the method becomes a good paper claim: pretraining supplies EM structure priors,
and BCAR turns them into agglomeration-stable affinity ordering.

## R13 experiment matrix

All R13 arms use:

- architecture: `unetr_aniso_em`
- pretraining source for pretrained arms:
  `outputs/pretrain_cremi_real_all_dbmim_r6/pretrained_latest.pt`
- crop: `32 x 160 x 160`
- max steps: `40000`
- optimizer split: encoder LR `1e-5`, decoder/head LR `8e-5`
- evaluation: CREMI A/B/C full-volume official waterz aggregate sweep

| run | pretraining | loss | BCAR |
|---|---|---|---|
| `em-shwmse-allpretrained-r13` | yes | SuperHuman weighted MSE | no |
| `em-shwmse-scratch-r13` | no | SuperHuman weighted MSE | no |
| `em-shwmse-bcar-allpretrained-r13` | yes | SuperHuman weighted MSE | yes |
| `em-shwmse-bcar-scratch-r13` | no | SuperHuman weighted MSE | yes |

Primary comparisons:

- `em-shwmse-allpretrained-r13` vs `em-shwmse-scratch-r13`: does MSE reveal
  a pretraining gain without BCAR?
- `em-shwmse-bcar-allpretrained-r13` vs `em-shwmse-bcar-scratch-r13`: does
  BCAR make the pretraining gain stable?
- `em-shwmse-bcar-allpretrained-r13` vs `em-shwmse-allpretrained-r13`: does
  BCAR improve the pretrained model beyond loss-only alignment?
