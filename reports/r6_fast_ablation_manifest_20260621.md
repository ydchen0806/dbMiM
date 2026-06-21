# R6 Fast Ablation Manifest

Date: 2026-06-21

Purpose: quickly test whether MSE-style affinity objectives and pretrained
encoder preservation can improve CREMI neuron segmentation. All finetune jobs
use one `sci.g21-3` GPU on `cpt-train`, train for 12k steps, then run official
CREMI-style full-volume sample-A waterz evaluation in the same pod
(`xy=1,z=0` boundary-ignore mask).

## Fast Finetune Jobs

| arm | UUID | config | output prefix | expected eval prefix |
|---|---|---|---|---|
| MSE pretrained | `8e55bbbd-1656-4437-a609-9c1ab12dd831` | `configs/finetune_cremi_real_unetr_aniso_superhuman_mse_pretrained_r6.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_mse_pretrained_r6/` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_mse_pretrained_r6/` |
| MSE scratch | `7e324047-325c-45e0-8482-f38651c682d1` | `configs/finetune_cremi_real_unetr_aniso_superhuman_mse_scratch_r6.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_mse_scratch_r6/` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_mse_scratch_r6/` |
| BCE+MSE hybrid pretrained | `a5b45b94-c98f-43d6-aff5-d1d8bb1337c4` | `configs/finetune_cremi_real_unetr_aniso_superhuman_hybrid_pretrained_r6.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_hybrid_pretrained_r6/` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_hybrid_pretrained_r6/` |
| BCE+MSE hybrid scratch | `aa80a2f1-26fc-4c97-9ff2-4b740d0407a3` | `configs/finetune_cremi_real_unetr_aniso_superhuman_hybrid_scratch_r6.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_hybrid_scratch_r6/` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_hybrid_scratch_r6/` |
| BCE freeze-encoder pretrained | `a4725c28-d0f7-4716-b127-ff2f6f30f1ee` | `configs/finetune_cremi_real_unetr_aniso_superhuman_bce_freezeenc_pretrained_r6.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_bce_freezeenc_pretrained_r6/` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_bce_freezeenc_pretrained_r6/` |
| BCE scratch seed2 | `d8c57a00-3c0e-4da8-ba9a-95e971877c22` | `configs/finetune_cremi_real_unetr_aniso_superhuman_bce_scratch_seed2_r6.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_bce_scratch_seed2_r6/` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_bce_scratch_seed2_r6/` |

## Long Pretrain Job

| arm | UUID | config | output prefix |
|---|---|---|---|
| CREMI A/B/C all-volume dbMiM pretrain R6 | `dd6c06d0-f077-42ab-b26b-eedd51cfb4a5` | `configs/pretrain_cremi_real_all_r6.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_all_dbmim_r6/` |

This job completed the planned 160k steps by 2026-06-21 03:44 UTC. The final
pretrain checkpoint is:

`tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_all_dbmim_r6/pretrained_latest.pt`

## R7 All-Pretrain Fast Jobs

R7 uses the live `pretrain_cremi_real_all_dbmim_r6/pretrained_latest.pt`
checkpoint to test whether the longer CREMI A/B/C dbMiM pretrain changes the
finetuning outcome. These are also 1-GPU, 12k-step jobs with same-pod official
sample-A waterz evaluation.

| arm | UUID | config | expected eval prefix |
|---|---|---|---|
| BCE all-pretrained | `6c4c754a-f09c-48fb-9b7b-faee47d8ea8f` | `configs/finetune_cremi_real_unetr_aniso_superhuman_bce_allpretrained_r7.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_bce_allpretrained_r7/` |
| MSE all-pretrained | `d2495633-1914-4df7-9897-c1b6a147a91c` | `configs/finetune_cremi_real_unetr_aniso_superhuman_mse_allpretrained_r7.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_mse_allpretrained_r7/` |
| weighted-MSE all-pretrained | `36a4991f-7f34-4e45-a993-7e0e338732b2` | `configs/finetune_cremi_real_unetr_aniso_superhuman_weightedmse_allpretrained_r7.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_weightedmse_allpretrained_r7/` |
| weighted-MSE scratch | `20940d64-f5fd-442b-99f2-33d5a8283ebc` | `configs/finetune_cremi_real_unetr_aniso_superhuman_weightedmse_scratch_r7.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_weightedmse_scratch_r7/` |

## External EM Pretrain Data

The current local dataset inventory contains CREMI samples A/B/C. The old
`configs/pretrain_fafb.yaml` points to `data/EM_pretrain_data/FAFB_hdf`, but
that directory is not present locally or in verified TOS assets.

`scripts/prepare_em_pretrain_data.py` now prepares the gated Hugging Face
dataset `cyd0806/EM_pretrain_data` for offline SiFlow use. Manifest-only
inventory was generated under `data/EM_pretrain_data/*_manifest.json`:

| group | zip files | zip bytes |
|---|---:|---:|
| FAFB | 7 | 117.38 GB |
| FIB-25 | 10 | 114.12 GB |
| Kasthuri2015 | 9 | 109.36 GB |
| MitoEM | 2 | 16.66 GB |
| mb_moc | 7 | 128.32 GB |
| all groups | 35 | 485.84 GB |

The HF repo file list is readable, but the zip objects are gated and return
401 without authentication. No `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` or local
Hugging Face token file was present on this machine at the time of this
manifest. Once an authorized token is available, the intended command is:

```bash
HF_TOKEN=... python scripts/prepare_em_pretrain_data.py \
  --group all --download --extract --upload-tos
```

## Current Baselines

| arm | metric mask | best VOI | best ARAND |
|---|---|---:|---:|
| BCE scratch R5 | official sample A | 0.169502 | 0.024062 |
| BCE pretrained R5 | official sample A | 0.185597 | 0.032785 |
| scratch R5 | official A/B/C | 1.121139 | 0.287647 |
| pretrained R5 | official A/B/C | 1.250565 | 0.332747 |

Decision rule: if an R6 fast arm beats `VOI_sum=0.169502` or
`ARAND=0.024062` on official sample A, expand that arm to full A/B/C.

## R6 Official Sample-A Results

The R6 jobs completed by 2026-06-21 03:27 UTC. These are the official sample-A
full-volume waterz sweep summaries with CREMI-style boundary ignore
(`xy=1,z=0`):

| arm | best VOI | best ARAND | note |
|---|---:|---:|---|
| MSE pretrained | 1.732468 | 0.840354 | pure MSE is not competitive |
| MSE scratch | 2.589315 | 0.914007 | pure MSE is not competitive |
| BCE+MSE hybrid pretrained | 0.242892 | 0.040465 | usable but not best |
| BCE+MSE hybrid scratch | 0.236050 | 0.046480 | close VOI, worse ARAND |
| BCE freeze-encoder pretrained | 0.232240 | 0.032987 | best R6 arm, shows some pretrained signal |
| BCE scratch seed2 | 0.246463 | 0.037691 | strong scratch but below freeze-encoder pretrained |

Conclusion: pure MSE should not be expanded. BCE remains the strongest
supervised loss so far. The best R6 pretrained arm improves over the R6 scratch
seed2 control but still does not beat the earlier R5 BCE scratch baseline
(`VOI=0.169502`, `ARAND=0.024062`), so the next meaningful test is R7
all-pretrain latest/final checkpoint plus BCE.

## R7 Official Sample-A Results

R7 used a live `pretrain_cremi_real_all_dbmim_r6/pretrained_latest.pt` during
the long pretrain run. The checkpoint was not yet the final 160k-step weight
for all R7 jobs, so these results are mainly a fast direction check.

| arm | best VOI | best ARAND | note |
|---|---:|---:|---|
| BCE all-pretrained | 0.306617 | 0.074514 | unfreezing the live all-pretrain checkpoint was worse than R5/R6 |
| MSE all-pretrained | 2.415206 | 0.909257 | pure MSE remains unusable |
| weighted-MSE all-pretrained | 0.255522 | 0.051037 | better than pure MSE but below BCE baselines |
| weighted-MSE scratch | 0.279585 | 0.045492 | scratch control also below BCE |

Conclusion: direct all-pretrain + all-unfreeze was not enough. The useful R7
signal was negative: pure MSE is not a viable supervised replacement, and
weighted-MSE needs the exact SuperHuman normalization / weighting before being
judged fairly.

## R8 Final-All-Pretrain Transfer Results

R8 used the final 160k-step all-CREMI dbMiM checkpoint and tested whether
preserving the encoder helps. These were still 12k-step fast jobs with
same-pod official sample-A waterz evaluation.

| arm | UUID | best VOI | best ARAND | note |
|---|---|---:|---:|---|
| BCE freeze-encoder all-pretrained | `5a8c35ff-159a-4712-a4d1-03a551994d18` | 0.225764 | 0.032160 | better than R7, still below R5 BCE scratch |
| BCE encoder-LR all-pretrained | `aec1cf0d-27aa-4428-8490-8deb8a96dad0` | 0.204602 | 0.024906 | ARAND is close to the R5 scratch baseline; VOI still worse |

Conclusion: the final all-CREMI pretrain is much better than the live R7
checkpoint when the encoder is preserved. The encoder-LR arm is the best
pretrained-side signal so far, but it still does not beat the R5 BCE scratch
sample-A baseline (`VOI=0.169502`, `ARAND=0.024062`). This motivates a fair
30k-step R9 comparison rather than stopping at 12k.

## R9 30k Fair Sweep

R9 is a 30k-step one-GPU-per-arm sweep submitted on 2026-06-21 04:25-04:28 UTC
to `cn-shanghai/changliu`, resource pool `med-model`, instance `sci.g21-3`.
Every arm trains with the R5 synchronized augmentation / anisotropic UNETR
recipe, then runs the same official sample-A waterz evaluation in the same pod.

| arm | UUID | config | purpose |
|---|---|---|---|
| BCE all-pretrained | `99ab7d58-8886-430e-86fa-92c9d4a0fcae` | `configs/finetune_cremi_real_unetr_aniso_superhuman_bce_allpretrained_r9.yaml` | fair 30k final-pretrain vs R5 BCE scratch |
| BCE encoder-LR all-pretrained | `33638465-d404-4229-b150-cdbf401e8159` | `configs/finetune_cremi_real_unetr_aniso_superhuman_bce_encoderlr_allpretrained_r9.yaml` | expand the best R8 transfer signal |
| BCE freeze-encoder all-pretrained | `7ad21665-e4a3-4766-903e-d89088429919` | `configs/finetune_cremi_real_unetr_aniso_superhuman_bce_freezeenc_allpretrained_r9.yaml` | test stronger encoder preservation |
| BCE ignore-edge all-pretrained | `37fe0514-76a0-4d74-adb5-c51e4b1d7dcc` | `configs/finetune_cremi_real_unetr_aniso_superhuman_bce_ignore_allpretrained_r9.yaml` | test whether widened-border label 0 should be ignored in loss |
| BCE ignore-edge scratch | `885406a6-6572-46a0-8bbe-da607132b26a` | `configs/finetune_cremi_real_unetr_aniso_superhuman_bce_ignore_scratch_r9.yaml` | paired scratch control for ignore-edge loss |
| SuperHuman weighted-MSE all-pretrained | `e0a7b840-518e-4bca-8a75-37e33b8859b4` | `configs/finetune_cremi_real_unetr_aniso_superhuman_shwmse_allpretrained_r9.yaml` | exact SuperHuman weighted-MSE normalization |
| SuperHuman weighted-MSE scratch | `6ebba848-1ca9-4454-9af6-559d24d8a5fc` | `configs/finetune_cremi_real_unetr_aniso_superhuman_shwmse_scratch_r9.yaml` | paired scratch control for SH weighted-MSE |
| SuperHuman weighted-MSE ignore-edge all-pretrained | `7c7b6119-07e5-4fe3-98bd-bab8d1f728e9` | `configs/finetune_cremi_real_unetr_aniso_superhuman_shwmse_ignore_allpretrained_r9.yaml` | combine exact SH weighted-MSE with ignore-edge masking |

Implementation note: `train_finetune.py` now supports
`loss.loss_type: superhuman_weighted_mse` with SuperHuman-style
`sum(weight * error) / (B * D * H * W)` normalization. It also supports
`loss.ignore_label_edges: true`, which masks affinity edges touching
`ignore_label` before BCE/MSE/Dice terms. This is separate from the older
`weighted_mse` used in R5/R7, so old runs remain reproducible.

Expected duration: based on R5/R8 throughput, each 30k arm should need roughly
1-1.5 hours for training plus 10-20 minutes for same-pod waterz evaluation.
The eight tasks run concurrently if the pool keeps admitting 1-GPU jobs.

### R9 weighted-MSE bug and R10 resubmission

The three R9 `superhuman_weighted_mse` tasks exposed an AMP bug at step 20:
`train_main_loss` was exactly `0.0`. The cause was dtype overflow in the
SuperHuman normalization scalar: `B*D*H*W = 1,638,400` was constructed with
`logits.new_tensor(...)` under fp16 autocast, becoming `inf` and zeroing the
main weighted-MSE term. The BCE R9 jobs do not use this path and were kept.

The buggy SH weighted-MSE R9 tasks were stopped:

| arm | stopped UUID |
|---|---|
| SH weighted-MSE all-pretrained R9 | `e0a7b840-518e-4bca-8a75-37e33b8859b4` |
| SH weighted-MSE scratch R9 | `6ebba848-1ca9-4454-9af6-559d24d8a5fc` |
| SH weighted-MSE ignore-edge all-pretrained R9 | `7c7b6119-07e5-4fe3-98bd-bab8d1f728e9` |

The fix is to construct the normalization scalar in fp32. CPU smoke now shows
nonzero SH weighted-MSE main loss. The corrected 30k-step R10 jobs were
submitted on 2026-06-21 04:33-04:34 UTC:

| arm | UUID | config |
|---|---|---|
| SH weighted-MSE all-pretrained R10 | `6c7675a1-7661-4363-9c65-90e1f2e7129e` | `configs/finetune_cremi_real_unetr_aniso_superhuman_shwmse_allpretrained_r10.yaml` |
| SH weighted-MSE scratch R10 | `af35134e-a7ea-4f27-bfeb-4777dd48ae5b` | `configs/finetune_cremi_real_unetr_aniso_superhuman_shwmse_scratch_r10.yaml` |
| SH weighted-MSE ignore-edge all-pretrained R10 | `20239e88-03f5-4269-a348-da79dd2adbb4` | `configs/finetune_cremi_real_unetr_aniso_superhuman_shwmse_ignore_allpretrained_r10.yaml` |
