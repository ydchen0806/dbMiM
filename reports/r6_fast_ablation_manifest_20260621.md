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

This job reached 90k+ steps by 2026-06-21 03:24 UTC and was still running
toward 160k steps. It continuously syncs
`tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_all_dbmim_r6/pretrained_latest.pt`.

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
