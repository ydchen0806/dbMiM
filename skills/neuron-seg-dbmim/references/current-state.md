# Current State Reference

Use this file to orient a new agent before reporting status or launching more
jobs. Re-check local files because job status and outputs change.

## Repository

- Repo path: `/volume/med-train/users/dchen02/code/dbMiM`
- Remote: `https://github.com/ydchen0806/dbMiM.git`
- Main maintained docs: `DBMIM_WORKFLOW.md` and `EXPERIMENTS.md`
- Maintained model path: `dbmim/models.py::UNETRAnisotropicAffinityNet`
- Maintained training entrypoints:
  - `train_pretrain.py`
  - `train_finetune.py`
  - `scripts/evaluate_cremi_segmentation.py`
  - `scripts/submit_siflow_dbmim.py`

## Current Paper-Aligned Experiment

The active scientific question is whether dbMiM pretraining improves CREMI
neuron segmentation with a paper-aligned anisotropic UNETR backbone and
VOI/ARAND evaluation.

As of 2026-06-20, the earlier R3/R4 supervised finetuning results are
invalidated as scientific evidence because the dataset applied random geometric
flips to the image but not to the instance label. This broke image/label
alignment during supervised affinity training. Keep their logs only as
debugging history.

Current controlled arms:

- `aniso UNETR pretrained`: existing dbMiM pretrained ViT encoder, loaded into
  anisotropic UNETR with interpolated positional embeddings.
- `aniso UNETR scratch`: same architecture, random initialization.
- `aniso UNETR long-pretrained`: long pretraining at `32x160x160`, then same
  finetune.

Open structural/loss/context ablations:

- `no-dtrans`: remove anisotropic z compression.
- `dtrans2`: z stride 2 instead of paper-style stride 4.
- `fs64`: decoder feature size 64.
- `boundary-loss`: stronger z/xy loss weighting.
- `context48`: larger crop `48x192x192`.

## Submitted Jobs Snapshot

All jobs below were submitted on 2026-06-17 via Shanghai changliu with
`sci.g21-3`, 8 GPUs per training pod. Saved submission JSONs show
`resource_pool: cn-shanghai-changliu-skyinfer-reserved-shared`, not
`med-model`.

| stage | UUID | TOS output prefix |
|---|---|---|
| long dbMiM pretrain | `56d6c8f0-184f-4dcd-98c2-01060b3230a0` | `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_long_dbmim/` |
| aniso pretrained finetune | `90e2ca36-6c50-4346-8fa3-d2320b914459` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained/` |
| aniso scratch finetune | `adb18e9f-e4a5-4935-a00e-485388c9545b` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch/` |
| no-dtrans | `1d9a076c-7bea-4224-8067-94bda7b39c95` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_no_dtrans/` |
| dtrans2 | `7a5631c4-63bc-4ae6-ad83-feca3fc78221` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_dtrans2/` |
| fs64 | `53424b91-e37c-4ecd-9eaf-311042355c5c` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_fs64/` |
| boundary-loss | `d015653e-fc71-434c-b6c8-96a51cc04e4a` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_boundary_loss/` |
| context48 | `c4982264-be93-4f50-8aaf-b09cc3feb655` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_context48/` |

Watcher logs live in `outputs/watchers/`. If their tail only says
`checkpoint_wait`, no checkpoint has been observed in TOS yet.

## R5 Root-Cause Fix Snapshot

On 2026-06-20 the supervised dataset and loss were fixed to align with the
SuperHuman/Kisuk affinity-training style:

- `dbmim/datasets.py` now uses synchronized image/label flips through
  `augment_image_and_label`.
- `data.widen_border: true` performs 2D in-plane instance-border invalidation
  before affinity generation.
- `train_finetune.py` supports `loss.loss_type: weighted_mse`, a
  SuperHuman-style per-sample/channel binary ratio weighting over sigmoid
  affinities.
- R5 configs are:
  - `configs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5.yaml`
  - `configs/finetune_cremi_real_unetr_aniso_superhuman_scratch_r5.yaml`

Current R5 SiFlow jobs:

| arm | UUID | status at last update | notes |
|---|---|---|---|
| pretrained R5 4-GPU train | `f9f74d70-4798-400f-a189-afad4c1d7e4e` | Running | med-model, global batch 8, loaded 77 pretrained keys |
| scratch R5 4-GPU train | `162150c0-9350-4254-bdce-c599b7fdcfa4` | Queueing | med-model actual availability was only 1-2 GPUs |
| pretrained R5 full-volume sample-A eval | `9cabcc88-d7ed-4936-89ca-e2891904b648` | Running | 1 GPU med-model, waterz/skimage, sample A full volume |

Stopped R5 attempts due insufficient actual `sci.g21-3` availability:

- `26b6b0c5-5c9c-497f-80b7-4daef7a305ce`: pretrained 8-GPU med-model.
- `8e5119d9-681d-41f7-afd4-4d318e18e0e6`: scratch 8-GPU med-model.
- `452f68a8-dc34-4a60-a3a6-85e9d9cf86c7`: pretrained 8-GPU shared pool.

R5 pretrained training signs are healthier than R4:

- step 600: `train_loss` about `0.321`, boundary Dice loss about `0.552`.
- step 10k-12k: common `train_loss` about `0.10-0.19`, boundary Dice loss
  about `0.14-0.27`.
- TOS checkpoint prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/`.

The R5 eval already confirmed true full-volume sample A scope:
`raw_shape [125,1250,1250]`, crop `[[0,125],[0,1250],[0,1250]]`, checkpoint
`outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/finetuned_latest.pt`.
Wait for waterz VOI rows before making a method claim.

## R2 Restart Snapshot

On 2026-06-18 the old local watchers were killed because no intermediate
checkpoint was visible. The root cause was the bootstrap script uploading
training output directories only after the training process exited. A second
attempt to stop the old remote SiFlow training UUIDs through the SDK hung in
network/proxy connection setup and was interrupted locally; do not assume the
remote tasks stopped unless a fresh SiFlow status query confirms it.

R2 training jobs were submitted explicitly to Shanghai changliu `med-model`,
`sci.g21-3`, 8 GPUs each:

| stage | UUID | TOS output prefix |
|---|---|---|
| pretrained-r2 | `9291405a-046f-4165-bbac-d0fdf71fb3eb` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained_r2/` |
| scratch-r2 | `151ff158-429d-486e-9ffd-59bc71dbe458` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch_r2/` |
| lsd-pretrained-r2 | `2bbd5c24-2124-4d0e-89a2-326932ecd866` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_lsd_pretrained_r2/` |
| lsd-scratch-r2 | `9cc450dd-ba4a-4c43-98ea-765aa363d768` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_lsd_scratch_r2/` |

R2 watcher logs:

- `outputs/watchers/eval_pretrained-r2_20260618T154324_setsid.log`
- `outputs/watchers/eval_scratch-r2_20260618T154324_setsid.log`
- `outputs/watchers/eval_lsd-pretrained-r2_20260618T154324_setsid.log`
- `outputs/watchers/eval_lsd-scratch-r2_20260618T154324_setsid.log`

The R2 bootstrap now uploads `finetuned_latest.pt`, `finetuned_best.pt`, and
JSONL logs periodically while training runs. Evaluation still waits for a
checkpoint before submitting 1-GPU VOI/ARAND jobs.

## Historical Results To Treat Carefully

- The simplified `UNETRAffinityNet` pretrained-vs-scratch run is a negative
  diagnostic, not a paper-aligned conclusion.
- Its crop evaluation produced identical ARAND for pretrained and scratch and
  VOI sum around 4.05 on small center crops. Do not use it to claim dbMiM does
  or does not help.
- MAE-head and graph-CC baselines are useful engineering references but do not
  replace the anisotropic UNETR comparison.

## Status Reporting Checklist

When asked for current status:

1. Check `date`, `outputs/watchers/*.log`, and relevant TOS prefixes.
2. Check submission JSONs under
   `/volume/med-train/users/dchen02/siflow_submissions/yinda_public_skill/`.
3. Report resource pool from the saved JSON, not from memory.
4. Separate "submitted", "running/queued", "checkpoint observed", "eval
   submitted", and "metrics available".
5. If SiFlow SDK log/status calls timeout, say so and fall back to watcher/TOS
   evidence. Do not invent loss values.
