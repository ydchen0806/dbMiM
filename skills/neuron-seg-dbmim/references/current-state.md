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
