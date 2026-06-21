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
| pretrained R5 4-GPU train | `f9f74d70-4798-400f-a189-afad4c1d7e4e` | Succeeded | med-model, global batch 8, loaded 77 pretrained keys, 30k final checkpoint uploaded |
| pretrained R5 full-volume sample-A eval, mid-training latest | `9cabcc88-d7ed-4936-89ca-e2891904b648` | Stopped after useful evidence | 1 GPU med-model, waterz/skimage, sample A full volume |
| pretrained R5 full-volume sample-A eval, final 30k | `68115eb6-ecd3-4f6b-af50-98c26c35faee` | Succeeded | 1 GPU med-model, waterz/skimage, sample A full volume |
| scratch R5 4-GPU train | `162150c0-9350-4254-bdce-c599b7fdcfa4` | Succeeded | med-model, global batch 8, no pretrained, 30k final checkpoint uploaded |
| scratch R5 full-volume sample-A eval, final 30k | `a4afc427-3b32-4439-b9b5-cf6f5aec37c7` | Succeeded | 1 GPU med-model, waterz/skimage, sample A full volume |
| no-widen pretrained R5 ablation | `31935cc0-64f0-4e55-bacd-14d043741acb` | Succeeded | cpt-train, 4 GPU, disables label border widening |
| BCE pretrained R5 ablation | `41fe0043-f7e5-4e10-8067-4c5c5618d926` | Succeeded | cpt-train, 4 GPU, uses BCE instead of weighted MSE |
| boundary-high pretrained R5 ablation | `e28c3264-aaa5-491a-a79f-f307bbe5ef43` | Succeeded | cpt-train, 4 GPU, boundary Dice weight 0.55 |
| no-widen pretrained R5 sample-A eval | `28295ae2-2b2a-4c00-a8b7-5e4c9ba3da16` | Running | cpt-train, 1 GPU, waterz/skimage |
| BCE pretrained R5 sample-A eval | `adb48dd4-e833-4744-addf-78c3defc2c5e` | Running | cpt-train, 1 GPU, waterz/skimage |
| boundary-high pretrained R5 sample-A eval | `c605d26c-163b-4b3b-8fa4-512fbe7029c5` | Running | cpt-train, 1 GPU, waterz/skimage |
| pretrained R5 A/B/C full-volume eval | `812df662-d464-49d3-8622-93da3e1709cc` | Submitted | med-model, 1 GPU, all CREMI samples |
| scratch R5 A/B/C full-volume eval | `98bc4983-7ca2-4f6e-9fe4-31f4a2791ba0` | Submitted | med-model, 1 GPU, all CREMI samples |

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

The R5 eval confirmed true full-volume sample A scope:
`raw_shape [125,1250,1250]`, crop `[[0,125],[0,1250],[0,1250]]`, checkpoint
`outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/finetuned_latest.pt`.

First raw `pred` waterz/skimage rows are in the normal CREMI scale:

| threshold | VOI sum | VOI split | VOI merge | ARAND | n_pred | n_gt |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.742093 | 0.449843 | 0.292251 | 0.126184 | 27535 | 37366 |
| 0.10 | 0.759984 | 0.430677 | 0.329307 | 0.161975 | 23599 | 37366 |
| 0.20 | 0.747036 | 0.410954 | 0.336082 | 0.162093 | 19172 | 37366 |
| 0.30 | 0.735735 | 0.396551 | 0.339184 | 0.161714 | 16231 | 37366 |
| 0.50 | 0.717879 | 0.371610 | 0.346268 | 0.161406 | 12209 | 37366 |

This strongly supports the root-cause diagnosis: the previous VOI 8-10 range
was caused mainly by broken supervised augmentation/target training, not by the
VOI implementation itself. Continue the pretrained R5 run to 30k and start/keep
the scratch R5 control for the pretraining-effect comparison.

Final pretrained R5 30k sample-A eval has improved further. On the final
checkpoint, uncalibrated waterz `hist_quantile` gives:

| threshold | VOI sum | VOI split | VOI merge | ARAND | n_pred | n_gt |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.645302 | 0.367920 | 0.277382 | 0.111310 | 21346 | 37366 |
| 0.10 | 0.637800 | 0.359174 | 0.278626 | 0.111051 | 18972 | 37366 |
| 0.20 | 0.628222 | 0.347504 | 0.280718 | 0.110802 | 16394 | 37366 |
| 0.30 | 0.620097 | 0.336619 | 0.283478 | 0.110683 | 14557 | 37366 |
| 0.50 | 0.604249 | 0.315553 | 0.288696 | 0.109885 | 11402 | 37366 |

The final calibration sweep improved both VOI and ARAND. Current best confirmed
pretrained R5 row is `VOI_sum=0.563006`, `ARAND=0.073386`, waterz threshold
`0.50`, calibration bias `z=-0.50,y=-1.00,x=-1.00`.

The scratch R5 train finished 30k with final `train_loss=0.122041`, affinity
Dice loss `0.032573`, and boundary Dice loss `0.161410`. Its full-volume
sample-A calibration eval finished as UUID
`a4afc427-3b32-4439-b9b5-cf6f5aec37c7`.

Scratch sample-A is currently better than pretrained sample-A:

- scratch best-by-VOI: `VOI_sum=0.546834`, `ARAND=0.066024`,
  bias `z=-0.25,y=-0.50,x=-0.50`, threshold `0.50`.
- scratch best-by-ARAND: `ARAND=0.058093`, `VOI_sum=0.548672`,
  bias `z=-0.50,y=-1.00,x=-1.00`, threshold `0.50`.
- pretrained best: `VOI_sum=0.563006`, `ARAND=0.073386`,
  bias `z=-0.50,y=-1.00,x=-1.00`, threshold `0.50`.

Interpretation: R5 fixed the segmentation method, but sample-A currently does
not support a positive dbMiM-pretraining effect. Wait for the A/B/C full-volume
evals before making a broader claim.

## 2026-06-20 Current R5 Addendum

Sample-A ablations completed with the raw-label `ignore_label=0` metric path:

- BCE pretrained R5 is the best current sample-A VOI row:
  `VOI_sum=0.535448`, `ARAND=0.080269`, bias `z=-0.50,y=-1.00,x=-1.00`,
  waterz threshold `0.50`.
- BCE's best ARAND row is `ARAND=0.067376`, `VOI_sum=0.543432`, threshold
  `0.20`, still weaker than scratch R5's best ARAND `0.058093`.
- Boundary-high pretrained R5 does not improve the main R5 stack:
  `VOI_sum=0.559225`, `ARAND=0.073207`.
- No-widen pretrained R5 is catastrophic:
  `VOI_sum=4.516494`, `ARAND=0.961479`. Treat 2D instance-border invalidation
  as a required part of the method.

Metric-mask clarification:

- CREMI train HDF5 `volumes/labels/neuron_ids` has no original zero labels on
  samples A/B/C, so raw-label `ignore_label=0` is stricter than the CREMI
  challenge boundary-ignore convention.
- `scripts/evaluate_cremi_segmentation.py` now supports
  `--cremi-boundary-ignore-distance-xy` and
  `--cremi-boundary-ignore-distance-z`.
- The official-style A/B/C stages use `xy=1,z=0`, matching the "same section"
  object-boundary wording. Do not mix these official-style numbers with
  raw-label numbers.
- Official-style sample-A rows observed so far:
  - pretrained R5 raw pred threshold `0.50`: `VOI_sum=0.246585`,
    `ARAND=0.077635`, effective `n_gt=10929`, ignored voxels `11.69%`.
  - scratch R5 raw pred threshold `0.50`: `VOI_sum=0.204558`,
    `ARAND=0.048634`, effective `n_gt=10929`, ignored voxels `11.69%`.
  - scratch R5 raw pred threshold `0.10`: best current ARAND
    `0.046854`, `VOI_sum=0.223857`.

Raw-label A/B/C follow-up finished on 2026-06-20:

- pretrained R5 strict raw-label A/B/C: best VOI `1.556559`, best ARAND
  `0.356289`.
- scratch R5 strict raw-label A/B/C: best VOI `1.441593`, best ARAND
  `0.312191`.

Interpretation: strict raw-label A/B/C confirms the sample-A trend. Scratch is
currently ahead of dbMiM-pretrained R5; do not claim a pretraining gain from
the current 30k setup.

Updated follow-up jobs:

| job | UUID | status note |
|---|---|---|
| raw-label pretrained R5 A/B/C | `812df662-d464-49d3-8622-93da3e1709cc` | succeeded, strict diagnostic |
| raw-label scratch R5 A/B/C | `98bc4983-7ca2-4f6e-9fe4-31f4a2791ba0` | succeeded, strict diagnostic |
| official-style pretrained R5 A/B/C | `44efdd27-baf7-418b-aa80-3b74207dc70b` | running, CREMI boundary-ignore xy=1 |
| official-style scratch R5 A/B/C | `5aad9226-1c61-40a4-b330-93bdf87abc31` | running, CREMI boundary-ignore xy=1 |
| corrected encoder-LR pretrained R5 | `c5f2a38a-d6a7-46d2-b46d-73b57a0f000b` | succeeded, encoder lr `2e-5`, decoder lr `8e-5` |
| encoder-LR official-A eval, failed packaging attempt | `1b8cf096-294e-46d5-a3f6-029c219dda08` | failed before eval due newer waterz pyproject/setuptools incompatibility |
| encoder-LR official-A eval, stable waterz_v08 bundle | `8400bb84-5126-4c96-9266-fbe54b5499a5` | running/submitted |
| BCE scratch R5 train | `9f2bbe23-abfa-41c0-b00c-63f76fd76868` | running, cpt-train 4 GPU |
| BCE pretrained official-A eval | `47ddaea5-1a31-4c7c-8518-326d4c46b869` | running, official boundary-ignore sample A |

The first low-encoder run at
`finetune_cremi_real_unetr_aniso_superhuman_lowencoder_pretrained_r5` is
invalid because DDP parameter names prevented encoder/decoder optimizer-group
splitting. It was stopped; use the corrected `encoderlr` prefix instead.

Official-style sample-A calibrated rows observed so far:

- pretrained R5 best observed: `VOI_sum=0.198980`, `ARAND=0.038487`, bias
  `z=-0.50,y=-1.00,x=-1.00`, threshold `0.50`.
- scratch R5 best observed: `VOI_sum=0.177157`, `ARAND=0.021736`, bias
  `z=-0.50,y=-1.00,x=-1.00`, threshold `0.50`.

This is the right CREMI-style direction: boundary-ignore brings VOI well below
the raw-label strict numbers. It still does not show a pretraining gain.

Final-checkpoint watcher logs:

- old scratch watcher, superseded by manual eval:
  `outputs/watchers/eval_scratch_r5_20260620T091340Z_final30k.log`
- no-widen watcher attempt:
  `outputs/watchers/eval_nowiden_pretrained_r5_20260620T093118Z_final30k_nohup.log`
- BCE watcher attempt:
  `outputs/watchers/eval_bce_pretrained_r5_20260620T093118Z_final30k_nohup.log`
- boundary-high watcher attempt:
  `outputs/watchers/eval_boundaryhigh_pretrained_r5_20260620T093118Z_final30k_nohup.log`

Those watcher attempts intentionally waited only for `checkpoint_step_00030000.pt`,
not `finetuned_latest.pt`, but local background processes were not reliable in
this environment. Prefer explicit SiFlow/TOS polling and manual eval
submission once the final checkpoint appears.

## 2026-06-21 Current Snapshot

Official-style A/B/C evaluations with the CREMI boundary-ignore mask
(`xy=1,z=0`) completed:

- pretrained R5 official A/B/C, UUID `44efdd27-baf7-418b-aa80-3b74207dc70b`,
  best VOI `1.250565`, best ARAND `0.332747`.
- scratch R5 official A/B/C, UUID `5aad9226-1c61-40a4-b330-93bdf87abc31`,
  best VOI `1.121139`, best ARAND `0.287647`.

The boundary-ignore mask effective GT counts were sample A `10929`, sample B
`1092`, and sample C `1878`. The ignored voxel fractions were `11.6916%`,
`8.4857%`, and `10.5215%`, respectively.

Conclusion: the VOI implementation and post-processing are now in the expected
CREMI-scale regime, but the current standard dbMiM-pretrained R5 arm does not
beat scratch. Treat scratch R5 as the current aggregate baseline.

Official sample-A ablations now available:

- BCE pretrained R5, UUID `47ddaea5-1a31-4c7c-8518-326d4c46b869`:
  best VOI/ARAND `VOI_sum=0.185597`, `ARAND=0.032785`, bias
  `z=-0.50,y=-1.00,x=-1.00`, threshold `0.20`.
- encoder-LR pretrained R5, UUID `8400bb84-5126-4c96-9266-fbe54b5499a5`:
  best VOI `0.210299`, best ARAND `0.036421`.

BCE pretrained is a credible sample-A improvement over the standard pretrained
arm but is not proven to be a pretraining effect. The paired BCE scratch train
finished as UUID `9f2bbe23-abfa-41c0-b00c-63f76fd76868`, final step 30000
`train_loss=0.147030`, `train_main_loss=0.091217`, `train_dice_loss=0.028496`,
and `train_boundary_dice_loss=0.155396`.

Paired BCE scratch evaluations submitted on 2026-06-21:

| job | UUID | status note |
|---|---|---|
| official-style BCE scratch sample A | `df6582ca-c87c-4843-a8ca-23e41030c3b5` | succeeded |
| raw-label BCE scratch sample A | `4ce29ab6-4775-41f5-ac70-c2b62d94b9c4` | succeeded |

Paired BCE scratch results:

- official-style sample A: best VOI `0.169502`, best ARAND `0.024062`.
- raw-label sample A: best VOI `0.517256`, best ARAND `0.058831`.

Both beat the paired BCE-pretrained rows (`0.185597`/`0.032785` official and
`0.535448`/`0.067376` raw-label). Report BCE as a useful loss/training-stack
choice, not as evidence for a positive dbMiM-pretraining effect. Expand BCE
scratch/pretrained to A/B/C only if a full aggregate comparison is needed.

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
