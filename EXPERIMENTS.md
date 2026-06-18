# dbMiM CREMI Reproduction and Ablations

This note tracks the maintained reproduction path and the real SiFlow
experiments run on CREMI.

## Paper-aligned Anisotropic UNETR Ablation

The maintained UNETR experiment now uses the original anisotropic design from
`model_unetr.py`: `patch_size=(4,16,16)`, hidden-state skip projection with
3/2/1 upsampling stages, and the z-only `dtrans` convolution between decoder3
and decoder2. The current self-contained implementation is
`UNETRAnisotropicAffinityNet` in `dbmim/models.py`, selected with
`architecture: unetr_aniso`.

The simplified `UNETRAffinityNet` jobs listed below are retained only as a
diagnostic failed run. They did not migrate the original anisotropic decoder
and should not be used as a paper-aligned conclusion about dbMiM pretraining.

Current aniso experiment plan:

| arm | backbone | initialization | config | output prefix |
|---|---|---|---|---|
| aniso UNETR pretrained | `UNETRAnisotropicAffinityNet` | existing dbMiM pretrained ViT encoder, with interpolated `pos_embed` | `configs/finetune_cremi_real_unetr_aniso_pretrained.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained/` |
| aniso UNETR scratch | `UNETRAnisotropicAffinityNet` | random initialization | `configs/finetune_cremi_real_unetr_aniso_scratch.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch/` |
| aniso UNETR long-pretrained | `UNETRAnisotropicAffinityNet` | long dbMiM pretrain at `32x160x160` | `configs/finetune_cremi_real_unetr_aniso_longpretrained.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_longpretrained/` |

Validation checks before submitting the aniso jobs:

| check | result |
|---|---|
| `py_compile` maintained entrypoints | passed |
| aniso UNETR forward at `16x64x64` | output `1x3x16x64x64` |
| aniso UNETR forward at `32x160x160` | output `1x3x32x160x160`, 18.07M parameters |
| load existing pretrain into aniso UNETR | 77 backbone keys loaded, including interpolated `pos_embed` |
| SiFlow dry-run bundles | aniso pretrained/scratch finetune, long pretrain, and aniso eval stages dry-run successfully |

Evaluation summaries now include both `best_by_adapted_rand` and
`best_by_voi_sum` in `cremi_segmentation_summary.json`.

Aniso jobs submitted on 2026-06-17 through Shanghai changliu with 8-GPU
training pods selected by `--resource-pool auto`:

| stage | UUID | output prefix |
|---|---|---|
| long dbMiM pretrain | `56d6c8f0-184f-4dcd-98c2-01060b3230a0` | `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_long_dbmim/` |
| aniso UNETR pretrained finetune | `90e2ca36-6c50-4346-8fa3-d2320b914459` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained/` |
| aniso UNETR scratch finetune | `adb18e9f-e4a5-4935-a00e-485388c9545b` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch/` |

Watchers running from the login node:

| purpose | watcher log |
|---|---|
| aniso pretrained eval | `outputs/watchers/eval_aniso_pretrained_20260617T140343.log` |
| aniso pretrained large-crop eval | `outputs/watchers/eval_aniso_large_pretrained_20260617T140344.log` |
| aniso scratch eval | `outputs/watchers/eval_aniso_scratch_20260617T140344.log` |
| aniso scratch large-crop eval | `outputs/watchers/eval_aniso_large_scratch_20260617T140344.log` |
| long pretrain -> long-pretrained finetune | `outputs/watchers/finetune_aniso_longpretrained_20260617T140406.log` |
| long-pretrained eval | `outputs/watchers/eval_aniso_longpretrained_20260617T140406.log` |
| long-pretrained large-crop eval | `outputs/watchers/eval_aniso_large_longpretrained_20260617T140406.log` |

Additional open ablations submitted on 2026-06-17 to use the H200 pool while
the main runs are training:

| stage | purpose | UUID | cards |
|---|---|---|---:|
| `no-dtrans` | remove the anisotropic z compression module | `1d9a076c-7bea-4224-8067-94bda7b39c95` | 8 |
| `dtrans2` | use z stride 2 instead of the paper-style z stride 4 | `7a5631c4-63bc-4ae6-ad83-feca3fc78221` | 8 |
| `fs64` | increase UNETR decoder feature size from 32 to 64 | `53424b91-e37c-4ecd-9eaf-311042355c5c` | 8 |
| `boundary-loss` | stronger z/xy boundary-aware affinity loss | `d015653e-fc71-434c-b6c8-96a51cc04e4a` | 8 |
| `context48` | larger crop/context `48x192x192` | `c4982264-be93-4f50-8aaf-b09cc3feb655` | 8 |

Each ablation has normal and large-crop eval watchers under
`outputs/watchers/eval_ablation_*_20260617T154639.log`. These jobs bring the
submitted training footprint from 24 GPUs to 64 GPUs, before queued 1-GPU evals
and the future long-pretrained finetune.

### R2 restart with visible checkpoints and LSD-style auxiliary target

The 2026-06-17 aniso jobs were not usable for fast decision making because the
TOS bootstrap script uploaded training outputs only after the training command
exited. Watchers therefore saw no intermediate checkpoint and most logs stayed
at `checkpoint_wait`. Local watcher processes were killed on 2026-06-18. A
second attempt to stop old remote SiFlow jobs through `client.tasks.stop` hung
inside SDK network/proxy connection handling and was interrupted locally; treat
remote stop state as unconfirmed until SiFlow task state is queried successfully.

Open-source review shifted the next method attempt toward architecture/objective
changes rather than CPU-only agglomeration:

- PyTorch Connectomics keeps connectomics training, inference, decode and
  evaluation as separate stages; this supports our current decision to keep
  VOI/ARAND decode sweeps explicit instead of judging affinity Dice alone.
- The funkelab LSD codebase motivates using instance-label-derived auxiliary
  shape descriptors to improve boundary/instance segmentation. Our
  implementation is a lightweight LSD-style centroid-offset target, not the
  full funlib local-moment descriptor.
- `affogato`/mutex watershed and GASP-style signed graph agglomeration remain
  plausible post-processing branches, but they are dependency-heavy and do not
  fix the current missing-checkpoint/training-objective problem. Keep
  affinity graph connected components with z/xy threshold sweep as the stable
  decode baseline for this round.

Code changes for the R2 restart:

- `labels_to_local_shape_descriptors` builds a 4-channel auxiliary target:
  foreground, dz, dy, dx centroid offsets.
- `train_finetune.py` now slices the first 3 output channels for affinity loss
  and optionally supervises extra descriptor channels with
  `train.loss.lsd_weight`.
- `scripts/evaluate_cremi_segmentation.py` always slices model output to the
  first 3 affinity channels before sigmoid/decode, so 7-channel LSD heads do
  not change VOI/ARAND semantics.
- `scripts/submit_siflow_dbmim.py` now starts a lightweight TOS sync loop for
  training stages, periodically uploading `finetuned_latest.pt`,
  `finetuned_best.pt`, and JSONL logs while training is still running; the final
  directory upload still runs after the command exits.

R2 controlled arms submitted on 2026-06-18 to Shanghai changliu `med-model`,
`sci.g21-3`, 8 GPUs each. All use global batch 16, crop `32x160x160`,
`max_steps: 30000`, `save_every: 2`, and explicit VOI/ARAND evaluation watchers:

| arm | config | UUID | output TOS prefix | watcher |
|---|---|---|---|---|
| pretrained-r2 | `configs/finetune_cremi_real_unetr_aniso_pretrained_r2.yaml` | `9291405a-046f-4165-bbac-d0fdf71fb3eb` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained_r2/` | `outputs/watchers/eval_pretrained-r2_20260618T154324_setsid.log` |
| scratch-r2 | `configs/finetune_cremi_real_unetr_aniso_scratch_r2.yaml` | `151ff158-429d-486e-9ffd-59bc71dbe458` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch_r2/` | `outputs/watchers/eval_scratch-r2_20260618T154324_setsid.log` |
| lsd-pretrained-r2 | `configs/finetune_cremi_real_unetr_aniso_lsd_pretrained_r2.yaml` | `2bbd5c24-2124-4d0e-89a2-326932ecd866` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_lsd_pretrained_r2/` | `outputs/watchers/eval_lsd-pretrained-r2_20260618T154324_setsid.log` |
| lsd-scratch-r2 | `configs/finetune_cremi_real_unetr_aniso_lsd_scratch_r2.yaml` | `9cc450dd-ba4a-4c43-98ea-765aa363d768` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_lsd_scratch_r2/` | `outputs/watchers/eval_lsd-scratch-r2_20260618T154324_setsid.log` |

Validation before submit:

| check | result |
|---|---|
| Python compile maintained entrypoints | passed |
| `configs/finetune_smoke_lsd.yaml` CPU smoke training | passed, 2 steps and validation |
| aniso pretrained-r2 forward | output `1x3x32x160x160` |
| aniso lsd-pretrained-r2 forward | output `1x7x32x160x160` |
| eval dry-run for LSD r2 | checkpoint download path and config were correct |

Do not report these R2 jobs as results until their watcher submits eval and the
corresponding `cremi_segmentation_summary.json` exists. The first checkpoint
should become visible much earlier than the previous run because the TOS sync
loop uploads latest/best weights during training.

## Diagnostic Simplified UNETR Run

This earlier run used the simplified `UNETRAffinityNet` decoder and a small
center-crop evaluation. It is not the current paper-aligned architecture.

| arm | backbone | initialization | config | output prefix |
|---|---|---|---|---|
| UNETR pretrained | `UNETRAffinityNet` | dbMiM pretrained ViT encoder | `configs/finetune_cremi_real_unetr_pretrained.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_pretrained/` |
| UNETR scratch | `UNETRAffinityNet` | random initialization | `configs/finetune_cremi_real_unetr_scratch.yaml` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_scratch/` |

Training jobs submitted on 2026-06-17 through Shanghai changliu `med-model`
with 8 GPUs per pod:

| arm | UUID | TOS bootstrap bundle |
|---|---|---|
| UNETR pretrained | `0ad158c2-b5dc-4a25-a916-07be456a56bb` | `tos://agi-data/users/dchen02/dbmim/bundles/dbmim-finetune-cremi-unetr-pretrained.20260617T101657+0800.tar.gz` |
| UNETR scratch | `a0b2978b-8237-4c0d-ba8a-fcadb8c79453` | `tos://agi-data/users/dchen02/dbmim/bundles/dbmim-finetune-cremi-unetr-scratch.20260617T101657+0800.tar.gz` |

Evaluation stages:

| arm | SiFlow stage | UUID | output prefix |
|---|---|---|---|
| UNETR pretrained | `eval-cremi-unetr-pretrained` | `74591aa2-b752-4ef8-8e19-dc2a5b0d4230` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_pretrained/` |
| UNETR scratch | `eval-cremi-unetr-scratch` | `d7cd2251-0682-4b9d-9e84-8561e7927a2b` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_scratch/` |

Watchers are running from the login node and will submit the two eval stages
when `finetuned_best.pt` or `finetuned_latest.pt` appears:

| arm | watcher log |
|---|---|
| UNETR pretrained | `outputs/watchers/eval_unetr_pretrained_20260617T102435_setsid.log` |
| UNETR scratch | `outputs/watchers/eval_unetr_scratch_20260617T102435_setsid.log` |

Premature eval record to ignore: `d5e1a290-ac9c-4913-9407-8fb195e75ac3`
was submitted before the watcher checkpoint probe was hardened, so the eval
should be retried by the strict watcher after the UNETR checkpoint is actually
available.

Headline reporting should use `best_by_adapted_rand` from
`cremi_segmentation_summary.json` and report `adapted_rand_error`, `voi_split`,
`voi_merge`, and `voi_sum` from the same row.

Completed result over CREMI A/B/C center crops (`32x256x256`, stride
`16x128x128`, 24 threshold/backend settings):

| arm | best backend | z threshold | xy threshold | ARAND | Rand F | VOI split | VOI merge | VOI sum | affinity Dice | affinity IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNETR pretrained | graph_cc | 0.65 | 0.90 | 0.808767 | 0.191233 | 4.054326 | 0.000000 | 4.054326 | 0.961900 | 0.926685 |
| UNETR scratch | graph_cc | 0.65 | 0.90 | 0.808767 | 0.191233 | 4.054326 | 0.000000 | 4.054326 | 0.961843 | 0.926580 |

Training best validation rows from the tail logs:

| arm | best epoch | step | val_dice | val_iou | val_loss |
|---|---:|---:|---:|---:|---:|
| UNETR pretrained | 1719 | 25800 | 0.966879 | 0.935917 | 0.360447 |
| UNETR scratch | 1899 | 28500 | 0.966754 | 0.935687 | 0.379407 |

This is a negative diagnostic result for the simplified UNETR decoder used in
that run:
dbMiM pretraining slightly improves affinity validation metrics, but the
instance-segmentation metric selected by the threshold sweep is identical to
scratch. Both UNETR arms are also worse than the earlier MAE-head graph-CC
baseline (`ARAND 0.745897`, `VOI sum 4.344215`) on the same 3-sample CREMI
evaluation. This is the result that motivated migrating the original
anisotropic decoder before drawing a conclusion about the pretraining effect.

## Baseline Weights

- Pretrain job: `c197160f-467f-4693-b544-3f52e52c1d3a`
- Finetune job: `0b49afe7-c929-4d54-85f5-28e7288ddb8e`
- Pretrained weight:
  `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_dbmim/pretrained_latest.pt`
- Finetuned weights:
  `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_dbmim/finetuned_best.pt`
  and
  `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_dbmim/finetuned_latest.pt`

Best baseline affinity validation during finetuning:

| epoch | step | val_dice | val_iou | val_loss |
|---:|---:|---:|---:|---:|
| 199 | 12600 | 0.970613 | 0.942943 | 0.175768 |

## Post-processing Ablations

### CPU and legacy backend sweep

SiFlow UUID: `2c6f6ee6-b9ed-49b8-a3fb-9eb6999a780e`

Output:
`tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_postprocess_sweep/`

Best row over 3 CREMI samples at crop `32x256x256`:

| backend | threshold | ARE | Rand F | VI split | VI merge | postprocess sec |
|---|---:|---:|---:|---:|---:|---:|
| graph_cc | 0.65 | 0.745897 | 0.254103 | 2.614762 | 1.729453 | 0.185 |

Other stable CPU backends were slower or less accurate:

| backend | best threshold | ARE | postprocess sec |
|---|---:|---:|---:|
| cc3d_mean | 0.85 | 0.771131 | 0.068 |
| mahotas_agglomeration | 0.85 | 0.782438 | 0.122 |
| scipy_agglomeration | 0.75 | 0.790835 | 0.124 |

`waterz` and `elf` were not available in the offline pod. `mahotas` and `scipy`
watershed without a stronger merge policy over-segmented badly.

### GPU graph and anisotropic threshold sweep

SiFlow UUID: `7d50e1dd-e70d-4517-ba15-a675a8b871e3`

Best row reconstructed from SiFlow logs:

| backend | z threshold | xy threshold | ARE | Rand F | VI split | VI merge | postprocess sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cupy_graph_cc | 0.75 | 0.90 | 0.745897 | 0.254103 | 2.614719 | 1.729624 | 0.151 |

The score matches the CPU `graph_cc@0.65` baseline within noise. This supports
using separate z/xy thresholds, but the current CuPy sparse connected-component
path does not provide a reliable speedup on larger crops.

### Large crop speed probe

SiFlow UUID: `0c1dcf0f-66de-4293-b26d-048ad7f647c2`

Crop: `64x512x512`, 3 CREMI samples, z/xy threshold `0.75/0.90`.

| backend | ARE | Rand F | postprocess sec |
|---|---:|---:|---:|
| graph_cc | 0.840717 | 0.159283 | 2.647 |
| cupy_graph_cc | 0.840717 | 0.159283 | 2.770 |

CuPy sparse graph CC was slightly slower at this size, likely due to transfer
and sparse connected-component overhead. For production, the current fastest
stable path is CPU sparse graph CC or a future custom CUDA/Numba union-find
kernel rather than the generic CuPy sparse graph routine.

### Seeded RAG / watershed agglomeration

SiFlow UUID: `ac510cdd-4584-4372-bf94-a46b38ccfc64`

This job was intentionally stopped after more than 1700 log rows on sample A:
RAG/watershed variants stayed around ARE `0.89` to `0.95` and were clearly worse
than graph CC. A local grid-seed smoke test showed stable over-segmentation but
too many fragments and slow quantile scoring. This branch is kept as a negative
result for now.

## Training Objective Ablations

The finetuning script now supports a config-driven affinity objective:

- Original behavior: BCE only.
- Optional `train.loss.channel_weights` for z/y/x channel weighting.
- Optional `train.loss.pos_weight` for channel-wise positive edge weighting.
- Optional `train.loss.dice_weight` soft Dice regularization.
- Optional `train.loss.focal_gamma` focal scaling for hard affinity edges.

Two SiFlow training jobs were launched from the CREMI pretrained weight and
completed successfully:

| stage | config | UUID | output TOS prefix |
|---|---|---|---|
| zdice | `configs/finetune_cremi_real_zdice.yaml` | `3cdab0a5-e6b0-4d76-a2dd-140fecb83085` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_zdice/` |
| zdice_focal | `configs/finetune_cremi_real_zdice_focal.yaml` | `f1c9f7be-9568-415c-ba4e-cb1059fe244f` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_zdice_focal/` |

Weights:

- `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_zdice/finetuned_best.pt`
- `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_zdice/finetuned_latest.pt`
- `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_zdice_focal/finetuned_best.pt`
- `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_zdice_focal/finetuned_latest.pt`

Evaluation jobs:

| stage | UUID | best backend | z threshold | xy threshold | ARE | Rand F | VI split | VI merge |
|---|---|---|---:|---:|---:|---:|---:|---:|
| zdice | `71909575-0140-427d-b454-23ac8bbc1366` | cupy_graph_cc | 0.75 | 0.95 | 0.745898 | 0.254102 | 2.614564 | 1.729930 |
| zdice_focal | `e3026149-00d1-4501-8c92-dd24816a19a0` | cupy_graph_cc | 0.65 | 0.85 | 0.746220 | 0.253780 | 2.617670 | 1.755964 |

These training-objective ablations did not improve CREMI segmentation over the
original finetuned checkpoint. They are kept as negative controls and as
infrastructure for future loss experiments.

## Current Recommendation

For this codebase, the most stable enhancement path is:

1. Treat UNETR pretrained vs UNETR scratch as the main paper-aligned comparison.
2. Keep graph connected components as the primary post-processing backend.
3. Tune z/xy thresholds separately and report VOI/ARAND from the same best
   adapted-Rand row.
4. Treat generic CuPy sparse graph CC as a probe only. For real acceleration,
   implement a custom GPU union-find or blockwise stitching kernel.
5. Keep waterz/elf/mahotas RAG variants as negative controls unless a stronger
   merge policy is introduced.
