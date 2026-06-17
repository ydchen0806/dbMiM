# dbMiM CREMI Reproduction and Ablations

This note tracks the maintained reproduction path and the real SiFlow
experiments run on CREMI.

## Paper-aligned UNETR Ablation

The current paper-aligned experiment uses a UNETR affinity backbone and reports
VOI plus adapted Rand error (ARAND). The goal is to isolate the effect of dbMiM
pretraining:

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

This is a negative result for the current UNETR decoder/postprocess setup:
dbMiM pretraining slightly improves affinity validation metrics, but the
instance-segmentation metric selected by the threshold sweep is identical to
scratch. Both UNETR arms are also worse than the earlier MAE-head graph-CC
baseline (`ARAND 0.745897`, `VOI sum 4.344215`) on the same 3-sample CREMI
evaluation. The next useful experiment is not more threshold search on this
decoder; it should address affinity calibration/segmentation topology, e.g.
stronger boundary-aware loss, longer/larger-context finetuning, or a
postprocess objective that penalizes the all-merge failure mode reflected by
`VOI merge = 0`.

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
