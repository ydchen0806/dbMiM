---
name: neuron-seg-dbmim
description: "Use for dbMiM neuron segmentation work in this repository: CREMI reproduction, anisotropic UNETR finetuning, dbMiM pretraining-effect ablations, VOI/ARAND evaluation, affinity post-processing sweeps, and SiFlow/TOS experiment operations."
metadata:
  short-description: dbMiM neuron segmentation workflow
---

# dbMiM Neuron Segmentation

Use this skill when continuing the dbMiM neuron segmentation project under
`/volume/med-train/users/dchen02/code/dbMiM`, especially when running CREMI
pretraining/finetuning/evaluation, comparing pretrained vs scratch UNETR, or
debugging VOI/ARAND and post-processing.

## Read First

Before acting, load the minimum relevant reference:

- For current experiment state and exact job IDs, read
  [`references/current-state.md`](references/current-state.md).
- For architecture/config/training decisions, read
  [`references/training.md`](references/training.md).
- For VOI/ARAND, threshold sweeps, and post-processing, read
  [`references/evaluation-postprocess.md`](references/evaluation-postprocess.md).
- For SiFlow/TOS submission and monitoring details, read
  [`references/siflow-tos.md`](references/siflow-tos.md).

Also read the live repository files before editing or reporting results:
`DBMIM_WORKFLOW.md`, `EXPERIMENTS.md`, relevant `configs/*.yaml`, and the
current watcher logs under `outputs/watchers/`.

## Core Rules

- Treat `UNETRAnisotropicAffinityNet` with `architecture: unetr_aniso` as the
  maintained paper-aligned path. The simplified `UNETRAffinityNet` runs are
  diagnostic failures only.
- Report segmentation quality with `adapted_rand_error`, `voi_split`,
  `voi_merge`, and `voi_sum`. Do not use affinity Dice alone as the conclusion.
- Prefer `best_by_adapted_rand` for headline ARAND and inspect
  `best_by_voi_sum` to catch threshold pathologies.
- Do not call small crop or center-crop evaluation "whole-volume CREMI".
  Whole-volume neuron segmentation requires blockwise inference and careful
  stitching/post-processing.
- Never print GitHub tokens, SiFlow credentials, TOS AK/SK, or TOS config
  contents. Use existing secret env/config files.
- SiFlow/changliu jobs usually cannot rely on local `/volume` code paths or
  public internet. Use TOS bootstrap bundles and ship all required assets.
- Do not trust a result until the checkpoint, config, evaluation command, crop
  size, threshold grid, backend, and metric-selection rule are all known.

## Standard CREMI Workflow

1. Verify data:
   `data/CREMI` should contain CREMI A/B/C HDF5 files with raw and neuron label
   datasets. Use `scripts/inspect_hdf5.py` or small schema probes; do not read
   whole HDF5 arrays just to inspect names.
2. Pretrain dbMiM:
   `train_pretrain.py --config configs/pretrain_cremi_real_long.yaml`.
3. Finetune anisotropic UNETR:
   compare `finetune_cremi_real_unetr_aniso_pretrained.yaml` against
   `finetune_cremi_real_unetr_aniso_scratch.yaml`. Keep seeds and schedules
   matched unless deliberately running an ablation.
4. Evaluate with `scripts/evaluate_cremi_segmentation.py` using z/xy affinity
   threshold sweeps and `graph_cc`/`cupy_graph_cc`.
5. Summarize both training curves and segmentation metrics. A lower validation
   loss or higher affinity Dice does not necessarily imply better instance
   segmentation.

## Quick Checks

Use these before submitting a new pack:

```bash
python -m py_compile dbmim/models.py train_finetune.py train_pretrain.py \
  scripts/evaluate_cremi_segmentation.py scripts/submit_siflow_dbmim.py
```

For model compatibility, run a tiny forward pass for the exact crop and
architecture. For pretrained loading, verify how many backbone keys load and
whether `pos_embed` interpolation was used.

## Common Pitfalls

- VOI around 4 on tiny center crops is not a faithful CREMI claim; the user
  expects CREMI-volume VOI near the usual scale, so align evaluation scope and
  post-processing before drawing conclusions.
- `waterz`/`elf`/`mahotas` are fragile in offline pods and usually CPU-bound.
  Keep them as negative controls unless the dependency and runtime are proven.
- CuPy sparse connected components may not speed up large crops because graph
  construction and host/device transfer dominate. Measure it; do not assume GPU
  means faster.
- Auto resource selection may choose `skyinfer-reserved-shared` instead of
  `med-model`. Check the saved submission JSON before telling the user which
  pool is used.
- Watcher logs that only show `checkpoint_wait` mean no checkpoint has been
  observed in TOS yet. Do not infer loss or convergence from that.
