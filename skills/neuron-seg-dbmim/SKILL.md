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
- Do not claim the dbMiM objective itself improves neuron segmentation from a
  pretrained-vs-scratch comparison alone. Require a matched plain-MAE
  pretraining control with the same data, steps, UNETR finetune recipe, and
  official A/B/C waterz evaluation. R23 is the maintained plain-MAE baseline;
  R24 is the current dbMiM++ attempt that must beat it.

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
- Any eval command using `--metric-backend skimage` must install/verify
  `scikit-image` before the sweep. Without it, graph/RAG/waterz rows can be
  emitted only as caught backend failures and are not usable evidence.
- Current production-style waterz evaluation is CPU-bound even when inference
  runs on GPU. Full-volume A/B/C jobs can spend most wall time in watershed/RAG
  construction and agglomeration; do not judge experiment speed from training
  steps alone.
- CuPy sparse connected components may not speed up large crops because graph
  construction and host/device transfer dominate. Measure it; do not assume GPU
  means faster.
- Auto resource selection may choose `skyinfer-reserved-shared` instead of
  `med-model`. Check the saved submission JSON before telling the user which
  pool is used.
- SiFlow `tasks.list(..., resource_pools=[pool])` can return the same UUID when
  queried across multiple pools. Deduplicate by UUID before reporting total GPU
  usage.
- Watcher logs that only show `checkpoint_wait` mean no checkpoint has been
  observed in TOS yet. Do not infer loss or convergence from that.
- Stop stale SiFlow jobs once a cleaner matched experiment supersedes them.
  Typical candidates are old arch-bench jobs after waterz-only A/B/C results
  exist, branches with a paired publicEM arm already clearly worse on partial
  A/B evidence, and dependency/runtime probes whose conclusion has been
  absorbed into the code. Keep active pairs only when they still answer a live
  scientific question.
- Clear proxy variables before GitHub, TOS, or SiFlow network operations in
  this environment. `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` pointing at
  `192.168.32.28:18000` can make `git push`, `tosutil`, and SDK calls hang.
  If no-proxy `git ls-remote` works but `git push` hangs, inspect credentials:
  this shell has no SSH key, no credential helper, and can inherit a VS Code
  `GIT_ASKPASS` script without the required IPC environment. Never put a
  GitHub token into the remote URL or logs.
- Do not make method claims from partial post-processing stdout. R20 DPP looked
  excellent on an intermediate 45-record fallback, but the complete 60-record
  A/B/C summary was worse than the no-DPP control.
- The first learned-RAG postprocess branch is a negative result: it saves a
  learned scorer, but one-pass pairwise fragment merging over boundary stats
  produced VOI/ARAND far worse than waterz. Keep it as evidence and baseline
  code, not as the current recommended post-processing method.
- MICrONS/CAVE-scale claims require blockwise evidence. Use
  `scripts/evaluate_cremi_blockwise_scale.py` to report full/seam/nonseam
  VOI/ARAND, chunk size, halo, voxels/sec, and RAG-edge density. A method that
  only improves whole-crop CREMI but worsens seam rows is not scale-ready.
- For learnable post-processing, prefer sparse fragment-edge scoring over dense
  global pairwise merging. `scripts/train_sparse_edge_postprocess.py` trains a
  small edge scorer on RAG boundary features and compares against deterministic
  affinity-score baselines; judge it on held-out sample C before calling it a
  positive learned-postprocess result.
- If the user's requirement is fast and robust learnable post-processing with
  waterz-comparable quality, use the R27 philosophy: keep learnable parameters
  tiny (`scale`/`bias` z-y-x affinity calibration), then benchmark deterministic
  fast backends against waterz. Do not spend cards on large learned RAG/global
  merge modules unless the tiny calibrator + graph/RAG path is already close
  enough on held-out sample C and has a clear `postprocess_sec` advantage.
- When a learned postprocess run includes waterz as a reference backend, make
  sure the submitter treats the stage as `needs_waterz_eval`; otherwise the pod
  can start but fail immediately because the bundle lacks `third_party/waterz`
  and Boost headers.
- A small pretrained-vs-scratch gain is not enough for a paper claim. Compare
  against `pretrain_public_em_plain_mae_r23` / `pretrain_em_full_plain_mae_r23`
  before attributing the gain to membrane weighting, structure loss, decision
  masking, or decoder-aware dbMiM.
- The current hard target is `dbMiM++ > plain MAE`, not just `pretrained >
  scratch`. For publicEM, use `scripts/poll_dbmim_tos_results.py --group
  r24_dbmim_vs_mae` and report the VOI/ARAND deltas of R24 vs R23 plain MAE.
