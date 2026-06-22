# Evaluation and Post-processing Reference

## Metrics

Primary CREMI neuron segmentation metrics:

- `adapted_rand_error` / ARAND: lower is better.
- `voi_split`: false-split component of variation of information,
  `H(pred | gt) = H(joint) - H(gt)`.
- `voi_merge`: false-merge component of variation of information,
  `H(gt | pred) = H(joint) - H(pred)`.
- `voi_sum = voi_split + voi_merge`: lower is better.

The code before 2026-06-19 had the `voi_split` and `voi_merge` labels swapped.
`voi_sum` and ARAND were unaffected, but do not use older reports to infer
split-versus-merge direction.

The current SuperHuman-aligned path uses
`skimage.metrics.adapted_rand_error(target, pred, ignore_labels=(0,))` and
`skimage.metrics.variation_of_information(target, pred, ignore_labels=(0,))`.
For CREMI neuron labels where `0` is ignore/background, this is the expected
ARAND/VOI convention.

CREMI challenge-style neuron segmentation additionally ignores ground-truth
pixels close to object boundaries on the same XY section. The downloaded train
HDF5 `volumes/labels/neuron_ids` for samples A/B/C has no original zero labels,
so plain `ignore_label=0` is a stricter raw-label diagnostic. For headline
CREMI-style numbers, run `scripts/evaluate_cremi_segmentation.py` with:

```bash
--metric-backend skimage \
--ignore-label 0 \
--cremi-boundary-ignore-distance-xy 1 \
--cremi-boundary-ignore-distance-z 0
```

Always report whether this boundary-ignore mask was enabled. On full sample A,
`xy=1,z=0` ignored about `11.69%` of voxels and reduced the effective GT
instance count from `37366` to `10929`. Do not put raw-label and
boundary-ignore rows in the same aggregate table unless the metric mask is a
column.

The evaluation summary should contain both:

- `best_by_adapted_rand`
- `best_by_voi_sum`

Use `best_by_adapted_rand` for headline ARAND, but always inspect
`best_by_voi_sum`. Bad thresholds can over-merge or all-merge and distort one
selection criterion.

## Scope Caveat

Do not call the current crop evaluations whole-volume CREMI. The current
evaluation commands use crops such as `32x320x320` or `64x512x512` over A/B/C.
Whole-volume CREMI requires blockwise inference, stitching, memory-aware graph
construction, and consistent post-processing across block boundaries.

The user expects normal CREMI volume-level VOI to be around the literature
scale, not the inflated/diagnostic values from tiny crops. If VOI looks around
4 on a small crop, suspect evaluation scope, thresholding, or post-processing
before making a scientific claim.

For true whole-volume R5 checks, `--crop-size 0 0 0` should be preserved as a
full-volume sentinel. Verify logs show `raw_shape [125,1250,1250]` and crop
`[[0,125],[0,1250],[0,1250]]`; older code accidentally coerced non-positive
crop sizes back to the model window.

## Standard Eval Shape

Use `scripts/evaluate_cremi_segmentation.py` with:

```bash
python scripts/evaluate_cremi_segmentation.py \
  --config configs/finetune_cremi_real_unetr_aniso_pretrained.yaml \
  --checkpoint outputs/finetune_cremi_real_unetr_aniso_pretrained/finetuned_best.pt \
  --data-dir data/CREMI \
  --output-dir outputs/eval_cremi_unetr_aniso_pretrained \
  --crop-size 32 320 320 \
  --stride 16 80 80 \
  --thresholds 0.0 \
  --backends graph_cc cupy_graph_cc \
  --min-size 32 \
  --z-thresholds 0.45 0.55 0.65 0.75 0.85 \
  --xy-thresholds 0.65 0.75 0.85 0.90 0.95 \
  --max-samples 3 \
  --device cuda
```

For debugging suspicious VOI/ARAND values, run
`scripts/evaluate_cremi_diagnostics.py`. It evaluates predicted, inverted, and
GT-oracle affinities with high z/xy thresholds and records `n_pred` versus
`n_gt`:

```bash
python scripts/evaluate_cremi_diagnostics.py \
  --config configs/finetune_cremi_real_unetr_aniso_pretrained_r2.yaml \
  --checkpoint outputs/finetune_cremi_real_unetr_aniso_pretrained_r2/finetuned_best.pt \
  --data-dir data/CREMI \
  --output-dir outputs/diagnose_cremi_unetr_aniso_pretrained_r2 \
  --crop-size 32 320 320 \
  --stride 16 80 80 \
  --backends graph_cc cupy_graph_cc \
  --min-size 0 \
  --z-thresholds 0.85 0.90 0.95 0.975 0.99 0.995 \
  --xy-thresholds 0.90 0.95 0.975 0.99 0.995 0.999 \
  --max-samples 3 \
  --device cuda \
  --include-oracle-affinity \
  --include-inverted-affinity \
  --diagnostics
```

SiFlow has matching `diagnose-cremi-unetr-aniso-<name>` stages.

Use large-crop evals as robustness checks:

- crop `64x512x512`
- stride `16x80x80` or similar
- same z/xy sweep

## Stable Backends

Current stable production-like backend:

- affinity graph connected components (`graph_cc`) with separate z/xy threshold
  sweep.

Useful but not proven faster:

- `cupy_graph_cc`. It can match CPU graph CC but often does not speed up because
  graph construction, sparse connected components, and transfers dominate.

Negative or fragile controls:

- `mahotas_agglomeration`: available sometimes, CPU-bound, not better in
  current sweeps.
- `waterz` and `elf`: often unavailable in offline SiFlow pods unless bundled
  correctly; CPU-bound and dependency-fragile.
- seeded RAG/watershed: over-segmented and slower in current probes.

## Post-processing Lessons

- Separate z and xy affinity thresholds matter for anisotropic EM volumes.
- All channels should not necessarily share one threshold.
- Always compare `n_pred` and `n_gt`. Very small `n_pred` means over-merging;
  enormous `n_pred` means over-splitting even if affinity Dice is high.
- R2 diagnostics showed learned affinities were saturated: low thresholds
  over-merged, while high thresholds (`xy~0.975-0.995`) over-split into
  hundreds of thousands of components on `32x320x320` crops.
- GT-oracle affinities reached VOI sum about `0.27` on the same crops, so the
  graph-CC convention and metric path are not the main failure.
- Graph CC is a strong simple baseline. Beat it before investing in complex
  watershed/agglomeration.
- In the R15/R16 full-volume architecture benchmarks, simple graph-CC with the
  early threshold grid again produced VOI around `7.9` on sample A. Do not use
  those early rows as a method conclusion; wait for waterz or run a standalone
  waterz-only official A/B/C eval.
- The poller may synthesize a `cremi_segmentation_summary.json` from SiFlow
  stdout before the canonical TOS summary exists. Treat that fallback summary
  as partial unless `num_records` and `sample_names` prove that A/B/C all
  completed. For the current waterz-only grid, expect three sample names and
  about `3 samples x 3 calibration biases x 5 thresholds = 45` records per
  arm.
- `scripts/poll_dbmim_tos_results.py` now prints `PARTIAL` for official A/B/C
  stages until `sample_A_20160501.hdf`, `sample_B_20160501.hdf`, and
  `sample_C_20160501.hdf` are all present. Do not count a `PARTIAL` summary as
  complete even if it has a best VOI/ARAND row.
- The R17 fine sweep uses 9 thresholds and 6 calibration biases across A/B/C,
  so a full summary should contain 162 records. Early stdout fallback with 9 or
  10 records is sample-A-only and only useful for sanity checking.
- CuPy sparse graph CC is not the same as a custom CUDA union-find; do not
  assume it eliminates the CPU bottleneck.
- Full-volume post-processing can become a memory problem. Plan blockwise or
  streaming graph construction before scaling from crop to whole volume.
- The min-size filter can change VOI/ARAND. Record it with every result.

## SuperHuman Waterz Lessons

For SuperHuman-style evaluation:

- Build 2D watershed fragments from xy boundary map
  `1 - 0.5 * (aff_y + aff_x)`.
- Use mahotas `maxima_distance` seeds and waterz agglomeration as the stable
  offline path.
- `waterz` v0.8 is stable enough when bundled with Boost headers, but it does
  not support the newer `MeanAffinityProvider` scoring string. Use
  `hist_quantile`/`OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, false>>`
  unless a newer waterz ABI is fixed and tested.
- The newer `funkey/waterz` source tree supports mean-affinity scoring, but a
  2026-06-20 SiFlow attempt failed before evaluation because the pod's
  setuptools rejected the newer `pyproject.toml` `project.license` metadata.
  Keep `waterz_v08` as the stable default bundle; run mean-scoring only after
  patching/package-building the newer source offline.
- Waterz can mutate the fragments array during agglomeration. Always pass a
  contiguous copy for each threshold/variant; otherwise threshold sweeps are
  contaminated by previous thresholds.
- Eval-time logit bias can trade merge for split, but R3/R4 calibration only
  moved full-volume sample-A VOI from about 9-10 to about 8-9. Treat that as a
  diagnostic, not a final method.
- After the R5 synchronized-augmentation and SuperHuman-style target/loss
  fixes, full-volume sample-A waterz/skimage moved into the expected CREMI
  scale. The final pretrained R5 30k checkpoint has uncalibrated best
  `VOI_sum=0.604249`, `ARAND=0.109885` at waterz threshold `0.50`, and
  calibrated best `VOI_sum=0.563006`, `ARAND=0.073386` at waterz threshold
  `0.50` with logit bias `z=-0.50,y=-1.00,x=-1.00`.
- The final scratch R5 30k checkpoint is currently stronger on sample A:
  best-by-VOI `VOI_sum=0.546834`, `ARAND=0.066024` with logit bias
  `z=-0.25,y=-0.50,x=-0.50`, and best-by-ARAND `ARAND=0.058093`,
  `VOI_sum=0.548672` with logit bias `z=-0.50,y=-1.00,x=-1.00`. Do not claim
  a positive pretraining effect from sample-A evidence.
- Under the CREMI-style boundary-ignore mask (`xy=1,z=0`), sample-A raw-pred
  waterz rows are much lower: scratch R5 threshold `0.50` reached
  `VOI_sum=0.204558`, `ARAND=0.048634`; pretrained R5 threshold `0.50`
  reached `VOI_sum=0.246585`, `ARAND=0.077635`. This confirms the user's
  metric concern: the previous raw-label numbers were strict diagnostics, not
  challenge-style headline metrics.
- With calibration under the same CREMI-style mask, the current sample-A bests
  are scratch R5 `VOI_sum=0.177157`, `ARAND=0.021736`, and pretrained R5
  `VOI_sum=0.198980`, `ARAND=0.038487`, both at waterz threshold `0.50` with
  bias `z=-0.50,y=-1.00,x=-1.00`.
- Official-style A/B/C aggregation with the same mask completed on 2026-06-21.
  Scratch R5 remained stronger: scratch best VOI `1.121139` and best ARAND
  `0.287647`; pretrained R5 best VOI `1.250565` and best ARAND `0.332747`.
  The effective GT counts after masking were sample A `10929`, sample B
  `1092`, and sample C `1878`. This is the expected CREMI-scale regime, but it
  is not a positive pretraining result.
- Official sample-A ablations: BCE pretrained reached
  `VOI_sum=0.185597`, `ARAND=0.032785`; encoder-LR pretrained reached best
  VOI `0.210299` and best ARAND `0.036421`. These are useful pretrained-side
  probes, but compare against the paired BCE scratch evaluation before
  attributing any improvement to pretraining.
- The paired BCE scratch sample-A evaluations completed and are stronger than
  BCE pretrained under both metric masks: official-style best VOI `0.169502`
  and best ARAND `0.024062`; raw-label best VOI `0.517256` and best ARAND
  `0.058831`. Therefore current BCE gains should be treated as a loss/training
  effect, not a dbMiM-pretraining effect.
- For fast method selection, run full-volume `sample_A_20160501.hdf` first
  with waterz/skimage and the same thresholds/biases. Once a candidate beats
  the current R5 baseline, expand to samples B/C and then report aggregate
  tables.
- For final-checkpoint comparisons, only trigger eval after
  `checkpoint_step_00030000.pt` exists. Periodic `finetuned_latest.pt` uploads
  are useful for debugging but can silently evaluate an intermediate model.

## Reporting Checklist

For every result table include:

- checkpoint URI or local path;
- config name;
- eval crop size and stride;
- samples used;
- backend;
- z and xy thresholds;
- `min_size`;
- ARAND, Rand F, VOI split, VOI merge, VOI sum;
- whether the row is selected by ARAND or VOI.

## R13/R14 Evaluation Lessons

Use `scripts/poll_dbmim_tos_results.py --group r13|r14q|r14 --once --logs
--siflow-fallback` to pull TOS summaries and, if needed, reconstruct summaries
from SiFlow stdout. Check the summary `source`, `sample_names`, and the `n`
field before calling a result A/B/C.

Rules learned from R14q:

- A fallback summary with `sample_names=['sample_A_20160501.hdf']` and `n=1`
  is sample-A-only even if the eval stage name contains `abc`.
- A credible A/B/C fallback summary should have sample A/B/C and aggregate rows
  with `n=3`.
- Full-volume official waterz rows normally include `threshold`,
  `waterz_scoring`, `rag_quantile`, `boundary_threshold`, calibration biases,
  `voi_sum`, and `adapted_rand_error`. Keep these columns in reports.
- Do not average a sample-A-only row with A/B/C rows. Re-run or wait for the
  full summary.
- The user's expected CREMI scale is roughly VOI around `1.x` for A/B/C
  official-style aggregation. Values near `0.2-0.4` may be valid for sample A
  under boundary-ignore masking, but are suspicious if presented as A/B/C.

Current quick-screen conclusion:

- `MAWS+BCAR-rank` is the best R14q complete A/B/C arm so far:
  best VOI `1.0407487`, ARAND at best VOI `0.1928854`.
- `MAWS15+BCAR-rank` is slightly worse than `MAWS0.75+BCAR-rank`, so do not
  blindly raise membrane weighting.
- `MAWS-only` improves VOI slightly but worsens ARAND; keep the paired BCAR
  comparison.

## R15 Post-processing Benchmark Lessons

The architecture-exploration post-processing benchmark intentionally measures
both quality and runtime. Preserve these columns whenever summarizing:

- `inference_sec`
- `postprocess_sec`
- `metrics_sec`
- backend/error status
- threshold, z/xy threshold, calibration bias, and scoring mode

Standalone stage `eval-cremi-arch-explore-postprocess-r15q` and post-train
`--post-train-arch-bench` runs use full CREMI volumes with:

```text
crop-size 0 0 0
stride 16 80 80
backends graph_cc cupy_graph_cc seeded_rag waterz
thresholds 0.10 0.20 0.30 0.50
z/xy thresholds 0.05 0.10 0.20 0.30 0.50
metric-backend skimage
CREMI boundary ignore xy=1,z=0
calibration biases: 0/0/0, -0.25/-0.5/-0.5, -0.5/-1.0/-1.0
```

Operational rules:

- If logs contain `ModuleNotFoundError("No module named 'skimage'")`, the
  benchmark is invalid, not a method failure. The submitter must package
  `wheelhouse_superhuman_eval` and verify `skimage`, `waterz`, and `mahotas`
  before evaluation.
- Use `--fail-on-backend-error` for architecture/post-process sweeps. Silent
  backend failure rows are useful for debugging but should not become a result
  table.
- `graph_cc` is the stable quality baseline. `cupy_graph_cc` can still be
  slower if graph construction and host/device transfer dominate. Treat GPU
  post-processing as a measured runtime hypothesis, not an assumption.
- `seeded_rag` and `waterz` are CPU-heavy controls. They are scientifically
  useful because they match common neuron-segmentation pipelines, but they are
  the current wall-time bottleneck on full volumes.
- Sample-A rows with VOI around `0.2-0.4` under boundary-ignore are plausible;
  A/B/C aggregate rows should normally be around `1.x`. Always state `n=1` or
  `n=3` before comparing to prior CREMI-scale numbers.
- A/B/C official waterz completed for R16/R17 on 2026-06-22. The expected
  scale check passed: best rows are around VOI `1.00-1.10`, not the broken
  `7-10` graph-CC regime. Current best VOI is R17 MAWS+MSE publicEM
  (`1.002919`); current best ARAND is R16 long-affinity SHW-MSE publicEM
  (`0.181445`). Use those as the active baselines until R18 and the fine
  calibration sweep finish.

## Learned Postprocess Evaluation Contract

As of the R21 end-to-end postprocess branch, `train_finetune.py` can save a
learned `postprocess` module inside the finetune checkpoint. The evaluator
`scripts/evaluate_cremi_segmentation.py` must load and apply that module before
sigmoid, thresholding, waterz, graph-CC, or affinity-stat reporting. It prints
and writes:

```text
learned_postprocess_loaded: true|false
```

If this field is `false` for a DPP checkpoint, the evaluation is not testing
the learned postprocess and must be rerun or debugged. The post-train official
A/B/C eval in `scripts/submit_siflow_dbmim.py` uses the same checkpoint path,
so DPP rows should have the module loaded automatically once the updated bundle
is used.

DPP rows should still report the normal calibration-bias sweep columns. The
learned postprocess is applied first; the explicit eval-time calibration biases
are then an additional sweep. This makes the comparison conservative: if DPP
only learns the same bias that the sweep already tests, it may not improve the
best-by-VOI row. A real DPP gain should appear as either lower best VOI/ARAND
or a more robust plateau across calibration biases.

## Large-scale Blockwise Inference Lessons

For MICRONS/CAVE-style deployment, do not extrapolate from a single whole-crop
CREMI forward pass. Run the blockwise scale proxy:

```bash
python scripts/evaluate_cremi_blockwise_scale.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q.yaml \
  --checkpoint outputs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q/finetuned_latest.pt \
  --data-dir data/CREMI \
  --output-dir outputs/eval_cremi_blockwise_scale_r17q/publicem \
  --crop-size 64 512 512 \
  --stride 16 80 80 \
  --chunk-size 32 256 256 \
  --halo 8 64 64 \
  --seam-width 2 16 16 \
  --backends graph_cc seeded_rag \
  --metric-backend skimage \
  --cremi-boundary-ignore-distance-xy 1 \
  --cremi-boundary-ignore-distance-z 0
```

Required reporting columns:

- `affinity_variant`: at least `blockwise_halo`; optionally `blockwise_no_halo`.
- `region`: `full`, `seam`, and `nonseam`.
- `chunk_size`, `halo`, `seam_width`, `num_chunks`, and `voxels_per_sec`.
- `num_fragments`, `num_rag_edges`, and `rag_edges_per_mvoxel` when RAG stats
  are enabled.

Scale-readiness rule: compare publicEM-pretrained and scratch on both full and
seam rows. A lower full-volume VOI with a worse seam VOI is not a convincing
large-scale method. For deployment, the post-processing output should be
chunked fragments plus sparse RAG edges, not a monolithic waterz/mahotas run
over the whole dataset.

## Sparse Learned Edge Postprocess

`scripts/train_sparse_edge_postprocess.py` is the current preferred scaffold
for a learnable, scale-aware post-processing experiment. It:

- builds slice-wise watershed fragments;
- extracts sparse RAG boundary features only for touching fragments;
- trains `LearnedRAGMergeScorer` on train samples A/B;
- reports held-out C metrics separately;
- compares against deterministic `mean`, `xy_mean`, and conservative affinity
  edge scores.

Do not call this a positive method unless the held-out best row beats the
deterministic sparse baseline and does not reproduce the earlier learned-RAG
failure mode of severe over-merging. The saved `sparse_edge_scorer.pt` is useful
only with its feature standardization metadata and exact fragment-generation
settings.
