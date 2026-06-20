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
- Waterz can mutate the fragments array during agglomeration. Always pass a
  contiguous copy for each threshold/variant; otherwise threshold sweeps are
  contaminated by previous thresholds.
- Eval-time logit bias can trade merge for split, but R3/R4 calibration only
  moved full-volume sample-A VOI from about 9-10 to about 8-9. Treat that as a
  diagnostic, not a final method.

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
