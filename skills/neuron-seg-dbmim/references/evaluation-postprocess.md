# Evaluation and Post-processing Reference

## Metrics

Primary CREMI neuron segmentation metrics:

- `adapted_rand_error` / ARAND: lower is better.
- `voi_split`: false-split component of variation of information.
- `voi_merge`: false-merge component of variation of information.
- `voi_sum = voi_split + voi_merge`: lower is better.

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
- Graph CC is a strong simple baseline. Beat it before investing in complex
  watershed/agglomeration.
- CuPy sparse graph CC is not the same as a custom CUDA union-find; do not
  assume it eliminates the CPU bottleneck.
- Full-volume post-processing can become a memory problem. Plan blockwise or
  streaming graph construction before scaling from crop to whole volume.
- The min-size filter can change VOI/ARAND. Record it with every result.

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
