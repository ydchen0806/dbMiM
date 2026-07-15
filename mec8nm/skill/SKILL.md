---
name: dbmim-neuron-seg
description: Reproduce, monitor, improve, or report dbMiM EM instance segmentation on MEC 8nm myelin or CREMI using anisotropic UNETR, affinity learning, VOI/ARAND, WaterZ, LoCoAgg, publicEM/fullEM pretraining, SiFlow/TOS, strict frozen splits, visual diagnostics, and petabyte-readiness gates. Use for MEC experiment continuation, result comparison, checkpoint selection, crop/full-block gating, external-EM pretraining, learned post-processing, large-volume planning, or project handoff.
---

# dbMiM Neuron Segmentation

## MEC 8nm Current Protocol

Read `STATE.md` before acting. Search its latest MEC sections first:

```bash
tail -n 320 /volume/med-train/users/dchen02/.codex/skills/dbmim-neuron-seg/STATE.md
grep -n "R81\|Hzzs32\|Hzzs33\|PB-Scale\|Strict" /volume/med-train/users/dchen02/.codex/skills/dbmim-neuron-seg/STATE.md | tail -n 100
```

- Treat MEC as myelin instance segmentation, not generic neuron-body or
  mitochondria segmentation. Labels are instance IDs; `0` is background.
- Use only M01--07 and M09 for supervised training and M08/M10 for validation.
  M11/M12 have now been consumed once by the hzzs28/hzzs29 designated replay;
  they are historically non-pristine and must never be used for calibration,
  architecture, scorer, threshold, alpha, or executor choice.
- Keep headline MEC metrics on WaterZ. Lower ARAND and VOI are better. Label
  crop metrics explicitly; never compare them directly with full-block or
  CREMI A/B/C publication metrics.
- Require crop output integrity: exactly 80 raw rows, samples exactly M08/M10,
  backend exactly WaterZ, finite metrics, and an empty failure ledger. Analyze
  with `scripts/analyze_mec8nm_paired_crop.py` against the frozen R58 CSV.
- Pass the crop gate only when mean ARAND, mean VOI, M08 ARAND, and M08 VOI all
  non-regress versus R58. Aggregate improvement with M08 regression is a fail.
- Submit no full-block task after a crop-gate failure. Submit no M11/M12,
  deployment, or 2PB task before an M08/M10 full-block pass.

Current frozen MEC anchors:

- Crop R58: mean ARAND/VOI `0.022546003667/0.311183450751`; M08
  `0.014948239287/0.256127966754`; M10
  `0.030143768046/0.366238934749`. Setting: zero bias, temperature `1.15`,
  threshold `.50`.
- Official M08/M10 full-block R57: mean ARAND/VOI
  `0.252279795612/1.062235821039`. This is the reliable absolute deployment-
  scale anchor; crop values around `.02/.31` are not full-volume results.
- R58 + frozen CREMI-trained Stem-Micro on full M08/M10 reaches mean
  ARAND/VOI `0.091720333957/0.590987031259`, versus same-R58 crop-frozen
  WaterZ `0.418965808348/1.536809043919`; it is the current validation and
  runtime research result, not the final MEC quality default.
- The one paired M11/M12 replay is mixed. Same-R58 Stem-Micro changes
  ARAND/VOI/time from WaterZ `0.019254036555/0.279892831006/68.807s` to
  `0.016160111808/0.334732313606/17.840s`: ARAND and latency improve, VOI
  regresses, and M11/M12 move in opposite directions. Do not reopen them.
- R71 EMA improves crop mean to `0.020969504209/0.299190914044`, but fails the
  M08 guard (`0.037575210047/0.433248168057`). Keep R58/R57 as anchors and do
  not promote R71.
- R72 photometric consistency is negative: best crop mean
  `0.028005094678/0.369409206168`, with M08
  `0.042332990281/0.473893471718`. It has zero strict gate passes. Do not
  promote R72 or sweep its consistency weight/ranges.
- R73 decoder-aware MEC raw pretraining is strongly negative: best crop mean
  `0.055075340982/0.486239101228`, with M08
  `0.037673258094/0.443902435184`. Do not repeat its masked-reconstruction plus
  raw-gradient pseudo-affinity objective or allow raw-only adaptation to
  overwrite the R57 decoder/head.
- R74 encoder-only MEC raw adaptation is also negative despite preserving all
  137 R57 decoder/head tensors: best crop mean
  `0.035560839373/0.359082802791`, with M08
  `0.037953939399/0.356534733043`. Do not tune its 4000-step masked-
  reconstruction schedule or learning rate.
- R75 reflect-reconstructed halo consistency is negative: best crop mean
  `0.036896625527/0.387556128221`, with M08
  `0.040070609288/0.377633169630`. Do not sweep its halo width, reflection
  mode, or consistency weight.
- R76 cross-run R58/R71 EMA weight soup is negative for alphas
  `.010,.025,.050,.100,.200`. Least-regressed alpha `.050` has crop mean
  `0.030034391192/0.338540905377` and M08
  `0.035741292818/0.342101735986`. Do not expand this alpha grid.
- R77 same-run R58 early stopping is negative. Step8000 has mean
  `0.025127315227/0.308270133998` but M08
  `0.043286870594/0.405479642617`; step10000 is worse. Do not sweep earlier
  R58 checkpoints.
- R78 real paired-overlap consistency improves crop mean to
  `0.021307430975/0.254415142584` but fails M08 at
  `0.037685313511/0.346308913501`. No setting preserves both M08 metrics. Do
  not sweep its shift, weight, or teacher direction.
- R79 crop-level topology-routed R58/R71-EMA distillation is strongly negative:
  best crop mean `0.062021381189/0.412646966283`, with M08
  `0.035591466186/0.337509064870` and M10
  `0.088451296192/0.487784867696`. Do not sweep route thresholds/widths,
  distillation weight, or teacher pair.
- R80 decoder-feature instance embedding improves crop mean to
  `0.018971582484/0.264526010140` and M10 to
  `0.004421046309/0.157600299074`, but fails M08 at
  `0.033522118658/0.371451721207`. Do not sweep embedding weight, stride,
  margins, instance filters/cap, or normalization.
- R81 instance-pair boundary-leak CVaR is negative: best crop mean
  `0.028377941932/0.352231169545`, with M08
  `0.032977813947/0.375969227906` and M10
  `0.023778069917/0.328493111184`. Do not sweep leak weight, tail fraction,
  maximum affinity, minimum edges, or pair cap.

### Interpret "Best" By Protocol

- For the frozen MEC crop gate, R58 remains the selected affinity model. No
  R59--R81 candidate passes mean ARAND, mean VOI, M08 ARAND, and M08 VOI
  simultaneously.
- For conservative full-block deployment quality, use R57 plus calibrated
  WaterZ as the absolute anchor. Do not compare it directly with crop metrics.
- For same-R58 M08/M10 validation quality and runtime, frozen CREMI LoCoAgg is
  the strongest research result. It is not a universal production default
  because its one M11/M12 replay has mixed quality and external AxonEM transfer
  exposes severe false merges.
- Hzzs32 signed-VI is a diagnostic mechanism win on already-consumed M07/M09,
  not a new selected MEC checkpoint. Hzzs33 fails both AxonEM gates.
- Never answer "best on MEC" without naming affinity checkpoint, postprocessor,
  crop versus full-block protocol, evaluated blocks, and whether those blocks
  were still untouched. No untouched MEC selection split remains; a new claim
  requires new labeled blocks or an official hidden evaluation.

MEC data and evidence paths:

- split: `/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/mec8nm_splits.json`
- manifest/license: `/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/mec8nm_manifest.json`
- machine-readable current analyses:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/`
- compact handoff:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_absolute_results_and_publication_comparison_20260713.md`
- visualization script:
  `/volume/med-train/users/dchen02/code/dbMiM/scripts/visualize_mec8nm_affinity_segmentation.py`

## MEC Experiment Selection

- Preserve the R57/PublicEM initialization and R58 supervised recipe unless a
  controlled hypothesis specifically changes one component.
- PublicEM dbMiM has controlled CREMI gains over same-seed plain MAE, but no
  strict R58 scratch-versus-PublicEM MEC ablation exists. R70's initializer-only
  FullEM-R33 swap regresses crop ARAND/VOI by 137.06%/56.65%, so do not equate
  more external EM data with better MEC transfer.
- Do not repeat rejected families: blind calibration/bias sweeps, gamma dose,
  simple-region sampling, soft-negative weighting, LSD auxiliary heads,
  fixed two-voxel affinity auxiliaries, border-width sweeps, R57/R59 logit
  mixing, direct FullEM-R33 initialization, or the R72 stopped-teacher
  photometric-consistency recipe, or R73 raw-gradient decoder-aware domain
  pretraining, R74 encoder-only MEC masked-reconstruction adaptation, or R75
  reflect-reconstructed halo consistency, or R76 cross-run R58/R71 linear
  checkpoint soup, R77 R58 intermediate-checkpoint selection, R78 real
  paired-overlap consistency, or R79 crop-level topology-routed R58/R71
  teacher distillation, or R80 decoder-feature discriminative instance
  embedding, or R81 local instance-pair boundary-leak CVaR.
- Treat EMA as a useful mechanism signal, not a selected checkpoint: it improves
  M10 and aggregate crop metrics while substantially harming M08.
- Require a materially new hypothesis before another GPU wave. R71, R78, and
  R80 improve M10/aggregate metrics while harming M08, while R79 shows that a
  crop-level training-label route between checkpoints can harm both blocks.
  Do not retry teacher blending, generic global instance embedding, or local
  boundary-weighting/focal/CVaR variants with threshold/weight/margin changes.
  Require a materially different region/hierarchy-aware training signal, or
  preserve R58 and gate an independent post-processing delta. Enforce the
  strict one-wave crop gate before any full-block work.
- External EM labels with different taxonomy must not become MEC supervision.
  Use raw FAFB/FIB25/Kasthuri/MitoEM/MB-MOC only for self-supervision unless
  label semantics and licenses are independently audited.
- The inherited approximately 485.8 GB FullEM mirror is research-grade lineage:
  object hashes exist, but publisher-level licenses/versions are incomplete.
  Do not describe it as production-provenance data.

## MEC SiFlow Pattern

- Use `yinda-public-skill` for Shanghai changliu/TOS submission. Prefer one
  8-GPU `sci.g21-3` pod in `med-model`; keep only one active MEC wave unless the
  user explicitly raises the cap.
- Bundle code, checkpoint, configs, WaterZ source/wheels, and evaluator. Do not
  assume changliu can see local paths or reach Hugging Face.
- Harden each experiment wrapper to force run/config name, train IDs
  `1 2 3 4 5 6 7 9`, `EVAL_SPLITS=val`, `EVAL_BACKENDS=waterz`,
  `EXPECTED_METRIC_ROWS=80`, and `NPROC=8`.
- Verify remote startup logs for CUDA 8, WaterZ, expected pretrained-key count,
  dataset size, loss config, and the experiment-specific delta before trusting
  a run.
- Fetch crop results from the run-specific TOS prefix and run:

```bash
python scripts/analyze_mec8nm_paired_crop.py \
  --metrics <candidate>/cremi_segmentation_metrics.csv \
  --baseline-metrics outputs/mec8nm_remote_results/r58_balanced_crop_20260711/cremi_segmentation_metrics.csv \
  --output <candidate>/paired_analysis_vs_r58.json
```

## Non-negotiable Evaluation Rules

- Report CREMI A/B/C full-volume metrics, not tiny-crop smoke metrics, when making method claims. VOI around 1.0 is normal for this protocol; very low VOI from small test crops is not comparable.
- Compute headline results by grouping complete A/B/C rows with identical `(affinity_variant, backend, threshold, seed_distance, boundary_threshold, min_boundary, score_mode, rag_quantile, waterz_scoring)`. Average `voi_sum` and `adapted_rand_error` only over groups containing all three samples.
- Use `ignore_label=0`, CREMI-style boundary ignore `xy=1`, `z=0`, and full-volume sliding-window inference for the current ablation protocol.
- Treat VOI and ARAND as separate operating points. The best-VOI threshold can differ from the best-ARAND threshold.

## Current Strong Baselines

- Best current result as of 2026-07-01: R48, publicEM dbMiM R16 pretrained encoder, seed309, 20k-step MSE+MAWS finetune. Official-style ABC from SiFlow logs: `VOI=0.962154`, `ARAND=0.178252`.
- Strong 12k same-seed dbMiM: R45, `VOI=0.986481`, `ARAND=0.186187`.
- Same-seed plain MAE control: R47, `VOI=1.043065`, `ARAND=0.190743`.
- Older publicEM MAE/dbMiM comparison: R17 dbMiM `VOI=1.002919`, R23 plain MAE `VOI=1.027073`.
- FullEM best: R33 fixed mixed edge/random dbMiM, `VOI=1.039372`, better than fullEM plain MAE and scratch but not better than R48.
- R52/R53 constrained adaptive prior policy finetunes are negative: R52 `VOI=1.056275`, `ARAND=0.194687`; R53 `VOI=1.080240`, `ARAND=0.209107`.

## Model And Loss Recipe

- Backbone should be `unetr_aniso_em`, not generic isotropic UNETR. Use crop `32x160x160`, patch size `4x16x16`, transformer skips, staged decoder upsampling, and z-only anisotropic transition before the final decoder block.
- Finetune labels are z/y/x nearest-neighbor affinities from CREMI instance labels.
- The stable supervised loss is pure MSE plus MAWS. Keep `membrane_weight=0.75` for the main line; `membrane_weight=1.0` improved ARAND in R50 but did not beat R48 VOI.
- Use `encoder_lr=1e-5`, base lr `8e-5`, weight decay `0.01`, AMP, batch size 2 per GPU. R48 shows that extending from 12k to 20k steps can matter.

## Mask-Policy Lessons

- Do not equate "more edge reward" with better dbMiM. R38/R39 and strong hard-edge rewards were negative downstream.
- R51 was unhealthy: mask ratio stuck around 0.55, edge fraction 1.0, edge coverage 0.0, policy loss frozen to zero. Treat it as a negative control.
- R52 is healthier than R51 as an RL-style policy diagnostic, but downstream R52/R53 finetunes are negative. Do not continue that branch without a materially new reward or policy objective.
- Stable dbMiM gains currently come from the combination of EM-aware masking, membrane weighting, anisotropic UNETR, and enough finetune steps; the policy module must be validated through downstream VOI/ARAND, not pretraining loss alone.

## Post-processing Rules

- The post-processing branch is now an independent Stem-Micro method line, not
  only a dbMiM ablation. The publication target is a Nature Methods-style system
  paper: backbone-agnostic EM affinity post-processing with calibrated
  hierarchy, blockwise large-volume execution, stable IDs, uncertainty, and
  CAVE/Neuroglancer-friendly outputs.
- For that target, the main claim cannot be "GPU connected components is fast".
  The main claim should be: learned/label-free calibration plus a deterministic
  hierarchy or signed-graph solver makes neuron segmentation more accurate,
  more transferable, and more deployable at MICrONS/CAVE scale.
- waterz remains the reference post-processing backend for headline numbers. It is CPU-heavy; first low-threshold agglomeration can take about 90s per full volume and later thresholds are typically about 25-30s per point.
- Keep threshold/calibration sweeps small on GPU pods to avoid wasting expensive H200 time on CPU-only agglomeration. For broad sweeps, use separate CPU jobs or sparse candidate points from previous best rows.
- A fast learned or differentiable post-process is useful only if it is robust and comparable to waterz under the same ABC grouping. Graph connected components and simple thresholding are speed baselines, not waterz replacements.
- Calibration biases around `(z=-0.25,y=-0.5,x=-0.5)` and `(z=-0.5,y=-1.0,x=-1.0)` are recurring useful regions. Always verify per run.
- 2026-07-02 R48 full-volume post-process comparison:
  - Deployable single-setting waterz: fine sweep `VOI=0.9772`, `ARAND=0.1857`, merge `17.36s/volume`; broad sweep `VOI=0.9858`, `ARAND=0.1835`, merge `15.89s/volume`.
  - Oracle-by-sample waterz upper bound: broad sweep `VOI=0.9384`, `ARAND=0.1755`, merge `15.69s/volume`.
  - Learned GPU RAG with GT edge labels: single threshold `0.98` gives `VOI=1.3001`, `ARAND=0.3104`, merge `9.36s/volume`; oracle-by-sample gives `VOI=1.2512`, `ARAND=0.3101`, merge `9.37s/volume`. Per-sample best is A `0.3253`, B `1.5772`, C `1.8511`.
  - Waterz-decision binary distillation from teacher threshold `0.20` is a negative or at best marginal result: single threshold `0.98` gives `VOI=1.2885`, `ARAND=0.3084`, merge `9.20s/volume`, but oracle-by-sample is still `1.2885` and worse than GT-edge learned RAG. Teacher positive fractions were high (A about `0.60`, B about `0.64` on full-volume eval target generation), causing over-merge at lower thresholds and no reliable B/C gain.
  - Pure CuPy graph connected components is not usable despite speed: full A around `VOI=7.91`, `ARAND=0.99`.
- For learned post-processing, report merge time separately from fragment/RAG feature time. On full CREMI volumes the final GPU merge is about `9-10s`, but watershed fragments plus RAG feature construction is about `90s/volume`, so the next speed bottleneck is blockwise/GPU RAG feature generation, not connected components itself.
- Do not spend more jobs on low-threshold waterz binary teacher labels without changing the target. Better next directions: conservative teacher threshold/ranking distillation, merge-order/hierarchy loss, edge calibration with global fragment-size/context features, and blockwise GPU feature construction.
- 2026-07-02/04 proxy/network pitfall: local env proxy variables pointing to `192.168.32.28:18000` can make GitHub, Hugging Face, TOS, and SiFlow SDK calls hang or time out. Direct access worked after unsetting `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, `all_proxy`, and `ALL_PROXY`; use temporary unsets or `git -c http.proxy= -c https.proxy=` for clone/push. Do not bake proxy settings into repo scripts. When a network command looks blocked, first retry with these proxy vars cleared before assuming GitHub/HF/SiFlow is down.
- 2026-07-02 post-processing literature-driven direction: waterz/MWS/GASP all point to hierarchy or signed-graph decisions, not one hard binary merge label. The next learned-postprocess branch is waterz-ladder distillation: generate teacher segmentations from multiple waterz thresholds in one hierarchy call, convert each to RAG edge labels, and train a soft merge target equal to the fraction of thresholds that merge each edge. This should reduce the over-merge failure seen with single-threshold `0.20` distillation.
- Longer-term postprocess directions worth testing after ladder distillation: add signed repulsive/cannot-link edges from long-range affinities, add an LSD/shape-context head or fragment descriptors for merge scoring, and optimize blockwise fragment/RAG construction because full-volume GPU connected components is no longer the dominant runtime.
- 2026-07-02 ladder distillation result: standard waterz ladder `[0.20,0.24,0.28,0.32]` trained fast and made sample A look strong, but ABC oracle best was only `VOI=1.4161`, `ARAND=0.3613`, merge `9.17s/volume` (`A=0.3434`, `B=1.7193`, `C=2.1855`). This is faster than waterz merge but far below waterz quality (`VOI~0.98` global), so do not use MLP waterz-ladder RAG as the main post-processing claim. High-threshold ladder increased teacher positive fraction and was stopped; low-threshold ladder reduced positive fraction but A degraded and was stopped. Next credible route is signed graph / MWS / GASP-style repulsive edges or waterz blockwise/parallel acceleration, not more scalar-threshold ladder tuning.
- 2026-07-03 waterz scoring alias sweep is confirmatory/negative: `hist_quantile` remains the only robust option tested. On R48 ABC, best deployable single config was `hist_quantile`, threshold `0.20`, bias `(-0.25,-0.50,-0.50)`, `VOI=0.962154`, `ARAND=0.178252`, merge about `15.38s/volume`; `min` and `max` scoring were much worse (`VOI` often around `5-7`). Do not spend more GPU time on waterz scoring aliases unless a genuinely new scorer implementation is added.
- 2026-07-03 signed-reweight learned RAG narrow sweep: subtracting a small signed boundary penalty is useful but not yet waterz-comparable. With thresholds only up to `0.96`, best ABC row was subtract alpha `0.15`, threshold `0.96`, `VOI=1.330421`, `ARAND=0.305353`, merge `9.46s/volume`; same-grid baseline at threshold `0.96` was `VOI=1.406976`, `ARAND=0.366906`. The gain mainly came from sample B (`2.0445 -> 1.5620` at threshold `0.96`), while sample C did not improve (`1.8511` baseline vs `1.8568` best signed row). This is a mechanism signal, not a final headline, because the older learned-RAG table included threshold `0.98` and got `VOI~1.3001`.
- 2026-07-03 signed high-threshold follow-ups are negative for VOI. `arch-explore-r48-signed-reweight-highthr` (`3cfa857a-1f35-44e5-9f2f-e50adb45e784`) completed: best VOI was still the plain learned-RAG GPU baseline at threshold `0.98`, `VOI=1.300060`, `ARAND=0.310392`; the best ARAND row, subtract alpha `0.025` at threshold `0.99`, improved ARAND to `0.233119` but worsened VOI to `1.358886`, mainly by hurting sample C. `arch-explore-r48-signed-veto-highthr` (`5cd248d2-ce11-471a-a587-0f8453e4a8b0`) completed and was effectively a no-op: best `VOI=1.299683`, `ARAND=0.310348`, only `0.0004` VOI better than the baseline and not a meaningful improvement. Stop scalar signed-veto/reweight tuning unless the model predicts true long-range repulsive edges.
- 2026-07-03 morphology-context learned RAG (`arch-explore-r48-rag-morphctx`, `03cde2c9-5b1a-441b-9d61-842ae5173df5`) completed. Adding fragment centroids, centroid deltas, boundary density, and boundary-mean interaction features improves the learned-RAG best-VOI row only slightly: basic `VOI=1.300060`, `ARAND=0.310392`; morphology `VOI=1.294653`, `ARAND=0.283912`, both at threshold `0.98`, merge about `9.2s/volume`. The ARAND gain is more visible than the VOI gain. Per-sample behavior is mixed: sample B improves strongly, while A/C degrade at some thresholds. Feature extraction overhead is small, so morphology context can stay as a cheap learned-RAG default, but it is not waterz-comparable and not a headline post-processing solution. The first morphology submission `fda63524-b5ff-4fa6-8c50-44333e0e10df` failed before training because the private submitter omitted `.yaml` from the config path; always inspect generated `run.sh` or do a dry-run before submitting newly added private stages.
- 2026-07-03 literature check for next post-processing work: Mutex Watershed/MWS is the cleanest algorithmic match because it explicitly combines attractive short-range and repulsive long-range cues without a waterz-like scalar threshold. LSD/shape descriptors are a strong feature-design prior for RAG scoring because they encode local morphology beyond boundary affinity. Connectivity-consensus / multimodal agglomeration papers support adding fragment morphology, raw/embedding context, and graph neighborhood features instead of relying on pairwise boundary mean alone. For MICrONS/CAVE scale, treat blockwise hierarchy and cross-block sparse reconciliation as part of the method, not just engineering.
- 2026-07-04 conservative seeded-RAG is negative for the "fast and universal" goal. It can tune sample A to strong numbers, but ABC aggregate degrades (`VOI=1.8250`, `ARAND=0.3759`) and postprocess time is about `181s`, much slower than batched waterz. Per-sample best VOI was A `0.6173`, B `2.3361`, C `2.3130`; this is volume-specific behavior, not a general EM-volume method.
- 2026-07-04 current best deployable postprocess remains batched waterz on R48: `VOI=0.962154`, `ARAND=0.1765-0.1783`, postprocess about `17.46s` for the threshold sweep in the latest summary. This is the only tested line that is both near the expected CREMI scale and plausibly portable across unseen EM volumes.
- 2026-07-04 avoid claiming learned-RAG or GPU connected components as production replacements. Learned-RAG final GPU merge is fast, but fragment/RAG construction plus cross-volume generalization remain the bottlenecks. CuPy/GPU CC, seeded-RAG, and blockwise seeded-RAG are fast baselines but failed quality (`VOI` often around `7` for CC/seeded variants). Use them as ablations, not headlines.
- 2026-07-04 a new label-free auto-calibration path was added to `scripts/evaluate_cremi_segmentation.py`: `--auto-calibration-targets` and `--auto-calibration-quantiles` choose per-volume z/y/x logit biases so a channel quantile maps to target affinity probabilities. This is meant for "any EM volume" deployment where GT labels are unavailable and fixed CREMI biases may not transfer.
- 2026-07-04 Stem-micro architecture exploration is running as four 8-GPU SiFlow packs, each internally launching four 2-GPU groups on one pod. Use lowercase `stem-micro-*` task prefixes because Kubernetes job names should be lowercase; refer to them as Stem-micro in reports. The old `archx-pack-a/b/c` submissions were stopped and replaced.
- 2026-07-04 Stem-micro pack matrix: `stem-micro-pack-a` runs waterz global/auto-median/scoring/metric-overhead; `stem-micro-pack-b` runs wide calibration/auto-quantile/no-label default/temperature; `stem-micro-pack-c` runs seed-density/blockwise waterz/large halo/MWS-affogato probe; `stem-micro-pack-d` runs learned-RAG waterz LOOC/all/GT LOOC/ABC ensemble.
- 2026-07-04 external MWS/GASP/affogato code cloning from GitHub initially failed locally because proxy `192.168.32.28:18000` timed out. After clearing proxy variables, `affogato`, `GASP`, and `mwatershed` cloned successfully under `/volume/med-train/users/dchen02/code/_refs/`. Do not block main experiments on external clone success; the `probe_external_postprocess_backends.py` stage records module/install availability and synthetic smoke results if sources or wheels are present; otherwise treat it as a dependency/probe result.
- 2026-07-04 Stem-micro A/B/C/D results: `stem-micro-pack-a/b/d` succeeded; `stem-micro-pack-c` was marked failed only because optional `r48-mws-affogato-probe` crashed on missing `affogato` parent import, while its waterz/blockwise stages completed and uploaded. Fixed the probe to catch `find_spec` parent-import exceptions so optional backend absence does not fail a pack.
- 2026-07-04 best confirmed global waterz refinement: `postproc_r48_waterz_wide_calib` / `postproc_r48_waterz_channel_temp` with bias `(-0.20,-0.45,-0.45)` and temperature `1.15`. Best VOI row: threshold `0.22`, `VOI=0.956401`, `ARAND=0.173080`, postprocess about `17.5s` in batched waterz. Best ARAND row: threshold `0.18`, `ARAND=0.172876`, `VOI=0.964369`. This is a small but real improvement over the previous R48 waterz headline `VOI=0.962154`, `ARAND=0.178252`.
- 2026-07-04 strongest new candidate is blockwise waterz, not learned postprocess. `postproc_r48_blockwise_waterz_prod` with chunk `32x512x512`, halo `8x64x64`, threshold `0.20` reached full ABC `VOI=0.937941`, `ARAND=0.185819`, postprocess about `22.24s`; threshold `0.18` gave `ARAND=0.183815`, `VOI=0.939757`, postprocess about `71.87s`. Treat this as a promising VOI gain that needs confirmation because ARAND is worse than global waterz and the first run included RAG-stat overhead.
- 2026-07-04 blockwise large halo is negative: chunk `48x768x768`, halo `12x96x96` gave full ABC `VOI=1.024512`, `ARAND=0.207184`. Do not assume larger halo is better.
- 2026-07-04 label-free auto-calibration is not yet good enough as a default. Auto median/quantile variants fell back to the fixed `(-0.20,-0.45,-0.45)` best rows around `VOI=0.968078`, and no-label default was `VOI=0.997257`, `ARAND=0.196692`. Keep auto-calibration as a deployment research branch, not the main headline.
- 2026-07-04 learned-RAG remains below waterz despite fast final merge. Best ABC ensemble was `VOI=1.148095`, `ARAND=0.206505`, final merge about `9.84s`; waterz-ladder/all and LOOC variants were worse. Do not present learned-RAG as waterz-comparable yet.
- 2026-07-04 `stem-micro-pack-e` was submitted to confirm the blockwise result and test blockwise calibration: UUID `de8d35eb-d5b8-431d-8c22-ade09901ae19`, stages `postproc_r48_blockwise_waterz_calib`, `postproc_r48_blockwise_waterz_prod_lite`, `postproc_r48_blockwise_waterz_chunk40`, and fixed `postproc_r48_mws_affogato_probe`. As of submission it was `Pending`; do not submit more packs until this 8-GPU pack starts or fails.
- Method guidance for "fast and good on any EM volume": keep waterz/MWS-style hierarchy or signed-graph partitioning as the robustness layer; move learning into affinity calibration, merge-score calibration, or waterz-teacher distillation, not direct hard CC. For MICrONS/CAVE scale, make blockwise hierarchy, halo stitching, and sparse cross-block reconciliation explicit method components.
- 2026-07-04 Nature Methods design direction: Stem-Micro should be a
  minimal-hyperparameter policy over calibrated hierarchical post-processing.
  Policy inputs are affinity histograms, anisotropy, boundary density,
  fragment-size/RAG statistics, halo seam disagreement, and optional backbone
  uncertainty. Policy outputs are channel-wise calibration, solver choice,
  hierarchy cut, halo/reconciliation confidence, and uncertainty for
  proofreading. The learnable part should be merge-rank distillation or
  score calibration, leaving the final connectivity to waterz/MWS/GASP-style
  deterministic graph solvers.
- Design document for the independent post-processing paper track:
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/docs/stem_micro_nature_methods_design.md`.

## LoCoAgg Signed-VI Frontier

- LoCoAgg uses anisotropic max-filter watershed fragments, exact tiled GPU RAG
  sufficient statistics, boundary plus local-competition features, one scorer,
  and one C++ dynamic agglomeration. The promoted v4 execution tile is fixed at
  `16x512x512`; tiles change memory use, not segmentation semantics.
- Hzzs30 raw pooled and hzzs31 local-competition pooled scorers fail from severe
  under-merging. Do not sweep their domain weights, seeds, architectures, or
  feature combinations.
- Hzzs32 replaces the heuristic target and alpha/count stop with exact signed
  local merge utility `Delta_VI = 2 H(B|G) - H(B)`, mapped so probability `0.5`
  is zero utility. It improves total VOI and ARAND on consumed M07/M09, but
  fails the predeclared M07 merge-VI guard. Keep it as mechanism evidence.
- Hzzs33 balances CREMI, SNEMI, MEC, AxonEM-H, and AxonEM-M without routing. It
  reduces AxonEM split VOI from `0.4195` to `0.2083` but raises merge VOI from
  `0.1112` to `0.4235`; false merges reject it as a transferable checkpoint.
- The remaining algorithmic problem is acquisition-stable merge ordering after
  components grow. The next admissible hypothesis is dirty-frontier component
  rescoring from maintained component/boundary sufficient statistics, trained
  with signed-VI utility, endpoint-local listwise regret, and group-worst loss.
  Retain `0.5` as the semantic zero. Confirm only on new untouched labels.

## SiFlow/TOS Habits

- Use the project TOS-bootstrap submitter for changliu jobs. Nodes may not see local `/volume` paths or public internet; bundle code, configs, wheelhouse assets, CREMI tarball, and checkpoints through TOS.
- Never print or commit SiFlow, TOS, GitHub, or Hugging Face credentials.
- For local network commands, source credentials only from secret files, then clear inherited proxy variables with `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy`. This is now the default fix for Git/HF/SiFlow/TOS stalls in this workspace.
- For this project, 2-GPU finetune+ABC-eval jobs are good for fast ablations; keep total active GPUs under the user's cap.
- If TOS result listing is slow, recover metrics from SiFlow logs by parsing dict rows containing `sample`, `backend`, and `voi_sum`, then write a local CSV under ignored `outputs/remote_eval_summaries/`.
- For postprocess exploration packs, prefer `scripts/submit_siflow_postprocess_experiments.py --pack stem-micro-pack-* --gpus-per-pod 8 --submit`. This runs four stage commands in one 8-GPU pod with `CUDA_VISIBLE_DEVICES=0,1`, `2,3`, `4,5`, `6,7`, reducing node fragmentation while preserving 2-GPU experiment slots.
- Inspect generated `run.sh` with `bash -n` before submitting any newly added pack stage. The pack wrapper uploads each sub-stage output to `tos://agi-data/users/dchen02/dbmim/outputs/<stage_output>` and pack logs to `tos://agi-data/users/dchen02/dbmim/outputs/<pack_name>`.
- When a pack contains optional dependency probes, make the probe write a JSON result and exit 0 for missing modules unless the task is specifically a dependency-install test. A missing optional backend should not mask successful segmentation stages.

## Repository Integration Discipline

- Preserve the original dbMiM root layout. Keep historical root-level configs,
  scripts, `dbmim/`, and training entrypoints because recorded commands and
  SiFlow bundles depend on them.
- Treat `mec8nm/` as the isolated, versionable MEC workspace. Apply shared
  runtime changes to both root and `mec8nm/` in one commit, then run
  `python mec8nm/scripts/verify_workspace.py` to reject mirror drift.
- Treat `dbmim-neuron-seg-skill/` as the repository skill source. Keep
  `mec8nm/skill/` and `$CODEX_HOME/skills/dbmim-neuron-seg/` byte-identical to
  it and validate both repository copies with `quick_validate.py`.
- Never commit HDF5 volumes, checkpoints, generated outputs, credentials,
  wheelhouses, or cache files. Version only split/manifest metadata and compact
  machine-readable evidence.
- Integrate through a branch based on current `origin/main`, run tests before
  merging, and use a normal merge commit. Do not rewrite or destructively reset
  the existing repository history to make mirrors agree.

## Large-scale EM Inference

- Current status is blockwise prototype, not PB-ready production. Full MEC
  blocks around `125x1250x1250` and CREMI/SNEMI-scale volumes run, but no
  end-to-end petabyte job has demonstrated bounded host memory, stable global
  IDs, cross-block quality, resumability, failure recovery, or storage cost.
- Do not run global WaterZ or one global dense RAG over PB data. Use chunked
  Zarr/TOS, deterministic chunk IDs, halo overlap, local hierarchy, sparse
  cross-block reconciliation, and final `uint64` IDs.
- Keep raw, calibrated affinity slabs, fragments, block metadata, and final IDs
  as separately versioned artifacts. Do not retain full logits plus affinities
  in memory. Add checksums and an append-only manifest so every chunk can be
  retried independently.
- LoCoAgg v4 bounds GPU allocation by tile at about 440 MiB on full SNEMI, but
  process RSS still grows to about 4.48 GiB. Exact v5 spill lowers RSS to about
  3.65--3.68 GiB and shrinks the partial-record workset 19--26x, but latency
  rises from about 2.28s to 4.01--4.08s because CPU fragments dominate. This is
  evidence of semantic streaming, not volume-independent memory.
- Before PB work, implement GPU fragment generation, slab-wise calibrated
  affinity storage, append-only partition spill/reduction, stable cross-block
  IDs, and overlap/reconciliation quality gates. Do not scale the current
  learned LoCoAgg checkpoint globally while AxonEM false merges remain open.
- Advance through fixed gates: single-block parity; `2x2x2` multi-block seam and
  stable-ID parity; `1--10 TB` resumability/I/O/throughput; `100 TB` failure
  recovery and cost; then PB. Report seam VOI/ARAND, split/merge VI,
  orphan/duplicate IDs, bytes/voxel, GPU-hours/TB, and peak RAM versus volume.
- A practical near-term large-volume baseline is conservative local WaterZ or
  MWS plus deterministic reconciliation. Blockwise affinity inference can run
  today, but globally coherent PB instance segmentation is not yet validated.
