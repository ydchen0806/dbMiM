---
name: dbmim-neuron-seg
description: Use when reproducing, monitoring, improving, or reporting dbMiM neuron segmentation experiments on CREMI with anisotropic UNETR, VOI/ARAND, waterz post-processing, publicEM/fullEM pretraining, MAE/dbMiM/RL mask-policy ablations, SiFlow/TOS jobs, or large-scale EM inference.
---

# dbMiM Neuron Segmentation

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
- 2026-07-02 proxy/network pitfall: local env proxy variables pointing to `192.168.32.28:18000` can make GitHub and SiFlow SDK calls time out. Direct access worked after unsetting `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, `all_proxy`, and `ALL_PROXY`; use temporary unsets or `git -c http.proxy= -c https.proxy=` for pushes. Do not bake proxy settings into repo scripts.
- 2026-07-02 post-processing literature-driven direction: waterz/MWS/GASP all point to hierarchy or signed-graph decisions, not one hard binary merge label. The next learned-postprocess branch is waterz-ladder distillation: generate teacher segmentations from multiple waterz thresholds in one hierarchy call, convert each to RAG edge labels, and train a soft merge target equal to the fraction of thresholds that merge each edge. This should reduce the over-merge failure seen with single-threshold `0.20` distillation.
- Longer-term postprocess directions worth testing after ladder distillation: add signed repulsive/cannot-link edges from long-range affinities, add an LSD/shape-context head or fragment descriptors for merge scoring, and optimize blockwise fragment/RAG construction because full-volume GPU connected components is no longer the dominant runtime.
- 2026-07-02 ladder distillation result: standard waterz ladder `[0.20,0.24,0.28,0.32]` trained fast and made sample A look strong, but ABC oracle best was only `VOI=1.4161`, `ARAND=0.3613`, merge `9.17s/volume` (`A=0.3434`, `B=1.7193`, `C=2.1855`). This is faster than waterz merge but far below waterz quality (`VOI~0.98` global), so do not use MLP waterz-ladder RAG as the main post-processing claim. High-threshold ladder increased teacher positive fraction and was stopped; low-threshold ladder reduced positive fraction but A degraded and was stopped. Next credible route is signed graph / MWS / GASP-style repulsive edges or waterz blockwise/parallel acceleration, not more scalar-threshold ladder tuning.
- 2026-07-03 waterz scoring alias sweep is confirmatory/negative: `hist_quantile` remains the only robust option tested. On R48 ABC, best deployable single config was `hist_quantile`, threshold `0.20`, bias `(-0.25,-0.50,-0.50)`, `VOI=0.962154`, `ARAND=0.178252`, merge about `15.38s/volume`; `min` and `max` scoring were much worse (`VOI` often around `5-7`). Do not spend more GPU time on waterz scoring aliases unless a genuinely new scorer implementation is added.
- 2026-07-03 signed-reweight learned RAG narrow sweep: subtracting a small signed boundary penalty is useful but not yet waterz-comparable. With thresholds only up to `0.96`, best ABC row was subtract alpha `0.15`, threshold `0.96`, `VOI=1.330421`, `ARAND=0.305353`, merge `9.46s/volume`; same-grid baseline at threshold `0.96` was `VOI=1.406976`, `ARAND=0.366906`. The gain mainly came from sample B (`2.0445 -> 1.5620` at threshold `0.96`), while sample C did not improve (`1.8511` baseline vs `1.8568` best signed row). This is a mechanism signal, not a final headline, because the older learned-RAG table included threshold `0.98` and got `VOI~1.3001`.
- 2026-07-03 signed high-threshold follow-ups are negative for VOI. `arch-explore-r48-signed-reweight-highthr` (`3cfa857a-1f35-44e5-9f2f-e50adb45e784`) completed: best VOI was still the plain learned-RAG GPU baseline at threshold `0.98`, `VOI=1.300060`, `ARAND=0.310392`; the best ARAND row, subtract alpha `0.025` at threshold `0.99`, improved ARAND to `0.233119` but worsened VOI to `1.358886`, mainly by hurting sample C. `arch-explore-r48-signed-veto-highthr` (`5cd248d2-ce11-471a-a587-0f8453e4a8b0`) completed and was effectively a no-op: best `VOI=1.299683`, `ARAND=0.310348`, only `0.0004` VOI better than the baseline and not a meaningful improvement. Stop scalar signed-veto/reweight tuning unless the model predicts true long-range repulsive edges.
- 2026-07-03 current learned-postprocess experiment: corrected `arch-explore-r48-rag-morphctx` (`03cde2c9-5b1a-441b-9d61-842ae5173df5`) runs a 2-GPU same-pod A/B/C full-volume comparison between old basic fragment context and new morphology context. The new `--context-feature-set morphology` adds fragment centroids, centroid deltas, boundary density, and boundary-mean interaction features to the learned RAG scorer while preserving the old `basic` feature set for control. The first morphology submission `fda63524-b5ff-4fa6-8c50-44333e0e10df` failed before training because the private submitter omitted `.yaml` from the config path; always inspect generated `run.sh` or do a dry-run before submitting newly added private stages.
- 2026-07-03 literature check for next post-processing work: Mutex Watershed/MWS is the cleanest algorithmic match because it explicitly combines attractive short-range and repulsive long-range cues without a waterz-like scalar threshold. LSD/shape descriptors are a strong feature-design prior for RAG scoring because they encode local morphology beyond boundary affinity. Connectivity-consensus / multimodal agglomeration papers support adding fragment morphology, raw/embedding context, and graph neighborhood features instead of relying on pairwise boundary mean alone. For MICrONS/CAVE scale, treat blockwise hierarchy and cross-block sparse reconciliation as part of the method, not just engineering.

## SiFlow/TOS Habits

- Use the project TOS-bootstrap submitter for changliu jobs. Nodes may not see local `/volume` paths or public internet; bundle code, configs, wheelhouse assets, CREMI tarball, and checkpoints through TOS.
- Never print or commit SiFlow, TOS, GitHub, or Hugging Face credentials.
- For this project, 2-GPU finetune+ABC-eval jobs are good for fast ablations; keep total active GPUs under the user's cap.
- If TOS result listing is slow, recover metrics from SiFlow logs by parsing dict rows containing `sample`, `backend`, and `voi_sum`, then write a local CSV under ignored `outputs/remote_eval_summaries/`.

## Large-scale EM Inference

- For MICrONS/CAVE-scale inference, plan blockwise Zarr/TOS outputs with halo overlap and deterministic chunk IDs. Do not assume global waterz over the entire volume is feasible.
- Store affinity maps, membrane/boundary maps, and block metadata separately from final IDs. This allows re-agglomeration without rerunning UNETR.
- Prefer hierarchical post-processing: local block segmentation, overlap reconciliation, then sparse cross-block agglomeration. Use full waterz sweeps on CREMI-like validation blocks, not on every production block.
- Benchmark throughput as GPU inference time plus CPU post-processing time. A segmentation method that improves VOI but multiplies CPU agglomeration cost may fail at CAVE scale.
