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

## Model And Loss Recipe

- Backbone should be `unetr_aniso_em`, not generic isotropic UNETR. Use crop `32x160x160`, patch size `4x16x16`, transformer skips, staged decoder upsampling, and z-only anisotropic transition before the final decoder block.
- Finetune labels are z/y/x nearest-neighbor affinities from CREMI instance labels.
- The stable supervised loss is pure MSE plus MAWS. Keep `membrane_weight=0.75` for the main line; `membrane_weight=1.0` improved ARAND in R50 but did not beat R48 VOI.
- Use `encoder_lr=1e-5`, base lr `8e-5`, weight decay `0.01`, AMP, batch size 2 per GPU. R48 shows that extending from 12k to 20k steps can matter.

## Mask-Policy Lessons

- Do not equate "more edge reward" with better dbMiM. R38/R39 and strong hard-edge rewards were negative downstream.
- R51 was unhealthy: mask ratio stuck around 0.55, edge fraction 1.0, edge coverage 0.0, policy loss frozen to zero. Treat it as a negative control.
- R52 is the healthier RL-style policy design: constrained mask-ratio bins, constrained edge-fraction bins, small edge-proxy reward, KL-to-prior regularization, clipped/normalized advantages, and policy freeze after warmup. Do not claim it as an improvement until R52/R53 finetunes finish.
- Stable dbMiM gains currently come from the combination of EM-aware masking, membrane weighting, anisotropic UNETR, and enough finetune steps; the policy module must be validated through downstream VOI/ARAND, not pretraining loss alone.

## Post-processing Rules

- waterz remains the reference post-processing backend for headline numbers. It is CPU-heavy; first low-threshold agglomeration can take about 90s per full volume and later thresholds are typically about 25-30s per point.
- Keep threshold/calibration sweeps small on GPU pods to avoid wasting expensive H200 time on CPU-only agglomeration. For broad sweeps, use separate CPU jobs or sparse candidate points from previous best rows.
- A fast learned or differentiable post-process is useful only if it is robust and comparable to waterz under the same ABC grouping. Graph connected components and simple thresholding are speed baselines, not waterz replacements.
- Calibration biases around `(z=-0.25,y=-0.5,x=-0.5)` and `(z=-0.5,y=-1.0,x=-1.0)` are recurring useful regions. Always verify per run.

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
