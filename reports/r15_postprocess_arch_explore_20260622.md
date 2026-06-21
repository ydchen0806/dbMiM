# R15 Post-Processing / Architecture Exploration

Date: 2026-06-22

## Motivation

The current CREMI pipeline still depends on CPU-heavy watershed/waterz
agglomeration. This is a real bottleneck because the network forward pass can
use H200 GPUs, while watershed, RAG construction, threshold sweeps, and waterz
priority-queue agglomeration are mostly CPU-bound.

The first R15 wave tests two complementary directions:

1. make post-processing cheaper at inference time;
2. move more of the post-processing objective into trainable UNETR outputs.

## Literature Signals

- MALIS optimizes affinities using maximin graph edges, so it is a direct
  precedent for training affinities against connectivity / adapted-Rand-like
  behavior rather than only local BCE or MSE.
- Mutex Watershed uses attractive and repulsive long-range graph cues, which
  suggests adding long-range affinity supervision to the UNETR head instead of
  relying only on nearest-neighbor z/y/x affinities.
- Dense voxel embedding / metric graph methods reduce segmentation to learned
  local metric decisions plus connected components, which is more GPU-friendly
  than waterz-style agglomeration.
- AGQ-style neuron segmentation avoids the conventional watershed +
  agglomeration stack and reports substantial inference acceleration; this is a
  stronger future rewrite, but R15 uses a smaller change that fits the current
  dbMiM/UNETR code.

## Implemented Code Changes

- `labels_to_offset_affinities` now supports arbitrary z/y/x offsets.
- `train.loss.affinity_offsets` lets a config train nearest-neighbor plus
  long-range affinities.
- `UNETREMAffinityNet` can output more than 3 channels. The first three remain
  z/y/x nearest-neighbor affinities for waterz compatibility; extra channels
  are auxiliary long-range cues.
- `evaluate_cremi_segmentation.py --affinity-channels` selects which model
  output channels are used as z/y/x post-processing affinities.
- Evaluation avoids redundant threshold products when explicit z/xy threshold
  grids are supplied.
- `submit_siflow_dbmim.py` has architecture-exploration stages and a
  `--post-train-arch-bench` hook that runs A/B/C post-processing comparison
  after each 12k-step quick training job.

## Submitted 8-GPU Wave

All tasks use `cn-shanghai/changliu`, pool `med-model`, instance `sci.g21-3`,
2 GPUs per pod.

| task | UUID | purpose |
|---|---|---|
| `dbmim-arch-explore-postprocess-r15q` | `bcac3b16-9896-4114-84bd-a70f854e2a8e` | Pure post-processing benchmark on the strongest R14q checkpoint: waterz vs seeded RAG vs CPU/GPU graph CC. |
| `dbmim-arch-explore-longaff-mempretrained-r15q` | `73a40fd4-287b-4bfc-98b6-c57aa6a38c1a` | Add long-range affinity offsets `(-2,0,0)`, `(0,-4,0)`, `(0,0,-4)` as auxiliary outputs. |
| `dbmim-arch-explore-longaff-lsd-mempretrained-r15q` | `08fad37f-4257-4f4f-9e9b-ad86c4b7f93f` | Long-range affinities plus lightweight LSD/shape descriptor auxiliary head. |
| `dbmim-arch-explore-longaff-bcar2-mempretrained-r15q` | `cb58f241-4fac-490f-b958-1ca6376bfb14` | Long-range affinities plus stronger BCAR rank loss (`0.10`, `8192` pairs, margin `1.2`). |

The three training tasks use MA-dbMiM R14 pretraining:

`tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_membrane_dbmim_r14/pretrained_latest.pt`

Each training task is 12k steps and then runs an A/B/C architecture benchmark:

- `graph_cc`
- `cupy_graph_cc` if the pod provides CuPy sparse graph support
- `seeded_rag`
- `waterz`

The benchmark records both segmentation quality and timing:

- `voi_sum`, `voi_split`, `voi_merge`
- `adapted_rand_error`
- `inference_sec`
- `postprocess_sec`
- `metrics_sec`

## Expected Readout

Use:

```bash
python scripts/poll_dbmim_tos_results.py --group r15q --once --logs --siflow-fallback
```

Interpretation:

- If GPU graph CC is much faster but VOI degrades badly, keep it only as a
  diagnostic / coarse screening backend.
- If seeded RAG is close to waterz with lower wall time, it is the safest
  near-term replacement because it preserves the watershed-fragment abstraction.
- If long-range affinity improves waterz and graph/RAG at the same time, it is
  the most paper-friendly architecture direction: the network is learning
  post-processing-compatible graph cues.
- If stronger BCAR helps ARAND but not VOI, frame it as merge-order stability,
  not universal segmentation quality.

## Follow-Up Options

The next step after R15q should depend on the first benchmark:

- If graph/CC is viable, implement a persistent GPU segmentation path and reduce
  waterz to final reporting only.
- If long-range affinity helps, convert the auxiliary channels into explicit
  attractive/repulsive heads and add a mutex-style loss.
- If neither helps, the bottleneck is probably the fragment generation and
  agglomeration abstraction itself; then the larger rewrite should be an
  embedding/metric-graph or AGQ-like segmentation head.
