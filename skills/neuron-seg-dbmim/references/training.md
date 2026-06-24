# Training Reference

## Maintained Architecture

Use `UNETRAnisotropicAffinityNet` in `dbmim/models.py`, selected by:

```yaml
model:
  architecture: unetr_aniso
```

This migrates the important anisotropic design from the old `model_unetr.py`:

- `patch_size: [4, 16, 16]`
- token grid such as `8x10x10` for `32x160x160`
- encoder skip projections from transformer hidden states
- decoder upsampling with 3/2/1 stages
- z-only `dtrans` between decoder3 and decoder2

The old simplified `UNETRAffinityNet` lacks this decoder structure and should
not be the main paper-aligned path.

## Pretrained Loading

The dbMiM pretrained checkpoint is a ViT/MAE encoder. When loading into UNETR:

- load only compatible backbone keys;
- interpolate `pos_embed` if the token grid differs;
- verify the printed `loaded_pretrained_keys` count;
- do not assume decoder weights exist in the pretrained checkpoint.

If pretrained and scratch runs look identical, check architecture first. A
simplified decoder can erase the intended inductive bias and make the
pretraining-effect comparison meaningless.

## Current Key Configs

- Long pretrain: `configs/pretrain_cremi_real_long.yaml`
- Public EM membrane pretrain:
  `configs/pretrain_public_em_membrane_r16.yaml`
- Aniso pretrained finetune:
  `configs/finetune_cremi_real_unetr_aniso_pretrained.yaml`
- Aniso scratch finetune:
  `configs/finetune_cremi_real_unetr_aniso_scratch.yaml`
- Long-pretrained finetune:
  `configs/finetune_cremi_real_unetr_aniso_longpretrained.yaml`
- Structural/loss ablations:
  - `finetune_cremi_real_unetr_aniso_no_dtrans.yaml`
  - `finetune_cremi_real_unetr_aniso_dtrans2.yaml`
  - `finetune_cremi_real_unetr_aniso_fs64.yaml`
  - `finetune_cremi_real_unetr_aniso_boundary_loss.yaml`
  - `finetune_cremi_real_unetr_aniso_context48.yaml`
- R16 public-EM matched controls:
  - `finetune_cremi_real_unetr_aniso_em_shwmse_longaff_publicem_r16q.yaml`
  - `finetune_cremi_real_unetr_aniso_em_shwmse_longaff_scratch_r16q.yaml`
  - `finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_publicem_r16q.yaml`
  - `finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_scratch_r16q.yaml`
- R17 MSE/MAWS matched controls:
  - `finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q.yaml`
  - `finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q.yaml`
  - `finetune_cremi_real_unetr_aniso_em_mse_maws_bcar_rank_publicem_r17q.yaml`
  - `finetune_cremi_real_unetr_aniso_em_mse_maws_bcar_rank_scratch_r17q.yaml`

For R16 evidence, compare matched publicEM-pretrained and scratch arms before
claiming a pretraining gain. The old all-CREMI or membrane-R14 pretrained arms
are useful baselines, but they do not isolate the new public-EM pretraining
effect.

For R17, the only intended changes versus R16 MAWS+BCAR are the supervised loss
family and optional BCAR. Keep data, crop, model, encoder LR, publicEM
checkpoint, and official A/B/C waterz eval fixed. This isolates whether MSE is
the better downstream affinity objective and whether BCAR helps in the MSE
regime.

R17 completed with the best current A/B/C VOI. The winning arm is
`UNETR-aniso-EM + MAWS + pure MSE + publicEM`, best VOI `1.002919`. The matched
scratch control is `1.095164`, so publicEM pretraining finally shows a stable
gain under the official A/B/C waterz protocol. R18 crosses this MSE loss with
long-affinity six-channel output to test whether the R16 ARAND advantage can be
combined with R17's VOI advantage.

R19 keeps the R17 winning recipe and changes only structure capacity:

- context `48x192x192` with publicEM/scratch controls;
- decoder `feature_size=48` with publicEM/scratch controls.

Both use per-GPU batch 1, 2 GPUs, 12k steps, pure MSE, MAWS, widened labels,
and the same official A/B/C waterz evaluation. This isolates whether bigger
context or decoder capacity improves EM segmentation and whether any gain
still depends on dbMiM pretraining.

Do not submit another "longer publicEM pretrain" unless there is new data or a
new pretraining objective. The current R16 publicEM checkpoint has already
reached `global_step=160000`. The larger HF `cyd0806/EM_pretrain_data` set is
manifested at about 486 GB, but it was not downloaded because no HF token is
available in the environment.

## Batch Size Semantics

`train.batch_size` is per rank/per GPU. Distributed jobs use:

```bash
python -m torch.distributed.run --nproc_per_node=8 ...
```

Thus:

- per-GPU batch 2 on 8 GPUs means global batch 16.
- per-GPU batch 1 on 8 GPUs means global batch 8.
- No gradient accumulation is currently configured.

Typical current settings:

| config class | per-GPU batch | global batch | crop | lr |
|---|---:|---:|---|---:|
| regular aniso finetune | 2 | 16 | `32x160x160` | `1e-4` |
| fs64/context48 | 1 | 8 | `32x160x160` or `48x192x192` | `8e-5` |
| long dbMiM pretrain | 2 | 16 | `32x160x160` | `1.5e-4` |
| R5 SuperHuman finetune | 2 | 8 on 4 GPUs | `32x160x160` | `8e-5` |

## Epoch and Checkpoint Timing

CREMI A/B/C with `length_multiplier: 4096` gives roughly 12288 random crop
samples. Finetune uses `val_fraction: 0.34`, so training has about 8110
samples.

Approximate steps:

- regular finetune global batch 16: about 507 optimizer steps per epoch.
- regular `save_every: 5`: first uploaded `finetuned_latest.pt` after about
  2535 optimizer steps if the job reaches save.
- pretrain global batch 16: about 768 steps per epoch; first
  `pretrained_latest.pt` after about 3840 steps.
- fs64/context48 global batch 8: about 1014 steps per epoch; first save after
  about 5070 steps.

If watcher logs still say `checkpoint_wait`, do not report convergence.

## Validation and Loss

Rank 0 prints JSON-like payloads with train and validation loss. Finetune loss
can include BCE, Dice, channel weights, and positive-edge weights from
`train.loss`.

Interpretation caveats:

- Affinity validation Dice/IoU can improve while instance segmentation ARAND/VOI
  does not.
- BCE-only loss may look low because most affinity edges are easy; inspect z/y/x
  thresholds and segmentation metrics.
- Boundary-heavy changes should be judged by VOI/ARAND after threshold sweep,
  not just by loss.

## Supervised Augmentation Contract

For instance segmentation finetuning, geometric augmentation must be shared by
the raw image and instance label. A 2026-06-20 audit found that older
`EMVolumeDataset.__getitem__` flipped only the image and attached the unflipped
label afterward. That makes affinity targets inconsistent with the input and
can produce deceptively low patch losses with unusable full-volume segmentation.

Current contract:

- use `augment_image_and_label(image, label)` for all random z/y/x flips;
- keep intensity gain/bias and Gaussian noise image-only;
- apply border widening to labels before generating affinity targets;
- test any future augmentation change with a synthetic label pattern and fixed
  RNG seed.

Do not interpret R3/R4 supervised results as method evidence unless the exact
bundle is known to include synchronized image/label flips.

## SuperHuman-Style R5 Loss

The R5 SuperHuman-aligned configs use:

```yaml
data:
  widen_border: true
  widen_border_radius: 1
train:
  replicate_affinity_boundary: true
  loss:
    loss_type: weighted_mse
    weight_alpha: 1.0
    dice_weight: 0.05
    boundary_dice_weight: 0.35
    channel_weights: [1.25, 1.0, 1.0]
```

`weighted_mse` computes sigmoid-affinity MSE and reweights positive/negative
edges per sample/channel by their binary ratio. The log field may still be
named `train_bce_loss` in older running bundles, but for R5 it is the main
weighted MSE term. Newer code also logs `train_main_loss`.

Current R5 ablation configs:

- `finetune_cremi_real_unetr_aniso_superhuman_nowiden_pretrained_r5.yaml`:
  disables `data.widen_border` to test whether SuperHuman/Kisuk-style 2D
  border invalidation is responsible for the gain.
- `finetune_cremi_real_unetr_aniso_superhuman_bce_pretrained_r5.yaml`:
  keeps synchronized augmentation and widened labels but replaces
  `weighted_mse` with BCE.
- `finetune_cremi_real_unetr_aniso_superhuman_boundaryhigh_pretrained_r5.yaml`:
  keeps weighted MSE and widened labels but raises `boundary_dice_weight` from
  `0.35` to `0.55`.
- `finetune_cremi_real_unetr_aniso_superhuman_encoderlr_pretrained_r5.yaml`:
  keeps the main R5 loss/data stack but trains the pretrained encoder with
  `2e-5` and decoder/head with `8e-5`.

When launching final eval watchers, wait for the explicit final step checkpoint
such as `checkpoint_step_00030000.pt`. Do not include `finetuned_latest.pt` in
`CHECKPOINT_FILES` for final comparisons because the bootstrap sync loop
uploads it periodically during training.

Current ablation lessons:

- No-widen is a failed ablation (`VOI_sum > 4.5`, `ARAND ~0.96` on sample A).
  Keep 2D border widening unless intentionally testing this failure mode.
- BCE pretrained R5 produced the best current raw-label sample-A VOI
  (`0.535448`), but not the best ARAND. It is a credible loss ablation to
  expand only after A/B/C checks.
- Because BCE pretrained can beat the main pretrained arm on raw-label and
  official-style sample-A VOI, always run the paired
  `superhuman-bce-scratch-r5` control before attributing the gain to
  pretraining. That control completed as SiFlow UUID
  `9f2bbe23-abfa-41c0-b00c-63f76fd76868`; final step 30000 logged
  `train_loss=0.147030`, `train_main_loss=0.091217`,
  `train_dice_loss=0.028496`, and `train_boundary_dice_loss=0.155396`.
  Paired evals completed on 2026-06-21: official sample-A
  `df6582ca-c87c-4843-a8ca-23e41030c3b5` reached best VOI `0.169502` and
  best ARAND `0.024062`; raw-label sample-A
  `4ce29ab6-4775-41f5-ac70-c2b62d94b9c4` reached best VOI `0.517256` and
  best ARAND `0.058831`. These beat the BCE-pretrained rows, so current BCE
  evidence supports the BCE loss choice rather than a pretraining gain.
- Boundary-high did not improve over the main R5 stack.
- The first `lowencoder` run was invalid because optimizer param grouping was
  built after DDP wrapping and names were prefixed by `module.`, so only the
  decoder group was matched. The corrected optimizer strips repeated
  `module.` prefixes before prefix matching and logs both tensor count and
  parameter count per group. Use the `encoderlr` config/output prefix, not the
  invalid `lowencoder` prefix.

## LSD-style Auxiliary Head

The R2 method branch adds a lightweight shape-descriptor auxiliary target,
inspired by funkelab LSD but not a full funlib local-moment implementation.
`labels_to_local_shape_descriptors` creates four target channels from instance
labels: foreground, dz, dy, dx centroid offsets. Configs with
`train.loss.lsd_weight > 0` must set `model.out_channels: 7`; the first three
channels remain z/y/x affinities and the last four are supervised descriptor
channels.

Rules:

- `train_finetune.py` must slice the first 3 channels before computing affinity
  BCE/Dice and affinity validation Dice/IoU.
- `scripts/evaluate_cremi_segmentation.py` must slice `logits[:, :3]` before
  sigmoid and post-processing. Otherwise VOI/ARAND will silently decode
  descriptor channels as affinities.
- Compare LSD-pretrained against LSD-scratch and pretrained-r2 against
  scratch-r2. Do not interpret a single LSD arm without the paired scratch
  control.

## Pre-submit Smoke Tests

Run:

```bash
python -m py_compile dbmim/models.py train_finetune.py train_pretrain.py \
  scripts/evaluate_cremi_segmentation.py scripts/submit_siflow_dbmim.py
```

For model changes, instantiate the exact crop and run a forward pass. Also test
the small `16x64x64` path if editing shape logic, because it catches many
anisotropic upsampling bugs cheaply.

## R13/R14 Method Lessons

The current paper-oriented branch is:

- `UNETREMAffinityNet` / `architecture: unetr_aniso_em` with anisotropic
  decoder and EM refinement depth.
- SuperHuman-style main loss:
  `loss_type: superhuman_weighted_mse`, `replicate_affinity_boundary: true`,
  `widen_border: true`, and `channel_weights: [1.35, 1.0, 1.0]`.
- `BCAR` for agglomeration-aligned supervision:
  `bcar_weight` ranks positive affinities above boundary affinities; optional
  calibration should be evaluated carefully because sample-A-only fallback can
  look misleadingly good.
- `MAWS` for membrane-aware weighted supervision:
  spatially reweight the pointwise affinity loss with a normalized raw-EM
  anisotropic membrane proxy. Keep `membrane_normalize: true`; logs should show
  `train_membrane_weight_mean` around `1.0`.
- `MA-dbMiM` for membrane-aware pretraining:
  masked reconstruction is weighted by a membrane/edge proxy and uses
  anisotropic structure gradients.

Quick R14q A/B/C lessons from 2026-06-21:

| run | best VOI | ARAND at best VOI | lesson |
|---|---:|---:|---|
| `bcar-rank-allpretrained-r14q` | 1.0818 | 0.1965 | BCAR rank-only baseline |
| `maws-allpretrained-r14q` | 1.0745 | 0.2011 | MAWS-only slightly improves VOI |
| `maws-bcar-rank-allpretrained-r14q` | 1.0407 | 0.1929 | best quick A/B/C result |
| `maws15-bcar-rank-allpretrained-r14q` | 1.0441 | 0.1973 | stronger MAWS was not better |

## R15 Architecture Exploration Lessons

R15 is the current fast architecture/method wave for EM-specific UNETR changes.
It keeps the R14 MA-dbMiM membrane-pretrained encoder and tests whether the
segmentation head/loss can better match anisotropic neuron boundaries.

Common R15 finetune settings:

- data: CREMI A/B/C HDF5, image keys `volumes/raw`, `raw`, `main`, labels
  `volumes/labels/neuron_ids`, `labels`, `label`, `gt`;
- crop: `32x160x160`, `length_multiplier: 2048`, synchronized image/label
  geometric augmentation, 2D border widening radius `1`;
- model: `architecture: unetr_aniso_em`, patch `[4,16,16]`, embed dim `192`,
  depth `6`, heads `6`, feature size `32`, dropout `0.05`,
  `em_refine_depth: 2`;
- optimization: per-GPU batch `2`, 2-GPU jobs use global batch `4`,
  `max_steps: 12000`, AMP on, `lr: 8e-5`, `encoder_lr: 1e-5`,
  `weight_decay: 0.01`;
- pretrained checkpoint:
  `outputs/pretrain_em_membrane_dbmim_r14/pretrained_latest.pt`, pulled from
  TOS inside SiFlow pods;
- main loss: `superhuman_weighted_mse`, `replicate_affinity_boundary: true`,
  membrane-weighted loss with normalized membrane proxy around mean `1.0`;
- affinity offsets: nearest z/y/x plus long-range channels `[-2,0,0]`,
  `[0,-4,0]`, `[0,0,-4]`. The first three channels must remain nearest
  z/y/x because the post-processing code decodes those as the main affinity
  graph.

Active R15 configs:

| config | channels | method delta |
|---|---:|---|
| `finetune_cremi_real_unetr_aniso_em_shwmse_longaff_mempretrained_r15q.yaml` | 6 | nearest + long-range affinity supervision |
| `finetune_cremi_real_unetr_aniso_em_shwmse_longaff_lsd_mempretrained_r15q.yaml` | 10 | six affinity channels plus four LSD-style foreground/offset channels, `lsd_weight: 0.15` |
| `finetune_cremi_real_unetr_aniso_em_shwmse_longaff_bcar2_mempretrained_r15q.yaml` | 6 | stronger BCAR rank term, `bcar_weight: 0.1`, margin `1.2`, `max_pairs: 8192` |

For LSD-style R15 heads, slice only the first six channels for affinity loss
and only the first three nearest-neighbor channels for standard graph/waterz
post-processing unless deliberately evaluating a long-range graph decoder. Do
not compare a 10-channel LSD checkpoint with a 3-channel eval path unless this
channel contract is verified in logs.

R15 should be judged by the post-train A/B/C architecture benchmark, not by the
12k-step training loss alone. The short jobs are for fast signal; promising
arms need a longer matched scratch/pretrained pair before becoming paper
evidence for dbMiM pretraining.

Do not use `bcar-calib-allpretrained-r14q` as an A/B/C conclusion unless its
summary has samples A/B/C and `n=3`; the first fallback parse only captured
sample A and gave an unrealistically low `VOI=0.3376`.

Full R14 follow-up submitted after MA-dbMiM pretraining completed:

| run | UUID | purpose |
|---|---|---|
| `em-shwmse-mempretrained-r14` | `f4c00499-19a5-4fb2-99a8-99adf54bad4d` | MA-dbMiM pretraining only |
| `em-shwmse-bcar-mempretrained-r14` | `d23472e9-567f-4bd3-9e04-24f450dbab85` | MA-dbMiM + original BCAR |
| `em-shwmse-maws-bcar-rank-mempretrained-r14` | `9494b4fa-f6e6-410c-90ba-052bb8e70d01` | current strongest combination |

If full R14 does not improve over scratch/CREMI-only pretraining, frame MAWS
and BCAR as supervised alignment gains and avoid claiming a dbMiM pretraining
benefit until all-EM pretraining data is actually available.

## R20/R21 Decoder-Aware And End-to-End Postprocess Lessons

R20 full-EM pretraining produced a useful but incomplete checkpoint:
`pretrain_em_full_membrane_dbmim_r20/pretrained_latest.pt` at step 60000. The
SiFlow task failed afterward, so downstream experiments using R20 should be
described as 60k-step full-EM pretraining, not final convergence.

The key hypothesis for R21 is that classic dbMiM only transfers the ViT/MAE
encoder keys into UNETR; decoder/head remain random. `DecoderAwareDBMIM3DMAE`
keeps the masked reconstruction objective but inherits `UNETREMAffinityNet`, so
its checkpoint contains compatible UNETR decoder/head keys as well:

- `encoder1/2/3/4`
- `decoder5/4/3/2`
- `dtrans`
- `head.shared`, `head.z_refine`, `head.xy_refine`, `head.z_head`,
  `head.xy_head`, `head.channel_bias`, `head.channel_scale`

During pretraining it also supervises an unsupervised membrane-derived affinity
proxy via sigmoid-MSE. The downstream loader already loads every shape-matched
key, so decoder-aware checkpoints should report more loaded keys than the
encoder-only 77-key publicEM/R20 checkpoints. If `loaded_pretrained_keys` stays
near 77 for an R21 downstream arm, inspect checkpoint key names before trusting
the result.

New configs:

- `configs/pretrain_em_full_decoderaware_r21.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r21q.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_decoderaware_r21q.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_dpp_scratch_r21q.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_decoderaware_dpp_r21q.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_dpp_fullem_r20q.yaml`

The differentiable postprocess path adds `LearnableAffinityPostProcessor`, a
small trainable logit calibration module with channel scale/bias plus local
depthwise smoothing residual. Training saves it in checkpoints under the
`postprocess` key. Its loss is not waterz itself; it is a continuous surrogate
over calibrated affinities:

- two-hop path consistency against the affinity target;
- foreground-smooth local affinity consistency;
- boundary contrast between positive and negative affinity logits.

Use DPP as an end-to-end calibration/postprocess experiment, not as a claim
that waterz became differentiable. The final segmentation still uses standard
waterz/threshold sweeps, but on logits that can include the learned calibration
module.

## Plain-MAE Control Requirement

Do not claim a dbMiM-specific pretraining benefit from `pretrained vs scratch`
alone. The current strongest positive signal (`R17 publicEM dbMiM +
MSE+MAWS`, VOI about `1.0029` vs scratch `1.0952`) proves that an EM-pretrained
initialization can help, but it does not by itself prove that the dbMiM
membrane/structure/decision objective is better than ordinary masked
autoencoding.

Use R23 for the clean control:

- publicEM dbMiM R16: `pretrain_public_em_membrane_dbmim_r16`, 160k steps;
- publicEM plain MAE R23: `pretrain_public_em_plain_mae_r23`, target 160k
  steps;
- matched downstream: `UNETR-aniso-EM + MSE + MAWS`, 12k finetune steps, same
  official A/B/C waterz sweep.

R24 is the current dbMiM++ attempt designed specifically to beat that R23
plain-MAE baseline under the same publicEM data and downstream recipe:

- `configs/pretrain_public_em_decoderaware_r24.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_decoderaware_r24q.yaml`
- pretrain stage `pretrain-public-em-decoderaware-r24`
- downstream stage
  `finetune-cremi-unetr-aniso-arch-explore-maws-mse-publicem-decoderaware-r24q`

R24 uses `DecoderAwareDBMIM3DMAE`: masked reconstruction plus anisotropic
gradient structure loss, membrane-weighted reconstruction, membrane-weighted
pseudo-affinity decoder/head loss, and decision masking. Its checkpoint should
load many more downstream keys than encoder-only MAE/dbMiM, including
UNETR-EM decoder/head keys. A local compatibility check loaded 214 keys into
the downstream UNETR-aniso-EM model, including encoder blocks, decoder modules,
and the affinity head.

Operational lesson from the first R24 run: the decision policy can become NaN
after the freeze point even when the main dbMiM losses remain finite. Keep
`use_frozen_policy_after_freeze: false` for long pretraining unless explicitly
testing policy transfer, and preserve the `DecisionModule` finite-value guard.
If R24 stops before 160k, resume from the latest `checkpoint_step_*.pt` instead
of treating the partial run as a method result.

Plain MAE means:

- encoder-only `DBMIM3DMAE`;
- random 75% mask;
- masked voxel reconstruction only;
- `structure_weight=0`;
- `membrane_weight=0`;
- `decision.enabled=false`;
- no decoder-aware affinity proxy.

For full-EM data, compare:

- R22 decoder-aware dbMiM / full-EM pretraining;
- R23 full-EM plain MAE, same data and 160k target;
- same downstream MSE+MAWS official A/B/C waterz evaluation.

Only the following interpretation is defensible:

| comparison | defensible claim |
|---|---|
| pretrained > scratch | EM pretraining helps |
| dbMiM > plain MAE > scratch | dbMiM objective adds value beyond ordinary MAE |
| plain MAE ≈ dbMiM > scratch | generic MAE/data initialization explains most of the gain |
| dbMiM ≈ plain MAE ≈ scratch | current pretraining claim is not stable |

Early 40k plain-MAE checkpoints are useful for debugging and trend checks, but
the paper-facing comparison should use the 160k checkpoint to match R16/R22.

As of 2026-06-23 15:40 CST, the completed publicEM comparison is:

- R16 publicEM dbMiM: `VOI=1.002919`, `best ARAND=0.188832`;
- R23 publicEM plain MAE: `VOI=1.027073`, `best ARAND=0.189247`;
- scratch: `VOI=1.095164`, `best ARAND=0.210442`.

This is a positive dbMiM-over-MAE signal, but weak: `-0.0242` VOI and
`-0.0004` best ARAND relative to plain MAE. R24 is the active attempt to make
the gain stronger; do not claim it until its downstream eval finishes.

R26 tests a specific R24 failure hypothesis. Because R24's decoder-aware
checkpoint is shape-compatible with the downstream UNETR-EM decoder and
affinity head, the default loader transfers about 214 keys, not just the MAE
encoder. If full R24 or R24 early3k is weak, do not immediately conclude that
the dbMiM++ encoder objective failed. First compare against R26:

- same R24 pretrained checkpoint;
- same downstream UNETR-aniso-EM + MSE+MAWS recipe;
- `pretrained_include_prefixes: [pos_embed, patch_embed, encoder_blocks, norm]`;
- decoder and affinity head initialized randomly as in R16/R23.

Interpretation:

| comparison | likely conclusion |
|---|---|
| R26 > R24 full-init | pseudo-affinity decoder/head transfer hurts downstream; keep encoder-only loading for paper comparisons |
| R24 >= R26 | decoder-aware initialization is not the issue; focus on pretraining objective/data |
| R26 <= R23 plain MAE | dbMiM++ objective still fails to beat plain MAE under this recipe |
| R26 > R23 plain MAE and R16 | decoder-aware dbMiM improves the encoder but should not transfer its head |

For the publicEM paper-facing comparison, use:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r24_dbmim_vs_mae --once --logs --siflow-fallback
```

Update after R24/R26 completion:

- R24 full decoder/head transfer is a negative result. Full 12k official A/B/C
  VOI is `1.482749`, much worse than R23 plain MAE `1.027073`, R16 dbMiM
  `1.002919`, and scratch `1.095164`.
- R26 encoder-only loading fixes most of the R24 damage (`VOI=1.050369`) but
  still does not beat R23 plain MAE or R16 dbMiM at 12k. This means the
  pseudo-affinity decoder/head transfer was harmful, but the R24 encoder
  objective is not strong enough for a `dbMiM++ > MAE` claim.
- R26 early3k has a small label-efficiency signal over R23 plain MAE
  (`1.553875` vs `1.566102` VOI, and `0.332548` vs `0.374662` best ARAND), but
  it is essentially tied with R16 dbMiM and scratch is still strong by ARAND.
  Do not build the paper claim on early3k alone.

Near-term submission priority:

1. Let fullEM plain MAE R23 finish as a matched baseline for R20/R22 full-data
   dbMiM. This is the highest-value missing comparison because it directly
   answers whether full-data dbMiM beats generic MAE.
2. Complete R18 no-BCAR long-affinity official A/B/C waterz eval. The stopped
   partial A/B evidence was not valid, but the question is cheap to close with
   1-GPU eval jobs.
3. The next dbMiM method attempt should be encoder-only at downstream load
   time. Design pretraining losses around membrane/affinity/boundary structure
   that improve the ViT encoder, then initialize the supervised UNETR decoder
   and affinity head randomly. Do not transfer pseudo-affinity decoder/head
   weights unless a small controlled eval proves they help.

## R29 Edge-biased Masking

R29 is the next encoder-only dbMiM attempt after the R24/R26 failure analysis.
It does not transfer any pseudo-affinity decoder/head. The pretraining change
is only in which patches are masked:

- baseline R23: random 75% mask, plain reconstruction;
- baseline R16: decision/membrane-weighted dbMiM with random or learned mask;
- R29: fixed 75% mask biased toward membrane/gradient-rich patches, plus the
  R16-style membrane-weighted reconstruction and gradient structure loss.

Implementation:

- `DBMIM3DMAE` supports `model.mask_strategy: edge_biased` with
  `edge_mask_power` and `edge_mask_noise`.
- Default behavior remains `random`, so existing configs are unchanged.
- R29 config: `configs/pretrain_public_em_edgemask_dbmim_r29.yaml`.
- Downstream config:
  `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_edgemask_r29q.yaml`.

R29 should be judged only against matched publicEM R23 plain MAE, R16 dbMiM,
and scratch under the same R17 MSE+MAWS 12k official A/B/C waterz protocol.
Do not use pretrain loss alone as evidence.
