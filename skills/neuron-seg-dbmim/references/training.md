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
