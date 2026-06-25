# dbMiM Neuron Segmentation

[![Chinese README](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-blue)](README_zh.md)
[![Hugging Face Weights](https://img.shields.io/badge/HuggingFace-weights-yellow)](https://huggingface.co/cyd0806/dbmim-neuron-segmentation)

This repository is the cleaned implementation used for our current dbMiM
neuron-segmentation experiments on CREMI. The maintained path is:

- self-supervised dbMiM / MAE-style pretraining on unlabeled EM volumes;
- anisotropic 3D UNETR affinity finetuning on CREMI;
- full-volume CREMI A/B/C evaluation with VOI and adapted Rand error (ARAND);
- waterz-based post-processing with calibration and threshold sweeps.

The old private cluster launchers, scratch reports, cached bytecode, legacy
models, and historical dataloaders have been removed from Git. Large data,
checkpoints, reports, and local experiment outputs are intentionally ignored.

## Current Method

The current best method is not the original minimal reproduction. It combines
the following changes that were stable in ablations:

1. **Anisotropic UNETR backbone**: `UNETRAnisotropicAffinityNet` with
   `32x160x160` input crops, `patch_size=(4,16,16)`, transformer hidden states
   used as UNETR skips, staged decoder upsampling, and a z-only anisotropic
   transition before the final decoder block.
2. **dbMiM pretraining on EM volumes**: a ViT/MAE encoder is pretrained with
   masked reconstruction, membrane-aware weighting, and a lightweight structure
   consistency loss. The pretrained encoder keys are then loaded into UNETR.
3. **MSE + MAWS supervised finetuning**: CREMI labels are converted to z/y/x
   nearest-neighbor affinities. Finetuning uses pure MSE with membrane-aware
   spatial weighting (MAWS), channel weights `[1.35, 1.0, 1.0]`, and synchronized
   image/label augmentations.
4. **Official-style waterz evaluation**: full CREMI A/B/C labeled volumes are
   evaluated with `ignore_label=0`, CREMI-style XY boundary ignore distance `1`,
   z boundary ignore `0`, logit calibration biases, and a waterz threshold sweep.

The most useful new finding is that **fixed mixed edge/random masking on full
EM data (R33)** improves over scratch, old fullEM dbMiM, pure edge masking, and
fullEM plain MAE. The best absolute VOI is still the smaller publicEM dbMiM
model (R17), so the README reports both.

The recommended R33 line does **not** use a reinforcement-learning masking
policy. It uses fixed mixed edge/random masking. RL-style decision modules are
retained only as ablations: R16 used the older decision module in the publicEM
pretraining line, while R34/R35 tested adaptive mixed masking and were negative
under the current CREMI A/B/C protocol.

## Model Zoo

Weights are hosted at:

**https://huggingface.co/cyd0806/dbmim-neuron-segmentation**

| Model | HF path | Intended use |
|---|---|---|
| PublicEM dbMiM R17 pretrain | `weights/publicem_dbmim_r17/pretrained_latest.pt` | ViT/dbMiM encoder checkpoint for UNETR initialization |
| PublicEM dbMiM R17 finetune | `weights/publicem_dbmim_r17/finetuned_latest.pt` | Best current publicEM segmentation checkpoint |
| FullEM mixed-mask dbMiM R33 pretrain | `weights/fullem_mixedmask_dbmim_r33/pretrained_latest.pt` | Recommended full-data dbMiM pretraining checkpoint |
| FullEM mixed-mask dbMiM R33 finetune | `weights/fullem_mixedmask_dbmim_r33/finetuned_latest.pt` | Recommended full-data segmentation checkpoint |

The pretraining checkpoints contain the masked-image-modeling encoder and
decoder state. During finetuning we load only compatible encoder prefixes
(`pos_embed`, `patch_embed`, `encoder_blocks`, `norm`) into the anisotropic
UNETR. The finetuned checkpoints are full affinity segmentation models.

## Data

### Supervised CREMI Data

Finetuning and evaluation use the public labeled CREMI 2016 training volumes:

```text
data/CREMI/sample_A_20160501.hdf
data/CREMI/sample_B_20160501.hdf
data/CREMI/sample_C_20160501.hdf
```

The raw key is `volumes/raw`; the instance-label key is
`volumes/labels/neuron_ids`.

### Pretraining Data

Two unlabeled EM pretraining sets were used.

| Name | Contents | Config examples |
|---|---|---|
| publicEM | CREMI raw + public ISBI 2012 + SNEMI3D raw volumes | `configs/pretrain_public_em_membrane_r16.yaml`, `configs/pretrain_public_em_plain_mae_r23.yaml` |
| fullEM | CREMI raw + `cyd0806/EM_pretrain_data` groups: FAFB, FIB-25, Kasthuri, MitoEM, MB-MOC | `configs/pretrain_em_full_mixedmask_dbmim_r33.yaml`, `configs/pretrain_em_full_plain_mae_r23.yaml` |

No hidden CREMI challenge labels are used in this repository.

### Evaluation Split

The reported numbers are **official-style validation on the public labeled
CREMI A/B/C training volumes**, not challenge-server hidden-test results.

The protocol is:

- train supervised affinity models from random crops sampled from CREMI A/B/C;
- run full-volume sliding-window inference on A, B, and C;
- apply CREMI-style boundary ignore with `xy=1`, `z=0`;
- sweep calibration biases and waterz thresholds;
- report aggregate A/B/C `voi_sum` and `adapted_rand_error`.

This split is small but matches the controlled ablation goal: isolate whether
dbMiM pretraining improves the same anisotropic UNETR finetuning recipe over
scratch and plain MAE controls.

## Results

Lower VOI and lower ARAND are better. `ARAND at best VOI` is the adapted Rand
error at the threshold selected by lowest VOI. `Best ARAND` is selected
independently and is included because VOI and ARAND can prefer different
post-processing thresholds.

### PublicEM Pretraining

| Arm | VOI | ARAND at best VOI | Best ARAND | Conclusion |
|---|---:|---:|---:|---|
| R17 publicEM random-mask dbMiM | **1.002919** | **0.188832** | 0.188832 | Best publicEM VOI |
| R23 publicEM random-mask plain MAE | 1.027073 | 0.192763 | 0.189247 | Matched MAE baseline |
| R29 publicEM pure edge-mask dbMiM | 1.033564 | 0.186827 | **0.186827** | Best publicEM ARAND, worse VOI |
| R32 publicEM fixed mixed-mask dbMiM | 1.046538 | 0.206256 | 0.193183 | Negative vs R17/R23 |
| R34 publicEM adaptive mixed dbMiM | 1.067471 | 0.205437 | 0.200604 | Negative adaptive result |
| R30 publicEM pure edge-mask plain MAE | 1.077594 | 0.203182 | 0.198562 | Edge-mask MAE control |
| R17 scratch UNETR | 1.095164 | 0.213401 | 0.210442 | Scratch control |

Key deltas:

- R17 dbMiM beats matched publicEM plain MAE R23 by `-0.0242` VOI and about
  `-0.0004` best ARAND.
- R29 edge-mask dbMiM beats same-mask plain MAE R30 by `-0.0440` VOI and
  `-0.0117` best ARAND, but its VOI is worse than R17/R23.

### FullEM Pretraining

| Arm | VOI | ARAND at best VOI | Best ARAND | Conclusion |
|---|---:|---:|---:|---|
| R33 fullEM fixed mixed-mask dbMiM | **1.039372** | **0.191216** | **0.190932** | Best fullEM result |
| R31 fullEM pure edge-mask dbMiM | 1.055438 | 0.195125 | 0.195125 | Positive but weaker than R33 |
| R20 fullEM old dbMiM | 1.085331 | 0.195722 | 0.195722 | Older fullEM baseline |
| R35 fullEM adaptive mixed dbMiM | 1.089639 | 0.205551 | 0.205551 | Negative vs R33/R31/R20 |
| R17 scratch UNETR | 1.095164 | 0.213401 | 0.210442 | Scratch control |
| R23 fullEM plain MAE | 1.440684 | 0.281216 | 0.281216 | Negative fullEM MAE baseline |

Key deltas:

- R33 fullEM mixed-mask dbMiM beats fullEM plain MAE R23 by `-0.4013` VOI and
  `-0.0903` best ARAND.
- R33 beats scratch by `-0.0558` VOI and `-0.0195` best ARAND.
- R33 beats old fullEM R20 by about `-0.0460` VOI.
- R33 is still slightly worse than R17 publicEM by VOI (`1.039372` vs
  `1.002919`), so the full-data recipe is the best fullEM result but not the
  best global checkpoint yet.

### Adaptive Masking

R34/R35 tested an adaptive mixed masking policy that chooses mask ratio and
edge fraction per crop. It did not improve downstream segmentation. After step
40k, the policy collapsed to sampled mask ratio `0.75`; the mean learned edge
fraction was `0.4456` for R34 and `0.3322` for R35. The current adaptive policy
is therefore kept as a negative ablation rather than the recommended method.

## Training Strategy

### Pretraining

Representative command:

```bash
python train_pretrain.py \
  --config configs/pretrain_em_full_mixedmask_dbmim_r33.yaml
```

Main settings:

| Setting | Value |
|---|---|
| Crop | `32x160x160` |
| Patch size | `4x16x16` |
| Encoder | ViT, `embed_dim=192`, `depth=6`, `heads=6` |
| Mask ratio | `0.75` |
| R33 mask strategy | `edge_random_mix`, `edge_mask_fraction=0.5`, `edge_mask_power=1.25` |
| dbMiM losses | reconstruction + structure loss `0.2` + membrane weighting `1.35` |
| Batch size | 2 per GPU |
| Schedule | 160k optimizer steps, AdamW, lr `1.5e-4`, weight decay `0.05`, AMP |

Plain MAE controls set `architecture: plain_mae`, `structure_weight: 0.0`, and
`membrane_weight: 0.0` while keeping data, crop, model size, mask ratio, and
schedule matched.

### Finetuning

Representative command:

```bash
python train_finetune.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_mixedmask_r33q.yaml
```

Main settings:

| Setting | Value |
|---|---|
| Backbone | `unetr_aniso_em` |
| Output | 3 affinity channels: z, y, x |
| Crop | `32x160x160` |
| Loss | MSE + MAWS, no BCE/Dice in the current winning recipe |
| Label handling | synchronized image/label augmentation, 2D border widening radius 1 |
| Batch size | 2 per GPU |
| Schedule | 12k optimizer steps, lr `8e-5`, encoder lr `1e-5`, weight decay `0.01`, AMP |
| Pretrained prefixes | `pos_embed`, `patch_embed`, `encoder_blocks`, `norm` |

### Evaluation

Representative full-volume command:

```bash
python scripts/evaluate_cremi_segmentation.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_mixedmask_r33q.yaml \
  --checkpoint outputs/finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_mixedmask_r33q/finetuned_latest.pt \
  --data-dir data/CREMI \
  --output-dir outputs/eval_cremi_r33_waterz_abc \
  --crop-size 0 0 0 \
  --stride 16 80 80 \
  --backends waterz \
  --thresholds 0.35 0.40 0.45 0.50 0.55 \
  --calibration-biases -0.50 -1.00 -1.00 -0.25 -0.50 -0.50 0.0 0.0 0.0 \
  --metric-backend skimage \
  --ignore-label 0 \
  --cremi-boundary-ignore-distance-xy 1 \
  --cremi-boundary-ignore-distance-z 0 \
  --max-samples 0 \
  --device cuda
```

The evaluation writes:

```text
cremi_segmentation_records.json
cremi_segmentation_metrics.csv
cremi_segmentation_summary.json
```

Use `best_by_voi_sum` for headline VOI and inspect `best_by_adapted_rand` as a
separate ARAND-selected operating point.

## Quick Start

Install the Python dependencies:

```bash
pip install -r requirements-dbMIM.txt
```

Run the synthetic smoke test:

```bash
bash scripts/run_smoke.sh
```

Compile the maintained entry points:

```bash
python -m py_compile \
  dbmim/*.py \
  train_pretrain.py \
  train_finetune.py \
  scripts/download_data.py \
  scripts/inspect_hdf5.py \
  scripts/prepare_public_em_pretrain_data.py \
  scripts/prepare_em_pretrain_data.py \
  scripts/evaluate_cremi_segmentation.py \
  scripts/evaluate_cremi_blockwise_scale.py
```

Download weights from Hugging Face with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="cyd0806/dbmim-neuron-segmentation",
    local_dir="outputs/hf_weights",
    allow_patterns=["weights/**", "configs/**"],
)
```

## Repository Layout

```text
dbmim/                         Core datasets, models, metrics, post-processing, utilities
configs/                       Maintained smoke, recommended, and matched ablation configs
scripts/download_data.py       CREMI download helper
scripts/prepare_*_data.py      PublicEM / fullEM pretraining data preparation helpers
scripts/evaluate_*.py          VOI/ARAND and blockwise-scale evaluation
train_pretrain.py              dbMiM / MAE pretraining entry point
train_finetune.py              affinity finetuning entry point
requirements-dbMIM.txt         Python dependency list
```

## Citation

```bibtex
@inproceedings{chen2023self,
  title={Self-supervised neuron segmentation with multi-agent reinforcement learning},
  author={Chen, Yinda and Huang, Wei and Zhou, Shenglong and Chen, Qi and Xiong, Zhiwei},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={609--617},
  year={2023}
}
```

## Data and Credential Notes

Use every external EM dataset under its original license and access policy.
Do not commit downloaded datasets, generated checkpoints, TOS credentials,
Hugging Face tokens, GitHub tokens, or cluster credentials.
