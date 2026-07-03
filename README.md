# dbMiM Neuron Segmentation

[中文](README_zh.md) | [Hugging Face weights](https://huggingface.co/cyd0806/dbmim-neuron-segmentation)

Official implementation for dbMiM pretraining and CREMI neuron segmentation.
The maintained workflow is:

1. prepare unlabeled EM volumes for self-supervised pretraining;
2. run dbMiM / MAE-style masked-image pretraining;
3. finetune an anisotropic 3D UNETR affinity model on CREMI;
4. evaluate full CREMI A/B/C volumes with VOI and adapted Rand error (ARAND);
5. decode instances with the reference waterz post-processing backend.

Learnable / differentiable post-processing is developed separately at
https://github.com/ydchen0806/nnEM-Seg-diff-postprocess.

## Method

The segmentation model is `UNETRAnisotropicAffinityNet`.

- Input crop: `32 x 160 x 160`
- Patch size: `(4, 16, 16)`
- Output: z/y/x nearest-neighbor affinity logits
- Backbone: ViT encoder initialized from dbMiM pretraining
- Decoder: UNETR-style staged upsampling with an anisotropic z transition
- Finetuning loss: MSE + membrane-aware spatial weighting (MAWS)
- Evaluation: full-volume CREMI A/B/C inference, `ignore_label=0`, boundary ignore `xy=1, z=0`

dbMiM pretraining masks 3D ViT patches and reconstructs EM voxels with
membrane-aware weighting and a lightweight structure-consistency term. Plain MAE
controls use the same data, model size, crop size, mask ratio, and schedule with
the dbMiM-specific terms disabled.

## Results

Lower is better for both VOI and ARAND.

| Run | Checkpoint | VOI | ARAND | Note |
|---|---|---:|---:|---|
| R48 | `weights/publicem_dbmim_r48_seed309_long20k/finetuned_latest.pt` | **0.962154** | 0.178252 | Best VOI |
| R57 | `weights/publicem_dbmim_r57_seed777_long20k/finetuned_latest.pt` | 0.964617 | **0.178248** | Best ARAND in repeat sweep |
| R33 | `weights/fullem_mixedmask_dbmim_r33/finetuned_latest.pt` | 1.039372 | 0.205380 | Best fullEM recipe |

Validation is run on public labeled CREMI A/B/C training volumes, not hidden
challenge labels.

## Weights

Weights are hosted on Hugging Face:

https://huggingface.co/cyd0806/dbmim-neuron-segmentation

| File | Use |
|---|---|
| `weights/publicem_dbmim_r48_seed309_long20k/finetuned_latest.pt` | Recommended segmentation checkpoint |
| `weights/publicem_dbmim_r57_seed777_long20k/finetuned_latest.pt` | ARAND-best repeat checkpoint |
| `weights/publicem_dbmim_r17/pretrained_latest.pt` | PublicEM dbMiM encoder pretraining checkpoint |
| `weights/publicem_dbmim_r17/finetuned_latest.pt` | Earlier publicEM finetuned checkpoint |
| `weights/fullem_mixedmask_dbmim_r33/pretrained_latest.pt` | FullEM mixed-mask pretraining checkpoint |
| `weights/fullem_mixedmask_dbmim_r33/finetuned_latest.pt` | FullEM mixed-mask finetuned checkpoint |

Download example:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="cyd0806/dbmim-neuron-segmentation",
    local_dir="weights/dbmim-neuron-segmentation",
)
```

## Setup

```bash
git clone https://github.com/ydchen0806/dbMiM.git
cd dbMiM
python -m pip install -r requirements-dbMIM.txt
```

The waterz reference backend is optional for training but required for the
reported instance-segmentation metrics.

## Data

CREMI finetuning/evaluation expects the public CREMI 2016 training files under:

```text
data/CREMI/sample_A_20160501.hdf
data/CREMI/sample_B_20160501.hdf
data/CREMI/sample_C_20160501.hdf
```

Prepare public EM pretraining data:

```bash
python scripts/prepare_public_em_pretrain_data.py \
  --target-dir data/EM_pretrain_data
```

Prepare the larger fullEM pretraining set:

```bash
HF_TOKEN=<your_token> python scripts/prepare_em_pretrain_data.py \
  --target-dir data/EM_pretrain_data
```

## Pretraining

PublicEM dbMiM pretraining:

```bash
python train_pretrain.py \
  --config configs/pretrain_public_em_membrane_r16.yaml
```

FullEM mixed-mask pretraining:

```bash
python train_pretrain.py \
  --config configs/pretrain_em_full_mixedmask_dbmim_r33.yaml
```

## Finetuning

Recommended R48 finetuning recipe:

```bash
python train_finetune.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r16_seed309_long20k_r48q.yaml
```

The config points to the pretrained encoder checkpoint. Update the path if the
weights are stored outside `outputs/`.

## Evaluation

Run full-volume CREMI A/B/C waterz evaluation:

```bash
python scripts/evaluate_cremi_segmentation.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r16_seed309_long20k_r48q.yaml \
  --checkpoint outputs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r16_seed309_long20k_r48q/finetuned_latest.pt \
  --data-dir data/CREMI \
  --output-dir outputs/eval_r48_cremi_abc \
  --crop-size 0 0 0 \
  --stride 16 80 80 \
  --backends waterz \
  --thresholds 0.16 0.18 0.20 0.22 0.24 \
  --calibration-biases -0.25 -0.50 -0.50 \
  --seed-method maxima_distance \
  --seed-distance 10 \
  --boundary-threshold 0.5 \
  --waterz-scoring hist_quantile \
  --batched-waterz \
  --metric-backend skimage \
  --ignore-label 0 \
  --cremi-boundary-ignore-distance-xy 1 \
  --cremi-boundary-ignore-distance-z 0 \
  --device cuda
```

`--batched-waterz` evaluates all waterz thresholds for each affinity variant in
one waterz hierarchy pass. It keeps the reported R48 VOI unchanged
(`0.962154`) and reduces threshold-sweep post-processing time from about
`75s` to about `17s` on CREMI A/B/C.

The summary is written to:

```text
outputs/eval_r48_cremi_abc/cremi_segmentation_summary.json
```

## Repository Layout

```text
dbmim/                         Models, datasets, metrics, utilities
configs/                       Pretraining and finetuning configs
scripts/prepare_*_data.py       Data preparation
scripts/evaluate_*.py           CREMI evaluation
train_pretrain.py               dbMiM / MAE pretraining
train_finetune.py               CREMI affinity finetuning
```

Large datasets, checkpoints, and generated outputs are not tracked in Git.
