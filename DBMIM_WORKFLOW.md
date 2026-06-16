# dbMiM Training Workflow

This repository now contains a self-contained dbMiM training path that does not
depend on the original private absolute paths.

## What Was Completed

- Added `dbmim/` with the core 3D patch MAE, shared actor-critic mask policy,
  gradient-structure reconstruction loss, affinity finetuning head, datasets,
  metrics, and checkpoint helpers.
- Added `train_pretrain.py` and `train_finetune.py`.
- Added smoke configs and full-run template configs under `configs/`.
- Added `scripts/download_data.py` for the FAFB Hugging Face dataset.
- Added `scripts/submit_siflow_dbmim.py` for TOS-bootstrap SiFlow submission.
- Added `scripts/evaluate_cremi_segmentation.py` and `dbmim/postprocess.py`
  for affinity-to-instance neuron segmentation, threshold sweep, adapted Rand,
  and VI metrics.

## Smoke Test

```bash
bash scripts/run_smoke.sh
```

Expected smoke outputs:

- `/volume/med-train/users/dchen02/code/dbMiM/outputs/pretrain_smoke/pretrained_latest.pt`
- `/volume/med-train/users/dchen02/code/dbMiM/outputs/finetune_smoke/finetuned_latest.pt`
- `/volume/med-train/users/dchen02/code/dbMiM/outputs/finetune_smoke/finetuned_best.pt`

## Data

The FAFB pretraining data is referenced from the Hugging Face dataset
`cyd0806/EM_pretrain_data`, subfolder `FAFB_hdf`. It is large and may require
gated Hugging Face access.

Dry-run manifest:

```bash
python scripts/download_data.py
```

Full download, after HF access is configured:

```bash
HF_TOKEN=... python scripts/download_data.py --allow-large-download
```

Do not commit tokens or generated large data.

## Full Jobs

Pretraining template:

```bash
python train_pretrain.py --config configs/pretrain_fafb.yaml
```

Finetuning template:

```bash
python train_finetune.py --config configs/finetune_cremi.yaml
```

SiFlow dry-run:

```bash
python scripts/submit_siflow_dbmim.py --stage smoke
python scripts/submit_siflow_dbmim.py --stage pretrain
python scripts/submit_siflow_dbmim.py --stage finetune
```

SiFlow submission requires explicit `--submit`.

## CREMI Instance Segmentation Postprocess

Finetuning optimizes nearest-neighbor z/y/x affinities. For neuron
segmentation metrics, run the postprocess/evaluation script to convert
affinities into instance labels and compute adapted Rand plus VI:

```bash
python scripts/evaluate_cremi_segmentation.py \
  --config configs/finetune_cremi_real.yaml \
  --checkpoint outputs/finetune_cremi_real_dbmim/finetuned_best.pt \
  --data-dir data/CREMI \
  --output-dir outputs/eval_cremi_real_dbmim \
  --crop-size 32 256 256 \
  --stride 16 128 128 \
  --thresholds 0.35 0.45 0.55 0.65 \
  --min-size 32 \
  --max-samples 3
```

The default postprocess is a dependency-light graph connected-components
baseline on the predicted affinity graph. It uses SciPy's sparse connected
components when available and falls back to a pure numpy union-find path.
This is deliberately safer for offline SiFlow pods than the old optional
`waterz`/`mahotas`/`elf` path.

To submit the CREMI eval through the same TOS bootstrap path:

```bash
python scripts/submit_siflow_dbmim.py \
  --stage eval-cremi \
  --resource-pool med-dev \
  --gpus-per-pod 1 \
  --submit
```

To avoid holding a GPU while finetuning is still running, launch the watcher
from the login node:

```bash
nohup bash scripts/watch_and_submit_eval.sh \
  > outputs/watchers/eval_submit_$(date +%Y%m%dT%H%M%S).log 2>&1 &
```

The watcher polls
`tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_dbmim/` and
submits `eval-cremi` as soon as `finetuned_best.pt` or `finetuned_latest.pt`
appears.
