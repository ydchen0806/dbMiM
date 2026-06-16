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
