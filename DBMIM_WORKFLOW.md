# dbMiM CREMI Workflow

This is the maintained workflow for the current dbMiM neuron segmentation
experiments. It is deliberately narrower than the historical repository notes:
anisotropic UNETR backbone, z/y/x affinity finetuning, VOI and ARAND
evaluation, and one main pretraining-effect ablation.

## 1. Pretrain

Train the dbMiM masked-image model on CREMI EM volumes:

```bash
python train_pretrain.py --config configs/pretrain_cremi_real.yaml
```

SiFlow:

```bash
python scripts/submit_siflow_dbmim.py \
  --stage pretrain-cremi \
  --resource-pool auto \
  --gpus-per-pod 8 \
  --submit
```

Expected checkpoint:

```text
tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_dbmim/pretrained_latest.pt
```

For the longer paper-aligned pretraining run at the original UNETR crop size:

```bash
python scripts/submit_siflow_dbmim.py \
  --stage pretrain-cremi-long \
  --resource-pool auto \
  --gpus-per-pod 8 \
  --submit
```

Expected checkpoint:

```text
tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_long_dbmim/pretrained_latest.pt
```

## 2. Finetune Anisotropic UNETR

Run the controlled pretrained-vs-scratch comparison:

```bash
python train_finetune.py \
  --config configs/finetune_cremi_real_unetr_aniso_pretrained.yaml

python train_finetune.py \
  --config configs/finetune_cremi_real_unetr_aniso_scratch.yaml
```

SiFlow:

```bash
python scripts/submit_siflow_dbmim.py \
  --stage finetune-cremi-unetr-aniso-pretrained \
  --resource-pool auto \
  --gpus-per-pod 8 \
  --submit

python scripts/submit_siflow_dbmim.py \
  --stage finetune-cremi-unetr-aniso-scratch \
  --resource-pool auto \
  --gpus-per-pod 8 \
  --submit
```

Expected checkpoint prefixes:

```text
tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained/
tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch/
tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_longpretrained/
```

Current R2 restart stages use the same anisotropic UNETR backbone with shorter
decision-making runs and periodic TOS checkpoint sync:

```bash
python scripts/submit_siflow_dbmim.py \
  --stage finetune-cremi-unetr-aniso-pretrained-r2 \
  --resource-pool med-model \
  --gpus-per-pod 8 \
  --submit

python scripts/submit_siflow_dbmim.py \
  --stage finetune-cremi-unetr-aniso-scratch-r2 \
  --resource-pool med-model \
  --gpus-per-pod 8 \
  --submit
```

The LSD-style auxiliary variants add a 4-channel descriptor head on top of the
3 affinity channels. Evaluation still uses only the first 3 channels:

```bash
python scripts/submit_siflow_dbmim.py \
  --stage finetune-cremi-unetr-aniso-lsd-pretrained-r2 \
  --resource-pool med-model \
  --gpus-per-pod 8 \
  --submit

python scripts/submit_siflow_dbmim.py \
  --stage finetune-cremi-unetr-aniso-lsd-scratch-r2 \
  --resource-pool med-model \
  --gpus-per-pod 8 \
  --submit
```

## 3. Evaluate VOI and ARAND

Local command shape:

```bash
python scripts/evaluate_cremi_segmentation.py \
  --config configs/finetune_cremi_real_unetr_aniso_pretrained.yaml \
  --checkpoint outputs/finetune_cremi_real_unetr_aniso_pretrained/finetuned_best.pt \
  --data-dir data/CREMI \
  --output-dir outputs/eval_cremi_unetr_aniso_pretrained \
  --crop-size 32 320 320 \
  --stride 16 80 80 \
  --thresholds 0.0 \
  --backends graph_cc cupy_graph_cc \
  --min-size 32 \
  --z-thresholds 0.45 0.55 0.65 0.75 0.85 \
  --xy-thresholds 0.65 0.75 0.85 0.90 0.95 \
  --max-samples 3 \
  --device cuda
```

SiFlow:

```bash
python scripts/submit_siflow_dbmim.py \
  --stage eval-cremi-unetr-aniso-pretrained \
  --resource-pool auto \
  --gpus-per-pod 1 \
  --submit

python scripts/submit_siflow_dbmim.py \
  --stage eval-cremi-unetr-aniso-scratch \
  --resource-pool auto \
  --gpus-per-pod 1 \
  --submit
```

Primary metrics:

```text
adapted_rand_error   # ARAND, lower is better
voi_split
voi_merge
voi_sum              # voi_split + voi_merge, lower is better
```

Main output files:

```text
cremi_segmentation_metrics.csv
cremi_segmentation_records.json
cremi_segmentation_summary.json
```

Use `best_by_adapted_rand` in the summary for the headline ARAND row, and
report VOI from the same row. Also inspect `best_by_voi_sum`; this catches
bad all-merge or over-merge rows that can otherwise hide behind a single
selection metric.

## 4. Notes on Post-processing

The stable production backend is affinity-graph connected components with a
z/xy threshold sweep. CPU SciPy sparse graph CC remains the most reliable
default; the CuPy graph backend is kept as a speed probe but did not show a
stable speedup in earlier CREMI crops.

The old `waterz`/`elf`/`mahotas` path is useful as a negative control only:
it is slower or unavailable in offline pods and has not beaten the graph-CC
baseline in the current runs.

External connectomics codebases worth tracking:

- PyTorch Connectomics for the explicit train/infer/decode/evaluate split and
  decode parameter sweeps.
- funkelab LSD for instance-label-derived shape descriptor supervision. This
  repository currently implements only a lightweight centroid-offset
  LSD-style auxiliary target to avoid adding fragile offline dependencies.
- `affogato`/mutex watershed and GASP-style agglomeration for future
  post-processing experiments after dependency packaging is stable.

## 5. Finished Baselines

Historical MAE-head, simplified UNETR, and loss-objective ablations are tracked
in `EXPERIMENTS.md`. They are not the current paper-aligned anisotropic UNETR
comparison.
