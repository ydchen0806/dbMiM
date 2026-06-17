# dbMiM Neuron Segmentation

This repository is maintained around the current paper-aligned CREMI
reproduction path:

- Backbone: 3D UNETR affinity segmentation model.
- Pretraining: dbMiM masked-image modeling on EM volumes with the same ViT
  encoder keys used by the UNETR finetuning model.
- Finetuning target: z/y/x nearest-neighbor affinities.
- Evaluation: instance segmentation on CREMI with VOI and adapted Rand error
  (ARAND).
- Main question: does dbMiM pretraining improve UNETR finetuning over the same
  UNETR architecture trained from scratch?

The old private-path scripts and historical notes are not the maintained entry
points. Use the files below for new experiments:

```text
dbmim/models.py                         # DBMIM3DMAE and UNETRAnisotropicAffinityNet
train_pretrain.py                       # dbMiM pretraining
train_finetune.py                       # affinity finetuning
scripts/evaluate_cremi_segmentation.py  # VOI/ARAND evaluation
scripts/submit_siflow_dbmim.py          # TOS-bootstrap SiFlow jobs
configs/pretrain_cremi_real.yaml
configs/pretrain_cremi_real_long.yaml
configs/finetune_cremi_real_unetr_aniso_pretrained.yaml
configs/finetune_cremi_real_unetr_aniso_scratch.yaml
configs/finetune_cremi_real_unetr_aniso_longpretrained.yaml
```

## Experiment Design

The primary ablation is controlled and intentionally narrow:

| Arm | Backbone | Initialization | Config |
|---|---|---|---|
| Anisotropic UNETR + dbMiM pretrain | `UNETRAnisotropicAffinityNet` | CREMI dbMiM pretrained ViT encoder | `configs/finetune_cremi_real_unetr_aniso_pretrained.yaml` |
| Anisotropic UNETR scratch | `UNETRAnisotropicAffinityNet` | random initialization | `configs/finetune_cremi_real_unetr_aniso_scratch.yaml` |
| Anisotropic UNETR + long dbMiM pretrain | `UNETRAnisotropicAffinityNet` | longer CREMI dbMiM pretrain at `32x160x160` | `configs/finetune_cremi_real_unetr_aniso_longpretrained.yaml` |

Both arms use the same data, crop size, optimizer, loss, decoder, postprocess
grid, and evaluation script. The only intended difference is whether the UNETR
transformer encoder starts from the dbMiM pretrained checkpoint.

The maintained UNETR path uses the original anisotropic decoder from
`model_unetr.py`: `patch_size=(4,16,16)`, UNETR skip upsampling by 3/2/1 stages,
and a z-only `dtrans` convolution with stride `patch_y / patch_z` before the
final decoder block. This is exposed as `architecture: unetr_aniso`.

Evaluation reports:

- `adapted_rand_error`: ARAND, lower is better.
- `voi_split`, `voi_merge`, `voi_sum`: variation of information, lower is
  better.
- `rand_fscore`: included for diagnostics, higher is better.

## Local Checks

Run a synthetic smoke test before touching real jobs:

```bash
bash scripts/run_smoke.sh
```

Compile the maintained Python entry points:

```bash
python -m py_compile \
  dbmim/models.py \
  train_pretrain.py \
  train_finetune.py \
  scripts/evaluate_cremi_segmentation.py \
  scripts/submit_siflow_dbmim.py
```

Check that the UNETR finetune model can load the pretrained ViT encoder:

```bash
python - <<'PY'
import torch
from dbmim.utils import load_checkpoint
from dbmim.models import UNETRAnisotropicAffinityNet, load_pretrained_backbone

model = UNETRAnisotropicAffinityNet(
    in_channels=1,
    out_channels=3,
    volume_size=(32, 160, 160),
    patch_size=(4, 16, 16),
    embed_dim=192,
    depth=6,
    num_heads=6,
    feature_size=32,
)
y = model(torch.randn(1, 1, 32, 160, 160))
print({"output_shape": tuple(y.shape)})
ckpt = load_checkpoint("outputs/final_weights/pretrained_latest.pt", map_location="cpu")
print({"loaded_encoder_keys": len(load_pretrained_backbone(model, ckpt))})
PY
```

## SiFlow Jobs

SiFlow jobs use TOS bootstrap because changliu pods should not rely on local
paths or public downloads. Use `--resource-pool auto` for live quota selection.
Training runs are submitted as 8-GPU DDP jobs; evaluation remains a
single-process job because the evaluation script runs one model instance and
sweeps post-processing thresholds inside that process.

```bash
python scripts/submit_siflow_dbmim.py \
  --stage pretrain-cremi \
  --resource-pool auto \
  --gpus-per-pod 8 \
  --submit

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

After both finetuning checkpoints have been uploaded to TOS, run the aligned
VOI/ARAND evaluation:

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

The evaluation sweeps anisotropic z/xy thresholds with the stable graph
connected-components backend and the CuPy graph probe. The summary file is
`cremi_segmentation_summary.json`; the best row is under
`best_by_adapted_rand`. The same summary also includes `best_by_voi_sum` to
make VOI failures visible instead of reporting only the ARAND-selected row.

## Outputs

Current TOS prefixes:

```text
tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_dbmim/
tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_long_dbmim/
tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained/
tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch/
tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_longpretrained/
tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_pretrained/
tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_scratch/
```

Local final-weight copies, when available:

```text
outputs/final_weights/pretrained_latest.pt
outputs/final_weights/finetuned_best.pt
outputs/final_weights/finetuned_latest.pt
```

See `EXPERIMENTS.md` for completed UUIDs, checkpoint paths, and ablation
results.

## Legacy Baseline

The previous `MAEBackboneAffinityNet` head, simplified `UNETRAffinityNet`, and
old post-processing/RAG experiments are kept for comparison only. They are not
the maintained paper-aligned backbone path. The recommended current baseline is
anisotropic UNETR scratch vs anisotropic UNETR initialized from dbMiM
pretraining.

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

## Data License Notes

Use external EM datasets only under their original licenses and access rules.
Do not commit downloaded datasets, TOS credentials, Hugging Face tokens, SiFlow
credentials, or generated large checkpoints.
