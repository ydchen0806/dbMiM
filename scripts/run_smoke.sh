#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python train_pretrain.py --config configs/pretrain_smoke.yaml
python train_finetune.py --config configs/finetune_smoke.yaml
