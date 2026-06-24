#!/usr/bin/env bash
set -euo pipefail

ROOT=${DBMIM_ROOT:-/volume/med-train/users/dchen02/code/dbMiM}

export DBMIM_R20_PRETRAIN_PREFIX=${DBMIM_R20_PRETRAIN_PREFIX:-tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_adaptive_dbmim_r35}
export DBMIM_R20_CHECKPOINT=${DBMIM_R20_CHECKPOINT:-checkpoint_step_00160000.pt}
export DBMIM_R20_MIN_STEP=${DBMIM_R20_MIN_STEP:-160000}
export DBMIM_WATCH_INTERVAL_SEC=${DBMIM_WATCH_INTERVAL_SEC:-180}
export DBMIM_MAX_POLLS=${DBMIM_MAX_POLLS:-240}
export DBMIM_RESOURCE_POOL=${DBMIM_RESOURCE_POOL:-med-model}
export DBMIM_GPUS_PER_POD=${DBMIM_GPUS_PER_POD:-2}
export DBMIM_FINETUNE_SUBMIT_MARKER=${DBMIM_FINETUNE_SUBMIT_MARKER:-$ROOT/outputs/watchers/full-em-adaptive-r35-finetune.submitted}
export DBMIM_FINETUNE_PARTIAL_MARKER=${DBMIM_FINETUNE_PARTIAL_MARKER:-$ROOT/outputs/watchers/full-em-adaptive-r35-finetune.partial}
export DBMIM_R20_FINETUNE_STAGES=${DBMIM_R20_FINETUNE_STAGES:-"finetune-cremi-unetr-aniso-arch-explore-maws-mse-fullem-adaptive-r35q"}

exec "$ROOT/scripts/watch_and_submit_full_em_finetune.sh"
