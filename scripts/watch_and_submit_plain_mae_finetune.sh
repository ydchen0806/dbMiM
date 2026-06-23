#!/usr/bin/env bash
set -euo pipefail

ROOT=${DBMIM_ROOT:-/volume/med-train/users/dchen02/code/dbMiM}

export DBMIM_R20_PRETRAIN_PREFIX=${DBMIM_R20_PRETRAIN_PREFIX:-tos://agi-data/users/dchen02/dbmim/outputs/pretrain_public_em_plain_mae_r23}
export DBMIM_R20_MIN_STEP=${DBMIM_R20_MIN_STEP:-40000}
export DBMIM_WATCH_INTERVAL_SEC=${DBMIM_WATCH_INTERVAL_SEC:-180}
export DBMIM_MAX_POLLS=${DBMIM_MAX_POLLS:-240}
export DBMIM_RESOURCE_POOL=${DBMIM_RESOURCE_POOL:-med-model}
export DBMIM_GPUS_PER_POD=${DBMIM_GPUS_PER_POD:-2}
export DBMIM_FINETUNE_SUBMIT_MARKER=${DBMIM_FINETUNE_SUBMIT_MARKER:-$ROOT/outputs/watchers/plain-mae-r23-finetune.submitted}
export DBMIM_FINETUNE_PARTIAL_MARKER=${DBMIM_FINETUNE_PARTIAL_MARKER:-$ROOT/outputs/watchers/plain-mae-r23-finetune.partial}
export DBMIM_R20_FINETUNE_STAGES=${DBMIM_R20_FINETUNE_STAGES:-"finetune-cremi-unetr-aniso-arch-explore-maws-mse-publicem-plainmae-r23q"}

exec "$ROOT/scripts/watch_and_submit_full_em_finetune.sh"
