#!/usr/bin/env bash
set -euo pipefail

ROOT=${DBMIM_ROOT:-/volume/med-train/users/dchen02/code/dbMiM}

source /volume/med-train/users/dchen02/secrets/siflow_env_dchen02.sh >/dev/null 2>&1
source /volume/med-train/users/dchen02/secrets/tos_env_dchen02.sh >/dev/null 2>&1

export DBMIM_R20_PRETRAIN_PREFIX=${DBMIM_R20_PRETRAIN_PREFIX:-tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_decoderaware_dbmim_r21}
export DBMIM_R20_MIN_STEP=${DBMIM_R20_MIN_STEP:-80000}
export DBMIM_WATCH_INTERVAL_SEC=${DBMIM_WATCH_INTERVAL_SEC:-180}
export DBMIM_MAX_POLLS=${DBMIM_MAX_POLLS:-160}
export DBMIM_RESOURCE_POOL=${DBMIM_RESOURCE_POOL:-med-model}
export DBMIM_GPUS_PER_POD=${DBMIM_GPUS_PER_POD:-2}
export DBMIM_FINETUNE_SUBMIT_MARKER=${DBMIM_FINETUNE_SUBMIT_MARKER:-$ROOT/outputs/watchers/r21-decoderaware-finetune.submitted}
export DBMIM_FINETUNE_PARTIAL_MARKER=${DBMIM_FINETUNE_PARTIAL_MARKER:-$ROOT/outputs/watchers/r21-decoderaware-finetune.partial}
export DBMIM_R20_FINETUNE_STAGES=${DBMIM_R20_FINETUNE_STAGES:-"finetune-cremi-unetr-aniso-arch-explore-maws-mse-scratch-r21q finetune-cremi-unetr-aniso-arch-explore-maws-mse-decoderaware-r21q finetune-cremi-unetr-aniso-arch-explore-maws-mse-dpp-scratch-r21q finetune-cremi-unetr-aniso-arch-explore-maws-mse-decoderaware-dpp-r21q"}

exec "$ROOT/scripts/watch_and_submit_full_em_finetune.sh"
