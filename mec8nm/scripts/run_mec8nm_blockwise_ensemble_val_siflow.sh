#!/usr/bin/env bash
set -euo pipefail

# This wrapper is intentionally validation-only. It always evaluates M08/M10
# and mixes the bundled R57 primary checkpoint with one named secondary asset.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

export SAMPLES="mec8nm_08.h5 mec8nm_10.h5"
export BACKENDS="waterz"
export REGIONS="full"
export FAIL_ON_BACKEND_ERROR="1"
export SKIP_PRED_VARIANT="1"

export CHECKPOINT_ASSET_DIR="${CHECKPOINT_ASSET_DIR:-assets/r57_mec_stage1}"
export CHECKPOINT_ASSET_NAME="${CHECKPOINT_ASSET_NAME:-finetuned_latest.pt}"
export ENSEMBLE_CHECKPOINT_TOS_URI="${ENSEMBLE_CHECKPOINT_TOS_URI:-tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r59_seed1789_dpp_stage1/train/finetune_mec8nm_r59_seed1789_dpp_stage1/finetuned_latest.pt}"
export ENSEMBLE_CHECKPOINT_ASSET_DIR="${ENSEMBLE_CHECKPOINT_ASSET_DIR:-assets/r59_mec_stage1}"
export ENSEMBLE_CHECKPOINT_ASSET_NAME="${ENSEMBLE_CHECKPOINT_ASSET_NAME:-finetuned_latest.pt}"
export ENSEMBLE_WEIGHTS="${ENSEMBLE_WEIGHTS:-0.85 0.15}"
export ENSEMBLE_MEMBER_BIASES="${ENSEMBLE_MEMBER_BIASES:--0.20 -0.45 -0.45 0.00 0.00 0.00}"
export ENSEMBLE_MEMBER_TEMPERATURES="${ENSEMBLE_MEMBER_TEMPERATURES:-1.20 1.20}"

# Member calibration happens before the logit mix. Keep outer calibration neutral.
export CALIBRATION_BIASES="0.00 0.00 0.00"
export CALIBRATION_TEMPERATURES="1.00"

exec bash scripts/run_mec8nm_blockwise_eval_siflow.sh
