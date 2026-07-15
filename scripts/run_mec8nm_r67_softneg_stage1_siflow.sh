#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

readonly EXPECTED_RUN_NAME="finetune_mec8nm_r67_seed1789_softneg_stage1"
readonly EXPECTED_CONFIG_NAME="finetune_mec8nm_r67_seed1789_softneg_stage1"
readonly EXPECTED_TRAIN_IDS="1 2 3 4 5 6 7 9"
readonly EXPECTED_NPROC="8"

for name in RUN_NAME CONFIG_NAME EVAL_SPLITS EVAL_BACKENDS TRAIN_SAMPLE_IDS EXPECTED_METRIC_ROWS NPROC; do
  value="${!name:-}"
  case "${name}" in
    RUN_NAME)
      expected="${EXPECTED_RUN_NAME}"
      ;;
    CONFIG_NAME)
      expected="${EXPECTED_CONFIG_NAME}"
      ;;
    EVAL_SPLITS)
      expected="val"
      ;;
    EVAL_BACKENDS)
      expected="waterz"
      ;;
    TRAIN_SAMPLE_IDS)
      expected="${EXPECTED_TRAIN_IDS}"
      ;;
    EXPECTED_METRIC_ROWS)
      expected="80"
      ;;
    NPROC)
      expected="${EXPECTED_NPROC}"
      ;;
  esac
  if [ -n "${value}" ] && [ "${value}" != "${expected}" ]; then
    echo "${name} is fixed for the R67 validation gate" >&2
    exit 2
  fi
done

export RUN_NAME="${EXPECTED_RUN_NAME}"
export CONFIG_NAME="${EXPECTED_CONFIG_NAME}"
export TRAIN_SAMPLE_IDS="${EXPECTED_TRAIN_IDS}"
export EVAL_SPLITS="val"
export EVAL_BACKENDS="waterz"
export EXPECTED_METRIC_ROWS="80"
export PRETRAINED_ASSET_DIR="assets/r57"
export PRETRAINED_ASSET_NAME="finetuned_latest.pt"
export NPROC="${EXPECTED_NPROC}"

exec bash "${ROOT}/scripts/run_mec8nm_stage1_siflow.sh"
