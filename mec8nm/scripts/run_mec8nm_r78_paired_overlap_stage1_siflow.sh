#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

readonly EXPECTED_RUN_NAME="finetune_mec8nm_r78_seed1789_paired_overlap_stage1"
readonly EXPECTED_CONFIG_NAME="finetune_mec8nm_r78_seed1789_paired_overlap_stage1"
readonly EXPECTED_TRAIN_IDS="1 2 3 4 5 6 7 9"
readonly EXPECTED_DOWNLOAD_IDS="1 2 3 4 5 6 7 8 9 10"
readonly EXPECTED_NPROC="8"

for name in RUN_NAME CONFIG_NAME EVAL_SPLITS EVAL_BACKENDS TRAIN_SAMPLE_IDS DOWNLOAD_SAMPLE_IDS EXPECTED_METRIC_ROWS NPROC; do
  value="${!name:-}"
  case "${name}" in
    RUN_NAME) expected="${EXPECTED_RUN_NAME}" ;;
    CONFIG_NAME) expected="${EXPECTED_CONFIG_NAME}" ;;
    EVAL_SPLITS) expected="val" ;;
    EVAL_BACKENDS) expected="waterz" ;;
    TRAIN_SAMPLE_IDS) expected="${EXPECTED_TRAIN_IDS}" ;;
    DOWNLOAD_SAMPLE_IDS) expected="${EXPECTED_DOWNLOAD_IDS}" ;;
    EXPECTED_METRIC_ROWS) expected="80" ;;
    NPROC) expected="${EXPECTED_NPROC}" ;;
  esac
  if [ -n "${value}" ] && [ "${value}" != "${expected}" ]; then
    echo "${name} is fixed for the R78 validation gate" >&2
    exit 2
  fi
done

export RUN_NAME="${EXPECTED_RUN_NAME}"
export CONFIG_NAME="${EXPECTED_CONFIG_NAME}"
export TRAIN_SAMPLE_IDS="${EXPECTED_TRAIN_IDS}"
export DOWNLOAD_SAMPLE_IDS="${EXPECTED_DOWNLOAD_IDS}"
export EVAL_SPLITS="val"
export EVAL_BACKENDS="waterz"
export EXPECTED_METRIC_ROWS="80"
export PRETRAINED_ASSET_DIR="assets/r57"
export PRETRAINED_ASSET_NAME="finetuned_latest.pt"
export PRETRAINED_SOURCE="${ROOT}/outputs/hf_release/dbmim-neuron-segmentation/weights/publicem_dbmim_r57_seed777_long20k/finetuned_latest.pt"
export NPROC="${EXPECTED_NPROC}"

python - "${ROOT}" "${EXPECTED_CONFIG_NAME}" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
path = root / "configs" / f"{sys.argv[2]}.yaml"
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
data = cfg["data"]
train = cfg["train"]
overlap = train.get("paired_overlap_consistency", {})
expected_paths = [f"mec8nm_{index:02d}.h5" for index in (1, 2, 3, 4, 5, 6, 7, 9)]
assert [Path(value).name for value in data["image_paths"]] == expected_paths
assert [Path(value).name for value in data["label_paths"]] == expected_paths
assert data.get("paired_overlap") is True
assert data.get("paired_overlap_shift") == [8, 32, 32]
assert overlap == {
    "weight": 0.05,
    "channels": 3,
    "expected_overlap": [24, 128, 128],
    "confidence_floor": 0.0,
}
assert "augmentation_consistency" not in train
assert "halo_consistency" not in train
assert int(train["max_steps"]) == 12000
assert int(cfg["seed"]) == 1789
print({"r78_static_contract": "ok", "shift": data["paired_overlap_shift"], "overlap": overlap})
PY

exec bash "${ROOT}/scripts/run_mec8nm_stage1_siflow.sh"
