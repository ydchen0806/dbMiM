#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

readonly EXPECTED_RUN_NAME="finetune_mec8nm_r80_seed1789_instance_embedding_stage1"
readonly EXPECTED_CONFIG_NAME="finetune_mec8nm_r80_seed1789_instance_embedding_stage1"
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
    echo "${name} is fixed for the R80 validation gate" >&2
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
embedding = train.get("instance_embedding", {})
expected_paths = [f"mec8nm_{index:02d}.h5" for index in (1, 2, 3, 4, 5, 6, 7, 9)]
assert [Path(value).name for value in data["image_paths"]] == expected_paths
assert [Path(value).name for value in data["label_paths"]] == expected_paths
assert not data.get("paired_overlap", False)
assert not data.get("return_topology_stats", False)
assert embedding == {
    "weight": 0.02,
    "feature_stride": [2, 4, 4],
    "min_voxels": 8,
    "max_instances": 64,
    "variance_margin": 0.20,
    "distance_margin": 1.0,
    "variance_weight": 1.0,
    "distance_weight": 1.0,
}
assert "augmentation_consistency" not in train
assert "halo_consistency" not in train
assert "paired_overlap_consistency" not in train
assert "topology_teacher_distillation" not in train
assert "dpp_weight" not in train.get("loss", {})
assert "lsd_weight" not in train.get("loss", {})
assert int(cfg["model"]["out_channels"]) == 3
assert int(train["max_steps"]) == 12000
assert int(cfg["seed"]) == 1789
print({"r80_static_contract": "ok", "instance_embedding": embedding})
PY

exec bash "${ROOT}/scripts/run_mec8nm_stage1_siflow.sh"
