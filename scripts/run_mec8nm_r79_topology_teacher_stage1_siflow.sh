#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

readonly EXPECTED_RUN_NAME="finetune_mec8nm_r79_seed1789_topology_teacher_stage1"
readonly EXPECTED_CONFIG_NAME="finetune_mec8nm_r79_seed1789_topology_teacher_stage1"
readonly EXPECTED_TRAIN_IDS="1 2 3 4 5 6 7 9"
readonly EXPECTED_DOWNLOAD_IDS="1 2 3 4 5 6 7 8 9 10"
readonly EXPECTED_NPROC="8"
readonly SIMPLE_TEACHER="assets/r79_teachers/r58.pt"
readonly COMPLEX_TEACHER="assets/r79_teachers/r71_ema.pt"
readonly SIMPLE_SHA256="4715c6679808eac5e073159202be1cca77307ab90202bd683f8ab97cacc55e09"
readonly COMPLEX_SHA256="9c82d7ac4d4196a19966451343fcc033b7ea279a428b5913a54d9f4075790a45"

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
    echo "${name} is fixed for the R79 validation gate" >&2
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

for teacher in "${SIMPLE_TEACHER}" "${COMPLEX_TEACHER}"; do
  if [ ! -s "${ROOT}/${teacher}" ]; then
    echo "missing R79 teacher asset: ${teacher}" >&2
    exit 3
  fi
done
printf '%s  %s\n%s  %s\n' \
  "${SIMPLE_SHA256}" "${ROOT}/${SIMPLE_TEACHER}" \
  "${COMPLEX_SHA256}" "${ROOT}/${COMPLEX_TEACHER}" | sha256sum -c -

python - "${ROOT}" "${EXPECTED_CONFIG_NAME}" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
path = root / "configs" / f"{sys.argv[2]}.yaml"
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
data = cfg["data"]
train = cfg["train"]
distill = train.get("topology_teacher_distillation", {})
expected_paths = [f"mec8nm_{index:02d}.h5" for index in (1, 2, 3, 4, 5, 6, 7, 9)]
assert [Path(value).name for value in data["image_paths"]] == expected_paths
assert [Path(value).name for value in data["label_paths"]] == expected_paths
assert data.get("return_topology_stats") is True
assert not data.get("paired_overlap", False)
assert distill == {
    "weight": 0.10,
    "channels": 3,
    "simple_teacher": "assets/r79_teachers/r58.pt",
    "complex_teacher": "assets/r79_teachers/r71_ema.pt",
    "expected_teacher_keys": 214,
    "foreground_threshold": 0.95,
    "foreground_width": 0.02,
    "boundary_threshold": 0.05,
    "boundary_width": 0.02,
    "unique_threshold": 5.5,
    "unique_width": 2.0,
    "confidence_floor": 0.0,
}
assert "augmentation_consistency" not in train
assert "halo_consistency" not in train
assert "paired_overlap_consistency" not in train
assert int(train["max_steps"]) == 12000
assert int(cfg["seed"]) == 1789
print({"r79_static_contract": "ok", "topology_teacher_distillation": distill})
PY

exec bash "${ROOT}/scripts/run_mec8nm_stage1_siflow.sh"
