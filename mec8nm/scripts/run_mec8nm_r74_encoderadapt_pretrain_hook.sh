#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DBMIM="${DBMIM:-${ROOT}}"
PRETRAINED_ASSET="${PRETRAINED_ASSET:?missing PRETRAINED_ASSET}"
NPROC="${NPROC:-8}"

PRETRAIN_CONFIG_NAME="pretrain_mec8nm_r74_seed1789_encoderadapt"
PRETRAIN_RUN_NAME="pretrain_mec8nm_r74_seed1789_encoderadapt"
EXPECTED_TRAIN_IDS="1 2 3 4 5 6 7 9"
EXPECTED_R57_SHA256="9b67e337aba837ee6c26e5c4b80627ab3745fdb3fe152190d0164c6e9a2fe2ba"
EXPECTED_STEPS="4000"
EXPECTED_ENCODER_KEYS="77"

if [ "${NPROC}" != "8" ]; then
  echo "R74 pretraining requires exactly 8 processes" >&2
  exit 2
fi
if [ ! -s "${PRETRAINED_ASSET}" ]; then
  echo "missing R57 initializer: ${PRETRAINED_ASSET}" >&2
  exit 3
fi
actual_sha256="$(sha256sum "${PRETRAINED_ASSET}" | awk '{print $1}')"
if [ "${actual_sha256}" != "${EXPECTED_R57_SHA256}" ]; then
  echo "R57 initializer checksum mismatch: ${actual_sha256}" >&2
  exit 3
fi

runtime_config="configs/runtime_${PRETRAIN_CONFIG_NAME}.yaml"
output_dir="outputs/${PRETRAIN_RUN_NAME}"
adapted_checkpoint="${output_dir}/pretrained_latest.pt"
hybrid_checkpoint="${output_dir}/r57_encoderadapt_hybrid.pt"
mkdir -p configs "${output_dir}"

R74_CONFIG_NAME="${PRETRAIN_CONFIG_NAME}" \
R74_RUNTIME_CONFIG="${runtime_config}" \
R74_OUTPUT_DIR="${output_dir}" \
R74_TRAIN_IDS="${EXPECTED_TRAIN_IDS}" \
python - <<'PY'
import json
import os
from pathlib import Path

import yaml

root = Path.cwd()
config_name = os.environ["R74_CONFIG_NAME"]
src = Path("external/dbMiM/configs") / f"{config_name}.yaml"
if not src.exists():
    src = Path("configs") / f"{config_name}.yaml"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
train_ids = [int(value) for value in os.environ["R74_TRAIN_IDS"].split()]
if train_ids != [1, 2, 3, 4, 5, 6, 7, 9]:
    raise RuntimeError(f"R74 train IDs changed: {train_ids}")
paths = [
    str((root / "data" / "MEC_8nm" / "hdf5" / f"mec8nm_{sample_id:02d}.h5").resolve())
    for sample_id in train_ids
]
missing = [path for path in paths if not Path(path).is_file()]
if missing:
    raise RuntimeError(f"missing R74 raw training blocks: {missing}")
data_cfg = cfg.setdefault("data", {})
data_cfg["train_paths"] = paths
if "label_paths" in data_cfg:
    raise RuntimeError("R74 pretraining config must not contain label_paths")
cfg["output_dir"] = str((root / os.environ["R74_OUTPUT_DIR"]).resolve())
out = Path(os.environ["R74_RUNTIME_CONFIG"])
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(
    json.dumps(
        {
            "event": "r74_pretrain_config",
            "runtime_config": str(out),
            "train_ids": train_ids,
            "label_paths_present": False,
            "max_steps": cfg["train"]["max_steps"],
            "architecture": cfg["model"]["architecture"],
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

torchrun --standalone --nproc_per_node="${NPROC}" \
  "${DBMIM}/train_pretrain.py" \
  --config "${runtime_config}" \
  --pretrained "${PRETRAINED_ASSET}" \
  --output-dir "${output_dir}"

python "${DBMIM}/scripts/build_mec8nm_r74_hybrid_checkpoint.py" \
  --r57 "${PRETRAINED_ASSET}" \
  --adapted "${adapted_checkpoint}" \
  --output "${hybrid_checkpoint}" \
  --expected-adapted-step "${EXPECTED_STEPS}" \
  --expected-encoder-keys "${EXPECTED_ENCODER_KEYS}"

if [ ! -s "${hybrid_checkpoint}" ]; then
  echo "R74 hybrid checkpoint was not created: ${hybrid_checkpoint}" >&2
  exit 3
fi
