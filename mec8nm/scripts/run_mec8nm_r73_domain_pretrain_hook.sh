#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DBMIM="${DBMIM:-${ROOT}}"
PRETRAINED_ASSET="${PRETRAINED_ASSET:?missing PRETRAINED_ASSET}"
NPROC="${NPROC:-8}"

readonly PRETRAIN_CONFIG_NAME="pretrain_mec8nm_r73_seed1789_decoderaware_domain"
readonly PRETRAIN_RUN_NAME="pretrain_mec8nm_r73_seed1789_decoderaware_domain"
readonly EXPECTED_TRAIN_IDS="1 2 3 4 5 6 7 9"
readonly EXPECTED_R57_SHA256="9b67e337aba837ee6c26e5c4b80627ab3745fdb3fe152190d0164c6e9a2fe2ba"
readonly EXPECTED_R73_PRETRAIN_SHA256="32ba0c3aa14a59cb3fa285bb0c5875f96574a1e1bd65da2c3751eeeecf3b2262"
readonly EXPECTED_STEPS="4000"
readonly EXPECTED_LOADED_KEYS="214"

if [ "${NPROC}" != "8" ]; then
  echo "R73 pretraining requires exactly 8 processes" >&2
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
mkdir -p configs "${output_dir}"

R73_PRETRAIN_CONFIG_NAME="${PRETRAIN_CONFIG_NAME}" \
R73_RUNTIME_CONFIG="${runtime_config}" \
R73_OUTPUT_DIR="${output_dir}" \
R73_EXPECTED_TRAIN_IDS="${EXPECTED_TRAIN_IDS}" \
python - <<'PY'
import json
import os
from pathlib import Path

import yaml

root = Path.cwd()
config_name = os.environ["R73_PRETRAIN_CONFIG_NAME"]
src = Path("external/dbMiM/configs") / f"{config_name}.yaml"
if not src.exists():
    src = Path("configs") / f"{config_name}.yaml"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
train_ids = [int(value) for value in os.environ["R73_EXPECTED_TRAIN_IDS"].split()]
if train_ids != [1, 2, 3, 4, 5, 6, 7, 9]:
    raise RuntimeError(f"R73 train IDs changed: {train_ids}")
paths = [
    str((root / "data" / "MEC_8nm" / "hdf5" / f"mec8nm_{sample_id:02d}.h5").resolve())
    for sample_id in train_ids
]
missing = [path for path in paths if not Path(path).is_file()]
if missing:
    raise RuntimeError(f"missing R73 raw training blocks: {missing}")
data_cfg = cfg.setdefault("data", {})
data_cfg["train_paths"] = paths
if "label_paths" in data_cfg:
    raise RuntimeError("R73 pretraining config must not contain label_paths")
cfg["output_dir"] = str((root / os.environ["R73_OUTPUT_DIR"]).resolve())
out = Path(os.environ["R73_RUNTIME_CONFIG"])
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(
    json.dumps(
        {
            "event": "r73_pretrain_config",
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

if [ -n "${R73_REUSE_PRETRAINED_ASSET:-}" ]; then
  if [ ! -s "${R73_REUSE_PRETRAINED_ASSET}" ]; then
    echo "missing completed R73 pretraining asset: ${R73_REUSE_PRETRAINED_ASSET}" >&2
    exit 3
  fi
  reuse_sha256="$(sha256sum "${R73_REUSE_PRETRAINED_ASSET}" | awk '{print $1}')"
  if [ "${reuse_sha256}" != "${EXPECTED_R73_PRETRAIN_SHA256}" ]; then
    echo "completed R73 pretraining checksum mismatch: ${reuse_sha256}" >&2
    exit 3
  fi
  cp "${R73_REUSE_PRETRAINED_ASSET}" "${output_dir}/pretrained_latest.pt"
  echo "{\"event\":\"r73_pretrain_checkpoint_reused\",\"sha256\":\"${reuse_sha256}\"}"
else
  torchrun --standalone --nproc_per_node="${NPROC}" \
    "${DBMIM}/train_pretrain.py" \
    --config "${runtime_config}" \
    --pretrained "${PRETRAINED_ASSET}" \
    --output-dir "${output_dir}"
fi

R73_CHECKPOINT="${output_dir}/pretrained_latest.pt" \
R73_EXPECTED_STEPS="${EXPECTED_STEPS}" \
R73_EXPECTED_LOADED_KEYS="${EXPECTED_LOADED_KEYS}" \
python - <<'PY'
import json
import os
from pathlib import Path

import torch

path = Path(os.environ["R73_CHECKPOINT"])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
step = int(checkpoint.get("global_step", -1))
loaded = list(checkpoint.get("pretrained_loaded_keys", []))
expected_steps = int(os.environ["R73_EXPECTED_STEPS"])
expected_loaded = int(os.environ["R73_EXPECTED_LOADED_KEYS"])
if step != expected_steps:
    raise RuntimeError(f"R73 pretraining stopped at {step}, expected {expected_steps}")
if len(loaded) != expected_loaded:
    raise RuntimeError(f"R73 loaded {len(loaded)} R57 keys, expected {expected_loaded}")
required = {"head.z_head.weight", "encoder1.conv.0.weight", "patch_embed.proj.weight"}
if not required.issubset(loaded):
    raise RuntimeError(f"R73 did not initialize required shared keys: {sorted(required - set(loaded))}")
print(
    json.dumps(
        {
            "event": "r73_pretrain_checkpoint_verified",
            "checkpoint": str(path),
            "global_step": step,
            "r57_loaded_keys": len(loaded),
        },
        sort_keys=True,
    ),
    flush=True,
)
PY
