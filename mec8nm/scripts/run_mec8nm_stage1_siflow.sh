#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -d "${ROOT}/external/dbMiM" ]; then
  DBMIM="${ROOT}/external/dbMiM"
else
  DBMIM="${ROOT}"
fi
cd "${ROOT}"

export PYTHONPATH="${DBMIM}:${DBMIM}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-finetune_mec8nm_r48_seed309_stage1}"
CONFIG_NAME="${CONFIG_NAME:-finetune_mec8nm_r48_seed309_stage1}"
TOS_MEC_PREFIX="${TOS_MEC_PREFIX:-tos://agi-data/users/dchen02/dbmim/assets/mec8nm_20260711/MEC_8nm}"
TOS_OUTPUT_PREFIX="${TOS_OUTPUT_PREFIX:-tos://agi-data/users/dchen02/dbmim/mec8nm_runs/${RUN_NAME}}"
NPROC="${NPROC:-8}"
PRETRAINED_ASSET_DIR="${PRETRAINED_ASSET_DIR:-assets/r48}"
PRETRAINED_ASSET_NAME="${PRETRAINED_ASSET_NAME:-finetuned_latest.pt}"
PRETRAINED_SOURCE="${PRETRAINED_SOURCE:-${DBMIM}/outputs/hf_release/dbmim-neuron-segmentation/weights/publicem_dbmim_r48_seed309_long20k/finetuned_latest.pt}"
DOWNLOAD_SAMPLE_IDS="${DOWNLOAD_SAMPLE_IDS:-}"
PRE_FINETUNE_HOOK="${PRE_FINETUNE_HOOK:-}"
PRE_FINETUNE_INIT="${PRE_FINETUNE_INIT:-}"
PRE_FINETUNE_OUTPUT_DIR="${PRE_FINETUNE_OUTPUT_DIR:-}"
RUNTIME_CONFIG="configs/runtime_${RUN_NAME}.yaml"
PRETRAINED_ASSET="${PRETRAINED_ASSET_DIR}/${PRETRAINED_ASSET_NAME}"
read -r -a EVAL_BACKENDS_ARRAY <<< "${EVAL_BACKENDS:-graph_cc waterz}"
read -r -a EVAL_SPLITS_ARRAY <<< "${EVAL_SPLITS:-val test}"
EXPECTED_METRIC_ROWS="${EXPECTED_METRIC_ROWS:-0}"

mkdir -p data/MEC_8nm configs outputs logs "${PRETRAINED_ASSET_DIR}" bin

if [ ! -x bin/tosutil ]; then
  if [ -x /volume/med-train/users/dchen02/bin/tosutil ]; then
    cp /volume/med-train/users/dchen02/bin/tosutil bin/tosutil
  else
    echo "missing tosutil" >&2
    exit 2
  fi
fi

TOS_CONF=/tmp/mec8nm_tosutil.conf
: > "${TOS_CONF}"
bin/tosutil config \
  -e "${TOS_ENDPOINT:?missing TOS_ENDPOINT}" \
  -re "${TOS_REGION:?missing TOS_REGION}" \
  -i "${TOS_ACCESS_KEY_ID:?missing TOS_ACCESS_KEY_ID}" \
  -k "${TOS_SECRET_ACCESS_KEY:?missing TOS_SECRET_ACCESS_KEY}" \
  -conf="${TOS_CONF}" >/dev/null

echo "[mec8nm] installing optional offline wheels"
python -m pip install --user --no-index --find-links wheelhouse_dbmim pyyaml h5py scipy mahotas connected-components-3d || true
python -m pip install --user --no-index --find-links wheelhouse_superhuman_eval scikit-image tifffile || true

need_waterz=0
for backend in "${EVAL_BACKENDS_ARRAY[@]}"; do
  if [ "${backend}" = "waterz" ]; then
    need_waterz=1
    break
  fi
done
if [ "${need_waterz}" -eq 1 ]; then
  if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("waterz") else 1)
PY
  then
    WATERZ_SRC=""
    if [ -d third_party/waterz ]; then
      WATERZ_SRC="third_party/waterz"
    elif [ -d "${DBMIM}/third_party/waterz" ]; then
      WATERZ_SRC="${DBMIM}/third_party/waterz"
    fi
    for boost_root in third_party/boost/include "${DBMIM}/third_party/boost/include" third_party/boost_1_84_0 "${DBMIM}/third_party/boost_1_84_0"; do
      if [ -d "${boost_root}/boost" ]; then
        export CPLUS_INCLUDE_PATH="${boost_root}:${CPLUS_INCLUDE_PATH:-}"
        export BOOST_INCLUDEDIR="${boost_root}"
        break
      fi
    done
    if [ -z "${WATERZ_SRC}" ]; then
      echo "waterz requested but no bundled source directory is available" >&2
      exit 4
    fi
    python -m pip install --user --no-build-isolation --no-index --find-links wheelhouse_superhuman_eval "${WATERZ_SRC}"
  fi
  if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("waterz") else 1)
PY
  then
    echo "waterz requested but installation failed" >&2
    exit 4
  fi
fi

echo "[mec8nm] downloading MEC HDF5 from ${TOS_MEC_PREFIX}"
if [ -n "${DOWNLOAD_SAMPLE_IDS}" ]; then
  mkdir -p data/MEC_8nm/hdf5
  for sample_id in ${DOWNLOAD_SAMPLE_IDS}; do
    if ! [[ "${sample_id}" =~ ^([1-9]|1[0-2])$ ]]; then
      echo "invalid DOWNLOAD_SAMPLE_IDS entry: ${sample_id}" >&2
      exit 2
    fi
    sample_name="mec8nm_$(printf '%02d' "${sample_id}").h5"
    bin/tosutil cp \
      "${TOS_MEC_PREFIX}/hdf5/${sample_name}" \
      "data/MEC_8nm/hdf5/${sample_name}" \
      -conf="${TOS_CONF}"
  done
else
  bin/tosutil cp "${TOS_MEC_PREFIX}/hdf5" data/MEC_8nm/ -r -conf="${TOS_CONF}"
fi
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_manifest.json" data/MEC_8nm/mec8nm_manifest.json -conf="${TOS_CONF}" || true
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_splits.json" data/MEC_8nm/mec8nm_splits.json -conf="${TOS_CONF}" || true

if [ ! -s "${PRETRAINED_ASSET}" ]; then
  if [ -s "${PRETRAINED_SOURCE}" ]; then
    cp "${PRETRAINED_SOURCE}" "${PRETRAINED_ASSET}"
  elif [ -s "${ROOT}/${PRETRAINED_ASSET}" ]; then
    :
  else
    echo "missing pretrained checkpoint asset: ${PRETRAINED_ASSET}" >&2
    exit 3
  fi
fi

if [ -n "${PRE_FINETUNE_HOOK}" ]; then
  if [ -z "${PRE_FINETUNE_INIT}" ]; then
    echo "PRE_FINETUNE_INIT is required when PRE_FINETUNE_HOOK is set" >&2
    exit 2
  fi
  echo "[mec8nm] running pre-finetune hook ${PRE_FINETUNE_HOOK}"
  pretrain_start="$(date +%s)"
  set +e
  PRETRAINED_ASSET="${PRETRAINED_ASSET}" NPROC="${NPROC}" DBMIM="${DBMIM}" \
    bash "${PRE_FINETUNE_HOOK}" \
    2>&1 | sed -u "s/^/[pretrain] /" | tee "logs/${RUN_NAME}.pretrain.log"
  pretrain_code="${PIPESTATUS[0]}"
  set -e
  pretrain_end="$(date +%s)"
  echo "{\"event\":\"pretrain_done\",\"code\":${pretrain_code},\"elapsed_sec\":$((pretrain_end-pretrain_start))}" | tee "logs/${RUN_NAME}.status.jsonl"
  if [ "${pretrain_code}" -ne 0 ] || [ ! -s "${PRE_FINETUNE_INIT}" ]; then
    bin/tosutil cp outputs "${TOS_OUTPUT_PREFIX}/outputs_pretrain_failed" -r -conf="${TOS_CONF}" || true
    bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_pretrain_failed" -r -conf="${TOS_CONF}" || true
    if [ "${pretrain_code}" -ne 0 ]; then
      exit "${pretrain_code}"
    fi
    echo "pre-finetune hook did not produce ${PRE_FINETUNE_INIT}" >&2
    exit 3
  fi
  PRETRAINED_ASSET="${PRE_FINETUNE_INIT}"
fi

RUN_NAME="${RUN_NAME}" CONFIG_NAME="${CONFIG_NAME}" RUNTIME_CONFIG="${RUNTIME_CONFIG}" PRETRAINED_ASSET="${PRETRAINED_ASSET}" TRAIN_SAMPLE_IDS="${TRAIN_SAMPLE_IDS:-}" python - <<'PY'
import json
import os
from pathlib import Path
import yaml

root = Path.cwd()
run_name = os.environ["RUN_NAME"]
config_name = os.environ["CONFIG_NAME"]
runtime_config = Path(os.environ["RUNTIME_CONFIG"])
pretrained_asset = Path(os.environ["PRETRAINED_ASSET"])
src = Path("external/dbMiM/configs") / f"{config_name}.yaml"
if not src.exists():
    src = Path("configs") / f"{config_name}.yaml"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
data_cfg = cfg.setdefault("data", {})
requested_ids = os.environ.get("TRAIN_SAMPLE_IDS", "").strip()
if requested_ids:
    train_ids = [int(value) for value in requested_ids.replace(",", " ").split()]
else:
    train_ids = [int(value) for value in data_cfg.get("train_sample_ids", [1, 2, 3, 4, 5, 6, 7, 9])]
allowed_train_ids = {1, 2, 3, 4, 5, 6, 7, 9}
if not train_ids or any(sample_id not in allowed_train_ids for sample_id in train_ids):
    raise ValueError(f"TRAIN_SAMPLE_IDS must use only MEC train blocks {sorted(allowed_train_ids)}, got {train_ids}")
paths = [str((root / "data" / "MEC_8nm" / "hdf5" / f"mec8nm_{i:02d}.h5").resolve()) for i in train_ids]
data_cfg["image_paths"] = paths
data_cfg["label_paths"] = paths
cfg["pretrained"] = str((root / pretrained_asset).resolve())
cfg["output_dir"] = str((root / "outputs" / run_name).resolve())
out = runtime_config
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(
    json.dumps(
        {
            "runtime_config": str(out),
            "train_ids": train_ids,
            "train_paths": len(paths),
            "output_dir": cfg["output_dir"],
        }
    ),
    flush=True,
)
PY

python - <<'PY'
import importlib.util, json, torch
mods = ["torch", "h5py", "yaml", "scipy", "mahotas", "skimage", "waterz", "cc3d"]
print(json.dumps({"module_probe": {m: bool(importlib.util.find_spec(m)) for m in mods}, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()}), flush=True)
PY

echo "[mec8nm] starting train ${RUN_NAME}"
train_start="$(date +%s)"
set +e
torchrun --standalone --nproc_per_node="${NPROC}" \
  "${DBMIM}/train_finetune.py" \
  --config "${RUNTIME_CONFIG}" \
  --pretrained "${PRETRAINED_ASSET}" \
  --output-dir "outputs/${RUN_NAME}" \
  2>&1 | sed -u "s/^/[train] /" | tee "logs/${RUN_NAME}.train.log"
train_code="${PIPESTATUS[0]}"
set -e
train_end="$(date +%s)"
echo "{\"event\":\"train_done\",\"code\":${train_code},\"elapsed_sec\":$((train_end-train_start))}" | tee -a "logs/${RUN_NAME}.status.jsonl"
if [ "${train_code}" -ne 0 ]; then
  bin/tosutil cp outputs "${TOS_OUTPUT_PREFIX}/outputs_failed" -r -conf="${TOS_CONF}" || true
  bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_failed" -r -conf="${TOS_CONF}" || true
  exit "${train_code}"
fi

for split in "${EVAL_SPLITS_ARRAY[@]}"; do
  if [ "${split}" = "val" ]; then
    names=(mec8nm_08.h5 mec8nm_10.h5)
  elif [ "${split}" = "test" ]; then
    names=(mec8nm_11.h5 mec8nm_12.h5)
  else
    echo "unsupported EVAL_SPLITS entry: ${split}" >&2
    exit 2
  fi
  echo "[mec8nm] evaluating ${split}: ${names[*]}"
  set +e
  python "${DBMIM}/scripts/evaluate_cremi_segmentation.py" \
    --config "${RUNTIME_CONFIG}" \
    --checkpoint "outputs/${RUN_NAME}/finetuned_latest.pt" \
    --data-dir data/MEC_8nm/hdf5 \
    --sample-names "${names[@]}" \
    --output-dir "outputs/eval_${RUN_NAME}_${split}_crop64x512" \
    --crop-size 64 512 512 \
    --stride 16 80 80 \
    --backends "${EVAL_BACKENDS_ARRAY[@]}" \
    --thresholds 0.16 0.18 0.20 0.22 0.24 0.30 0.40 0.50 \
    --calibration-biases 0 0 0 -0.20 -0.45 -0.45 \
    --calibration-temperatures 1.0 1.15 \
    --seed-method maxima_distance \
    --seed-distance 10 \
    --boundary-threshold 0.5 \
    --waterz-scoring hist_quantile \
    --fail-on-backend-error \
    --metric-backend skimage \
    --ignore-label 0 \
    --cremi-boundary-ignore-distance-xy 1 \
    --cremi-boundary-ignore-distance-z 0 \
    --replicate-affinity-boundary \
    --device cuda \
    2>&1 | sed -u "s/^/[eval-${split}] /" | tee "logs/${RUN_NAME}.eval_${split}.log"
  eval_code="${PIPESTATUS[0]}"
  set -e
  echo "{\"event\":\"eval_done\",\"split\":\"${split}\",\"code\":${eval_code}}" | tee -a "logs/${RUN_NAME}.status.jsonl"
  if [ "${eval_code}" -ne 0 ]; then
    bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/train" -r -conf="${TOS_CONF}" || true
    bin/tosutil cp "outputs/eval_${RUN_NAME}_${split}_crop64x512" "${TOS_OUTPUT_PREFIX}/eval_${split}_crop64x512_failed" -r -conf="${TOS_CONF}" || true
    bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_failed" -r -conf="${TOS_CONF}" || true
    exit "${eval_code}"
  fi
  if [ "${EXPECTED_METRIC_ROWS}" -gt 0 ]; then
    EVAL_OUTPUT="outputs/eval_${RUN_NAME}_${split}_crop64x512" \
      EXPECTED_METRIC_ROWS="${EXPECTED_METRIC_ROWS}" \
      EXPECTED_SAMPLES="${names[*]}" \
      EXPECTED_BACKENDS="${EVAL_BACKENDS_ARRAY[*]}" \
      python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

output_dir = Path(os.environ["EVAL_OUTPUT"])
metrics_path = output_dir / "cremi_segmentation_metrics.csv"
summary_path = output_dir / "cremi_segmentation_summary.json"
expected_rows = int(os.environ["EXPECTED_METRIC_ROWS"])
expected_samples = set(os.environ["EXPECTED_SAMPLES"].split())
expected_backends = set(os.environ["EXPECTED_BACKENDS"].split())

if not metrics_path.is_file():
    raise RuntimeError(f"missing expected metrics file: {metrics_path}")
if not summary_path.is_file():
    raise RuntimeError(f"missing expected summary file: {summary_path}")
with metrics_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != expected_rows:
    raise RuntimeError(f"expected {expected_rows} metric rows, found {len(rows)}")
samples = {row.get("sample", "") for row in rows}
backends = {row.get("backend", "") for row in rows}
if samples != expected_samples:
    raise RuntimeError(f"metric sample set mismatch: expected {sorted(expected_samples)}, got {sorted(samples)}")
if backends != expected_backends:
    raise RuntimeError(f"metric backend set mismatch: expected {sorted(expected_backends)}, got {sorted(backends)}")
for row in rows:
    for field in ("adapted_rand_error", "voi_sum"):
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid {field} in metrics row: {row}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite {field} in metrics row: {row}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
failures = summary.get("failures", [])
if failures:
    raise RuntimeError(f"evaluator reported failures: {failures}")
print(
    json.dumps(
        {
            "event": "metric_assertion_passed",
            "rows": len(rows),
            "samples": sorted(samples),
            "backends": sorted(backends),
        },
        sort_keys=True,
    ),
    flush=True,
)
PY
  fi
done

echo "[mec8nm] uploading outputs to ${TOS_OUTPUT_PREFIX}"
if [ -n "${PRE_FINETUNE_OUTPUT_DIR}" ] && [ -d "${PRE_FINETUNE_OUTPUT_DIR}" ]; then
  bin/tosutil cp "${PRE_FINETUNE_OUTPUT_DIR}" "${TOS_OUTPUT_PREFIX}/pretrain" -r -conf="${TOS_CONF}" || true
fi
bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/train" -r -conf="${TOS_CONF}"
for split in "${EVAL_SPLITS_ARRAY[@]}"; do
  eval_output="outputs/eval_${RUN_NAME}_${split}_crop64x512"
  if [ -d "${eval_output}" ]; then
    bin/tosutil cp "${eval_output}" "${TOS_OUTPUT_PREFIX}/eval_${split}_crop64x512" -r -conf="${TOS_CONF}" || true
  fi
done
bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs" -r -conf="${TOS_CONF}" || true
echo "{\"event\":\"upload_done\",\"tos_output_prefix\":\"${TOS_OUTPUT_PREFIX}\"}" | tee -a "logs/${RUN_NAME}.status.jsonl"
