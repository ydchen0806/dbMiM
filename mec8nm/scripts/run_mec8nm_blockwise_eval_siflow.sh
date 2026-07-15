#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -d "${ROOT}/external/dbMiM" ]; then
  DBMIM="${ROOT}/external/dbMiM"
else
  DBMIM="${ROOT}"
fi
cd "${ROOT}"

export PYTHONPATH="${DBMIM}:${DBMIM}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-eval_mec8nm_r57_seed777_fullblock_test_graphcc}"
CONFIG_NAME="${CONFIG_NAME:-finetune_mec8nm_r57_seed777_stage1}"
TOS_MEC_PREFIX="${TOS_MEC_PREFIX:-tos://agi-data/users/dchen02/dbmim/assets/mec8nm_20260711/MEC_8nm}"
TOS_OUTPUT_PREFIX="${TOS_OUTPUT_PREFIX:-tos://agi-data/users/dchen02/dbmim/mec8nm_runs/${RUN_NAME}}"
CHECKPOINT_ASSET_DIR="${CHECKPOINT_ASSET_DIR:-assets/r57_mec_stage1}"
CHECKPOINT_ASSET_NAME="${CHECKPOINT_ASSET_NAME:-finetuned_latest.pt}"
CHECKPOINT_ASSET="${CHECKPOINT_ASSET_DIR}/${CHECKPOINT_ASSET_NAME}"
CHECKPOINT_TOS_URI="${CHECKPOINT_TOS_URI:-}"
ENSEMBLE_CHECKPOINT_TOS_URI="${ENSEMBLE_CHECKPOINT_TOS_URI:-}"
ENSEMBLE_CHECKPOINT_ASSET_DIR="${ENSEMBLE_CHECKPOINT_ASSET_DIR:-assets/ensemble_secondary}"
ENSEMBLE_CHECKPOINT_ASSET_NAME="${ENSEMBLE_CHECKPOINT_ASSET_NAME:-finetuned_latest.pt}"
ENSEMBLE_CHECKPOINT_ASSET="${ENSEMBLE_CHECKPOINT_ASSET_DIR}/${ENSEMBLE_CHECKPOINT_ASSET_NAME}"
ENSEMBLE_WEIGHTS="${ENSEMBLE_WEIGHTS:-}"
ENSEMBLE_MEMBER_BIASES="${ENSEMBLE_MEMBER_BIASES:-}"
ENSEMBLE_MEMBER_TEMPERATURES="${ENSEMBLE_MEMBER_TEMPERATURES:-}"
EXPECTED_METRIC_ROWS="${EXPECTED_METRIC_ROWS:-0}"
DATA_DIR="${DATA_DIR:-data/MEC_8nm/hdf5}"
RUNTIME_CONFIG="configs/runtime_${RUN_NAME}.yaml"

read -r -a SAMPLE_ARRAY <<< "${SAMPLES:-mec8nm_11.h5 mec8nm_12.h5}"
read -r -a CROP_SIZE_ARRAY <<< "${CROP_SIZE:-0 0 0}"
read -r -a CHUNK_SIZE_ARRAY <<< "${CHUNK_SIZE:-32 512 512}"
read -r -a HALO_ARRAY <<< "${HALO:-8 64 64}"
read -r -a STRIDE_ARRAY <<< "${STRIDE:-16 80 80}"
read -r -a SEAM_WIDTH_ARRAY <<< "${SEAM_WIDTH:-2 16 16}"
read -r -a BACKENDS_ARRAY <<< "${BACKENDS:-graph_cc}"
read -r -a THRESHOLDS_ARRAY <<< "${THRESHOLDS:-0.5}"
read -r -a Z_THRESHOLDS_ARRAY <<< "${Z_THRESHOLDS:-0.5}"
read -r -a XY_THRESHOLDS_ARRAY <<< "${XY_THRESHOLDS:-0.5}"
read -r -a CALIBRATION_BIASES_ARRAY <<< "${CALIBRATION_BIASES:--0.20 -0.45 -0.45}"
read -r -a CALIBRATION_TEMPERATURES_ARRAY <<< "${CALIBRATION_TEMPERATURES:-1.0}"
read -r -a AUTO_CALIBRATION_TARGETS_ARRAY <<< "${AUTO_CALIBRATION_TARGETS:-}"
read -r -a AUTO_CALIBRATION_QUANTILES_ARRAY <<< "${AUTO_CALIBRATION_QUANTILES:-}"
read -r -a REGIONS_ARRAY <<< "${REGIONS:-full seam nonseam}"
read -r -a SCORE_MODE_ARRAY <<< "${SCORE_MODE:-mean q25}"
read -r -a RAG_QUANTILE_ARRAY <<< "${RAG_QUANTILE:-0.25}"
INFERENCE_NORMALIZATION="${INFERENCE_NORMALIZATION:-volume}"

mkdir -p "${DATA_DIR}" configs outputs logs "${CHECKPOINT_ASSET_DIR}" "${ENSEMBLE_CHECKPOINT_ASSET_DIR}" bin

if [ ! -x bin/tosutil ]; then
  if [ -x /volume/med-train/users/dchen02/bin/tosutil ]; then
    cp /volume/med-train/users/dchen02/bin/tosutil bin/tosutil
  else
    echo "missing tosutil" >&2
    exit 2
  fi
fi

TOS_CONF=/tmp/mec8nm_blockwise_eval_tosutil.conf
: > "${TOS_CONF}"
bin/tosutil config \
  -e "${TOS_ENDPOINT:?missing TOS_ENDPOINT}" \
  -re "${TOS_REGION:?missing TOS_REGION}" \
  -i "${TOS_ACCESS_KEY_ID:?missing TOS_ACCESS_KEY_ID}" \
  -k "${TOS_SECRET_ACCESS_KEY:?missing TOS_SECRET_ACCESS_KEY}" \
  -conf="${TOS_CONF}" >/dev/null

if [ -n "${CHECKPOINT_TOS_URI}" ]; then
  echo "[mec8nm-eval] downloading checkpoint from TOS"
  bin/tosutil cp "${CHECKPOINT_TOS_URI}" "${CHECKPOINT_ASSET}" -conf="${TOS_CONF}"
fi
if [ -n "${ENSEMBLE_CHECKPOINT_TOS_URI}" ]; then
  echo "[mec8nm-eval] downloading ensemble checkpoint from TOS"
  bin/tosutil cp "${ENSEMBLE_CHECKPOINT_TOS_URI}" "${ENSEMBLE_CHECKPOINT_ASSET}" -conf="${TOS_CONF}"
fi

echo "[mec8nm-eval] installing optional offline wheels"
python -m pip install --user --no-index --find-links wheelhouse_dbmim pyyaml h5py scipy mahotas connected-components-3d || true
python -m pip install --user --no-index --find-links wheelhouse_superhuman_eval scikit-image tifffile || true
need_waterz=0
for backend in "${BACKENDS_ARRAY[@]}"; do
  if [ "${backend}" = "waterz" ]; then
    need_waterz=1
    break
  fi
done
if [ "${need_waterz}" -eq 1 ] && ! python - <<'PY'
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
if [ "${need_waterz}" -eq 1 ] && ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("waterz") else 1)
PY
then
  echo "waterz requested but installation failed" >&2
  exit 4
fi

echo "[mec8nm-eval] downloading selected MEC HDF5 samples from ${TOS_MEC_PREFIX}"
for sample in "${SAMPLE_ARRAY[@]}"; do
  bin/tosutil cp "${TOS_MEC_PREFIX}/hdf5/${sample}" "${DATA_DIR}/${sample}" -conf="${TOS_CONF}"
done
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_manifest.json" data/MEC_8nm/mec8nm_manifest.json -conf="${TOS_CONF}" || true
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_splits.json" data/MEC_8nm/mec8nm_splits.json -conf="${TOS_CONF}" || true

if [ ! -s "${CHECKPOINT_ASSET}" ]; then
  echo "missing checkpoint asset: ${CHECKPOINT_ASSET}" >&2
  exit 3
fi

ENSEMBLE_ENABLED=0
if [ -n "${ENSEMBLE_CHECKPOINT_TOS_URI}" ] || [ -n "${ENSEMBLE_WEIGHTS}" ] || [ -n "${ENSEMBLE_MEMBER_BIASES}" ] || [ -n "${ENSEMBLE_MEMBER_TEMPERATURES}" ]; then
  ENSEMBLE_ENABLED=1
fi
if [ "${ENSEMBLE_ENABLED}" -eq 1 ] && [ ! -s "${ENSEMBLE_CHECKPOINT_ASSET}" ]; then
  echo "missing ensemble checkpoint asset: ${ENSEMBLE_CHECKPOINT_ASSET}" >&2
  exit 3
fi

RUN_NAME="${RUN_NAME}" CONFIG_NAME="${CONFIG_NAME}" RUNTIME_CONFIG="${RUNTIME_CONFIG}" CHECKPOINT_ASSET="${CHECKPOINT_ASSET}" DATA_DIR="${DATA_DIR}" SAMPLES="${SAMPLES:-mec8nm_11.h5 mec8nm_12.h5}" python - <<'PY'
import json
import os
from pathlib import Path
import yaml

root = Path.cwd()
run_name = os.environ["RUN_NAME"]
config_name = os.environ["CONFIG_NAME"]
runtime_config = Path(os.environ["RUNTIME_CONFIG"])
checkpoint_asset = Path(os.environ["CHECKPOINT_ASSET"])
data_dir = Path(os.environ["DATA_DIR"])
samples = os.environ["SAMPLES"].split()
src = Path("external/dbMiM/configs") / f"{config_name}.yaml"
if not src.exists():
    src = Path("configs") / f"{config_name}.yaml"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
paths = [str((root / data_dir / sample).resolve()) for sample in samples]
cfg.setdefault("data", {})
cfg["data"]["image_paths"] = paths
cfg["data"]["label_paths"] = paths
cfg["pretrained"] = str((root / checkpoint_asset).resolve())
cfg["output_dir"] = str((root / "outputs" / run_name).resolve())
runtime_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(
    json.dumps(
        {
            "runtime_config": str(runtime_config),
            "samples": samples,
            "checkpoint": cfg["pretrained"],
            "output_dir": cfg["output_dir"],
        }
    ),
    flush=True,
)
PY

python - <<'PY'
import importlib.util
import json
import torch

mods = ["torch", "h5py", "yaml", "scipy", "mahotas", "skimage", "waterz", "cc3d"]
print(
    json.dumps(
        {
            "module_probe": {m: bool(importlib.util.find_spec(m)) for m in mods},
            "cuda": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
        }
    ),
    flush=True,
)
PY

EXTRA_ARGS=()
if [ "${ENSEMBLE_ENABLED}" -eq 1 ]; then
  read -r -a ENSEMBLE_WEIGHTS_ARRAY <<< "${ENSEMBLE_WEIGHTS}"
  read -r -a ENSEMBLE_MEMBER_BIASES_ARRAY <<< "${ENSEMBLE_MEMBER_BIASES}"
  read -r -a ENSEMBLE_MEMBER_TEMPERATURES_ARRAY <<< "${ENSEMBLE_MEMBER_TEMPERATURES}"
  if [ "${#ENSEMBLE_WEIGHTS_ARRAY[@]}" -ne 2 ]; then
    echo "ENSEMBLE_WEIGHTS must provide primary and secondary weights" >&2
    exit 2
  fi
  if [ "${#ENSEMBLE_MEMBER_BIASES_ARRAY[@]}" -ne 6 ]; then
    echo "ENSEMBLE_MEMBER_BIASES must provide z/y/x triplets for both members" >&2
    exit 2
  fi
  if [ "${#ENSEMBLE_MEMBER_TEMPERATURES_ARRAY[@]}" -ne 2 ]; then
    echo "ENSEMBLE_MEMBER_TEMPERATURES must provide one temperature for both members" >&2
    exit 2
  fi
  EXTRA_ARGS+=(
    --ensemble-checkpoints "${ENSEMBLE_CHECKPOINT_ASSET}"
    --ensemble-weights "${ENSEMBLE_WEIGHTS_ARRAY[@]}"
    --ensemble-member-biases "${ENSEMBLE_MEMBER_BIASES_ARRAY[@]}"
    --ensemble-member-temperatures "${ENSEMBLE_MEMBER_TEMPERATURES_ARRAY[@]}"
  )
fi
if [ "${SKIP_PRED_VARIANT:-1}" = "1" ]; then
  EXTRA_ARGS+=(--skip-pred-variant)
fi
if [ "${COMPARE_NO_HALO:-0}" = "1" ]; then
  EXTRA_ARGS+=(--compare-no-halo)
fi
if [ "${COMPUTE_RAG_STATS:-0}" = "1" ]; then
  EXTRA_ARGS+=(--compute-rag-stats)
fi
if [ "${RECORD_AFFINITY_STATS:-0}" = "1" ]; then
  EXTRA_ARGS+=(--record-affinity-stats)
fi
if [ "${#AUTO_CALIBRATION_TARGETS_ARRAY[@]}" -gt 0 ] || [ "${#AUTO_CALIBRATION_QUANTILES_ARRAY[@]}" -gt 0 ]; then
  if [ "${#AUTO_CALIBRATION_TARGETS_ARRAY[@]}" -eq 0 ] || [ "${#AUTO_CALIBRATION_QUANTILES_ARRAY[@]}" -eq 0 ]; then
    echo "AUTO_CALIBRATION_TARGETS and AUTO_CALIBRATION_QUANTILES must both be non-empty" >&2
    exit 2
  fi
  EXTRA_ARGS+=(
    --auto-calibration-targets "${AUTO_CALIBRATION_TARGETS_ARRAY[@]}"
    --auto-calibration-quantiles "${AUTO_CALIBRATION_QUANTILES_ARRAY[@]}"
  )
fi
if [ "${FAIL_ON_BACKEND_ERROR:-1}" = "1" ]; then
  EXTRA_ARGS+=(--fail-on-backend-error)
fi

echo "[mec8nm-eval] starting blockwise eval ${RUN_NAME}: ${SAMPLE_ARRAY[*]}"
eval_start="$(date +%s)"
set +e
python "${DBMIM}/scripts/evaluate_cremi_blockwise_scale.py" \
  --config "${RUNTIME_CONFIG}" \
  --checkpoint "${CHECKPOINT_ASSET}" \
  --data-dir "${DATA_DIR}" \
  --sample-names "${SAMPLE_ARRAY[@]}" \
  --output-dir "outputs/${RUN_NAME}" \
  --crop-size "${CROP_SIZE_ARRAY[@]}" \
  --chunk-size "${CHUNK_SIZE_ARRAY[@]}" \
  --halo "${HALO_ARRAY[@]}" \
  --stride "${STRIDE_ARRAY[@]}" \
  --seam-width "${SEAM_WIDTH_ARRAY[@]}" \
  --backends "${BACKENDS_ARRAY[@]}" \
  --thresholds "${THRESHOLDS_ARRAY[@]}" \
  --z-thresholds "${Z_THRESHOLDS_ARRAY[@]}" \
  --xy-thresholds "${XY_THRESHOLDS_ARRAY[@]}" \
  --seed-nms-radius "${SEED_NMS_RADIUS:-0}" \
  --calibration-biases "${CALIBRATION_BIASES_ARRAY[@]}" \
  --calibration-temperatures "${CALIBRATION_TEMPERATURES_ARRAY[@]}" \
  --min-boundary "${MIN_BOUNDARY:-4}" \
  --score-mode "${SCORE_MODE_ARRAY[@]}" \
  --rag-quantile "${RAG_QUANTILE_ARRAY[@]}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --metric-backend "${METRIC_BACKEND:-skimage}" \
  --regions "${REGIONS_ARRAY[@]}" \
  --ignore-label "${IGNORE_LABEL:-0}" \
  --cremi-boundary-ignore-distance-xy "${CREMI_BOUNDARY_IGNORE_DISTANCE_XY:-1}" \
  --cremi-boundary-ignore-distance-z "${CREMI_BOUNDARY_IGNORE_DISTANCE_Z:-0}" \
  --inference-normalization "${INFERENCE_NORMALIZATION}" \
  --device "${DEVICE:-cuda}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | sed -u "s/^/[blockwise-eval] /" | tee "logs/${RUN_NAME}.blockwise_eval.log"
eval_code="${PIPESTATUS[0]}"
set -e
eval_end="$(date +%s)"
echo "{\"event\":\"blockwise_eval_done\",\"code\":${eval_code},\"elapsed_sec\":$((eval_end-eval_start))}" | tee "logs/${RUN_NAME}.status.jsonl"

if [ "${eval_code}" -ne 0 ]; then
  bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/outputs_failed" -r -conf="${TOS_CONF}" || true
  bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_failed" -r -conf="${TOS_CONF}" || true
  exit "${eval_code}"
fi

if [ "${EXPECTED_METRIC_ROWS}" -gt 0 ]; then
  if ! python - "outputs/${RUN_NAME}/blockwise_scale_records.csv" "outputs/${RUN_NAME}/blockwise_failures.partial.json" "${EXPECTED_METRIC_ROWS}" "${SAMPLE_ARRAY[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

records_path = Path(sys.argv[1])
failures_path = Path(sys.argv[2])
expected_rows = int(sys.argv[3])
expected_samples = set(sys.argv[4:])
with records_path.open(newline="", encoding="utf-8") as handle:
    records = list(csv.DictReader(handle))
if len(records) != expected_rows:
    raise SystemExit(f"expected {expected_rows} metric rows, found {len(records)}")
if {record.get("sample", "") for record in records} != expected_samples:
    raise SystemExit("metric samples do not match the explicitly requested samples")
failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else []
if failures:
    raise SystemExit(f"backend failure ledger is non-empty: {len(failures)} entries")
print(json.dumps({"event": "metric_assertion_passed", "rows": len(records), "samples": sorted(expected_samples)}))
PY
  then
    bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/outputs_failed" -r -conf="${TOS_CONF}" || true
    bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_failed" -r -conf="${TOS_CONF}" || true
    exit 5
  fi
fi

echo "[mec8nm-eval] uploading outputs to ${TOS_OUTPUT_PREFIX}"
bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/blockwise_eval" -r -conf="${TOS_CONF}"
bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs" -r -conf="${TOS_CONF}" || true
echo "{\"event\":\"upload_done\",\"tos_output_prefix\":\"${TOS_OUTPUT_PREFIX}\"}" | tee -a "logs/${RUN_NAME}.status.jsonl"
