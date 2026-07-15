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

RUN_NAME="${RUN_NAME:-infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_fullblock}"
CONFIG_NAME="${CONFIG_NAME:-finetune_mec8nm_r57_seed777_stage1}"
TOS_MEC_PREFIX="${TOS_MEC_PREFIX:-tos://agi-data/users/dchen02/dbmim/assets/mec8nm_20260711/MEC_8nm}"
TOS_OUTPUT_PREFIX="${TOS_OUTPUT_PREFIX:-tos://agi-data/users/dchen02/dbmim/mec8nm_runs/${RUN_NAME}}"
CHECKPOINT_ASSET_DIR="${CHECKPOINT_ASSET_DIR:-assets/r57_mec_stage1}"
CHECKPOINT_ASSET_NAME="${CHECKPOINT_ASSET_NAME:-finetuned_latest.pt}"
CHECKPOINT_ASSET="${CHECKPOINT_ASSET_DIR}/${CHECKPOINT_ASSET_NAME}"
DATA_DIR="${DATA_DIR:-data/MEC_8nm/hdf5}"
RUNTIME_CONFIG="configs/runtime_${RUN_NAME}.yaml"

read -r -a SAMPLE_ARRAY <<< "${SAMPLES:-mec8nm_12.h5}"
read -r -a CROP_START_ARRAY <<< "${CROP_START:-0 0 0}"
read -r -a CROP_SIZE_ARRAY <<< "${CROP_SIZE:-0 0 0}"
read -r -a CHUNK_SIZE_ARRAY <<< "${CHUNK_SIZE:-32 512 512}"
read -r -a HALO_ARRAY <<< "${HALO:-8 64 64}"
read -r -a STRIDE_ARRAY <<< "${STRIDE:-16 80 80}"
read -r -a CHUNK_SEAM_WIDTH_ARRAY <<< "${CHUNK_SEAM_WIDTH:-2 16 16}"
read -r -a CALIBRATION_BIASES_ARRAY <<< "${CALIBRATION_BIASES:--0.20 -0.45 -0.45}"
read -r -a CALIBRATION_TEMPERATURES_ARRAY <<< "${CALIBRATION_TEMPERATURES:-1.20}"
read -r -a REGIONS_ARRAY <<< "${REGIONS:-full slab_seam slab_nonseam chunk_seam chunk_nonseam}"

mkdir -p "${DATA_DIR}" configs outputs logs "${CHECKPOINT_ASSET_DIR}" bin references

if [ ! -x bin/tosutil ]; then
  if [ -x /volume/med-train/users/dchen02/bin/tosutil ]; then
    cp /volume/med-train/users/dchen02/bin/tosutil bin/tosutil
  else
    echo "missing tosutil" >&2
    exit 2
  fi
fi

TOS_CONF=/tmp/mec8nm_slabwise_tosutil.conf
: > "${TOS_CONF}"
bin/tosutil config \
  -e "${TOS_ENDPOINT:?missing TOS_ENDPOINT}" \
  -re "${TOS_REGION:?missing TOS_REGION}" \
  -i "${TOS_ACCESS_KEY_ID:?missing TOS_ACCESS_KEY_ID}" \
  -k "${TOS_SECRET_ACCESS_KEY:?missing TOS_SECRET_ACCESS_KEY}" \
  -conf="${TOS_CONF}" >/dev/null

echo "[mec8nm-slabwise] installing optional offline wheels"
python -m pip install --user --no-index --find-links wheelhouse_dbmim pyyaml h5py scipy mahotas connected-components-3d || true
python -m pip install --user --no-index --find-links wheelhouse_superhuman_eval scikit-image tifffile || true
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
  if [ -n "${WATERZ_SRC}" ]; then
    python -m pip install --user --no-build-isolation --no-index --find-links wheelhouse_superhuman_eval "${WATERZ_SRC}" || true
  fi
fi

echo "[mec8nm-slabwise] downloading selected MEC HDF5 samples from ${TOS_MEC_PREFIX}"
for sample in "${SAMPLE_ARRAY[@]}"; do
  bin/tosutil cp "${TOS_MEC_PREFIX}/hdf5/${sample}" "${DATA_DIR}/${sample}" -conf="${TOS_CONF}"
done
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_manifest.json" data/MEC_8nm/mec8nm_manifest.json -conf="${TOS_CONF}" || true
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_splits.json" data/MEC_8nm/mec8nm_splits.json -conf="${TOS_CONF}" || true

REFERENCE_ARGS=()
if [ -n "${REFERENCE_SEG_TOS_URI:-}" ]; then
  ref_name="$(basename "${REFERENCE_SEG_TOS_URI}")"
  ref_path="references/${ref_name}"
  echo "[mec8nm-slabwise] downloading reference segmentation ${REFERENCE_SEG_TOS_URI}"
  bin/tosutil cp "${REFERENCE_SEG_TOS_URI}" "${ref_path}" -conf="${TOS_CONF}"
  REFERENCE_ARGS+=(--reference-seg-h5 "${ref_path}" --reference-dataset "${REFERENCE_DATASET:-segmentation}")
fi

if [ ! -s "${CHECKPOINT_ASSET}" ]; then
  echo "missing checkpoint asset: ${CHECKPOINT_ASSET}" >&2
  exit 3
fi

RUN_NAME="${RUN_NAME}" CONFIG_NAME="${CONFIG_NAME}" RUNTIME_CONFIG="${RUNTIME_CONFIG}" CHECKPOINT_ASSET="${CHECKPOINT_ASSET}" DATA_DIR="${DATA_DIR}" SAMPLES="${SAMPLES:-mec8nm_12.h5}" python - <<'PY'
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
if [ "${EVALUATE:-1}" = "1" ]; then
  EXTRA_ARGS+=(--evaluate)
fi
if [ "${SAVE_SLABS:-1}" = "1" ]; then
  EXTRA_ARGS+=(--save-slabs)
fi
if [ "${FAIL_ON_SAMPLE_ERROR:-1}" = "1" ]; then
  EXTRA_ARGS+=(--fail-on-sample-error)
fi

echo "[mec8nm-slabwise] starting slabwise inference/stitch ${RUN_NAME}: ${SAMPLE_ARRAY[*]}"
infer_start="$(date +%s)"
python "${DBMIM}/scripts/infer_mec8nm_slabwise_stitch.py" \
  --config "${RUNTIME_CONFIG}" \
  --checkpoint "${CHECKPOINT_ASSET}" \
  --data-dir "${DATA_DIR}" \
  --sample-names "${SAMPLE_ARRAY[@]}" \
  --output-dir "outputs/${RUN_NAME}" \
  --crop-start "${CROP_START_ARRAY[@]}" \
  --crop-size "${CROP_SIZE_ARRAY[@]}" \
  --slab-axis "${SLAB_AXIS:-0}" \
  --slab-size "${SLAB_SIZE:-48}" \
  --slab-overlap "${SLAB_OVERLAP:-12}" \
  --slab-seam-width "${SLAB_SEAM_WIDTH:-2}" \
  --chunk-size "${CHUNK_SIZE_ARRAY[@]}" \
  --halo "${HALO_ARRAY[@]}" \
  --stride "${STRIDE_ARRAY[@]}" \
  --chunk-seam-width "${CHUNK_SEAM_WIDTH_ARRAY[@]}" \
  --backend "${BACKEND:-waterz}" \
  --threshold "${THRESHOLD:-0.36}" \
  --z-threshold "${Z_THRESHOLD:-0.5}" \
  --xy-threshold "${XY_THRESHOLD:-0.5}" \
  --seed-method "${SEED_METHOD:-maxima_distance}" \
  --seed-distance "${SEED_DISTANCE:-10}" \
  --boundary-threshold "${BOUNDARY_THRESHOLD:-0.5}" \
  --min-boundary "${MIN_BOUNDARY:-4}" \
  --score-mode "${SCORE_MODE:-q25}" \
  --rag-quantile "${RAG_QUANTILE:-0.25}" \
  --waterz-scoring "${WATERZ_SCORING:-hist_quantile}" \
  --calibration-biases "${CALIBRATION_BIASES_ARRAY[@]}" \
  --calibration-temperatures "${CALIBRATION_TEMPERATURES_ARRAY[@]}" \
  --min-contact-voxels "${MIN_CONTACT_VOXELS:-1}" \
  --min-overlap-ratio "${MIN_OVERLAP_RATIO:-0.5}" \
  --contact-chunk-voxels "${CONTACT_CHUNK_VOXELS:-5000000}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --metric-backend "${METRIC_BACKEND:-skimage}" \
  --regions "${REGIONS_ARRAY[@]}" \
  --ignore-label "${IGNORE_LABEL:-0}" \
  --cremi-boundary-ignore-distance-xy "${CREMI_BOUNDARY_IGNORE_DISTANCE_XY:-1}" \
  --cremi-boundary-ignore-distance-z "${CREMI_BOUNDARY_IGNORE_DISTANCE_Z:-0}" \
  --compression "${COMPRESSION:-gzip}" \
  --compression-level "${COMPRESSION_LEVEL:-4}" \
  --device "${DEVICE:-cuda}" \
  "${REFERENCE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | sed -u "s/^/[slabwise] /" | tee "logs/${RUN_NAME}.slabwise.log"
infer_code="${PIPESTATUS[0]}"
infer_end="$(date +%s)"
echo "{\"event\":\"slabwise_done\",\"code\":${infer_code},\"elapsed_sec\":$((infer_end-infer_start))}" | tee "logs/${RUN_NAME}.status.jsonl"

if [ "${infer_code}" -ne 0 ]; then
  bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/outputs_failed" -r -conf="${TOS_CONF}" || true
  bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_failed" -r -conf="${TOS_CONF}" || true
  exit "${infer_code}"
fi

echo "[mec8nm-slabwise] uploading outputs to ${TOS_OUTPUT_PREFIX}"
bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/slabwise_export" -r -conf="${TOS_CONF}"
bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs" -r -conf="${TOS_CONF}" || true
echo "{\"event\":\"upload_done\",\"tos_output_prefix\":\"${TOS_OUTPUT_PREFIX}\"}" | tee -a "logs/${RUN_NAME}.status.jsonl"
