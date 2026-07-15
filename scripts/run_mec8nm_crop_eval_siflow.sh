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

RUN_NAME="${RUN_NAME:-eval_mec8nm_r63_seed1789_photodomain_crop_postprocessfix}"
CONFIG_NAME="${CONFIG_NAME:-finetune_mec8nm_r63_seed1789_photodomain_stage1}"
TOS_MEC_PREFIX="${TOS_MEC_PREFIX:-tos://agi-data/users/dchen02/dbmim/assets/mec8nm_20260711/MEC_8nm}"
TOS_OUTPUT_PREFIX="${TOS_OUTPUT_PREFIX:-tos://agi-data/users/dchen02/dbmim/mec8nm_runs/${RUN_NAME}}"
CHECKPOINT_TOS_URI="${CHECKPOINT_TOS_URI:-tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r63_seed1789_photodomain_stage1/train/finetune_mec8nm_r63_seed1789_photodomain_stage1/finetuned_latest.pt}"
CHECKPOINT_ASSET_DIR="${CHECKPOINT_ASSET_DIR:-assets/r63}"
CHECKPOINT_ASSET_NAME="${CHECKPOINT_ASSET_NAME:-finetuned_latest.pt}"
CHECKPOINT_ASSET="${CHECKPOINT_ASSET_DIR}/${CHECKPOINT_ASSET_NAME}"
RUNTIME_CONFIG="configs/runtime_${RUN_NAME}.yaml"
DATA_DIR="data/MEC_8nm/hdf5"
SAMPLES=(mec8nm_08.h5 mec8nm_10.h5)
read -r -a EVAL_THRESHOLDS_ARRAY <<< "${EVAL_THRESHOLDS:-0.16 0.18 0.20 0.22 0.24 0.30 0.40 0.50}"
read -r -a EVAL_CALIBRATION_BIASES_ARRAY <<< "${EVAL_CALIBRATION_BIASES:-0 0 0 -0.20 -0.45 -0.45}"
read -r -a EVAL_CALIBRATION_TEMPERATURES_ARRAY <<< "${EVAL_CALIBRATION_TEMPERATURES:-1.0 1.15}"

if [ "${#EVAL_THRESHOLDS_ARRAY[@]}" -eq 0 ] || [ "${#EVAL_CALIBRATION_TEMPERATURES_ARRAY[@]}" -eq 0 ]; then
  echo "evaluation thresholds and calibration temperatures must be nonempty" >&2
  exit 2
fi
if [ "${#EVAL_CALIBRATION_BIASES_ARRAY[@]}" -eq 0 ] || [ $(( ${#EVAL_CALIBRATION_BIASES_ARRAY[@]} % 3 )) -ne 0 ]; then
  echo "evaluation calibration biases must contain z/y/x triplets" >&2
  exit 2
fi
CALIBRATION_VARIANT_COUNT=$(( (${#EVAL_CALIBRATION_BIASES_ARRAY[@]} / 3) * ${#EVAL_CALIBRATION_TEMPERATURES_ARRAY[@]} + 1 ))
EXPECTED_METRIC_ROWS=$(( CALIBRATION_VARIANT_COUNT * ${#EVAL_THRESHOLDS_ARRAY[@]} * ${#SAMPLES[@]} ))

mkdir -p "${DATA_DIR}" configs outputs logs "${CHECKPOINT_ASSET_DIR}" bin

if [ ! -x bin/tosutil ]; then
  if [ -x /volume/med-train/users/dchen02/bin/tosutil ]; then
    cp /volume/med-train/users/dchen02/bin/tosutil bin/tosutil
  else
    echo "missing tosutil" >&2
    exit 2
  fi
fi

TOS_CONF=/tmp/mec8nm_crop_eval_tosutil.conf
: > "${TOS_CONF}"
bin/tosutil config \
  -e "${TOS_ENDPOINT:?missing TOS_ENDPOINT}" \
  -re "${TOS_REGION:?missing TOS_REGION}" \
  -i "${TOS_ACCESS_KEY_ID:?missing TOS_ACCESS_KEY_ID}" \
  -k "${TOS_SECRET_ACCESS_KEY:?missing TOS_SECRET_ACCESS_KEY}" \
  -conf="${TOS_CONF}" >/dev/null

echo "[mec8nm-crop-eval] installing optional offline wheels"
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
  echo "waterz installation failed" >&2
  exit 4
fi

echo "[mec8nm-crop-eval] downloading validation HDF5 only: ${SAMPLES[*]}"
for sample in "${SAMPLES[@]}"; do
  bin/tosutil cp "${TOS_MEC_PREFIX}/hdf5/${sample}" "${DATA_DIR}/${sample}" -conf="${TOS_CONF}"
done
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_manifest.json" data/MEC_8nm/mec8nm_manifest.json -conf="${TOS_CONF}" || true
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_splits.json" data/MEC_8nm/mec8nm_splits.json -conf="${TOS_CONF}" || true

if [ ! -s "${CHECKPOINT_ASSET}" ]; then
  echo "[mec8nm-crop-eval] downloading checkpoint from TOS"
  bin/tosutil cp "${CHECKPOINT_TOS_URI}" "${CHECKPOINT_ASSET}" -conf="${TOS_CONF}"
fi

RUN_NAME="${RUN_NAME}" CONFIG_NAME="${CONFIG_NAME}" RUNTIME_CONFIG="${RUNTIME_CONFIG}" CHECKPOINT_ASSET="${CHECKPOINT_ASSET}" DATA_DIR="${DATA_DIR}" python - <<'PY'
import json
import os
from pathlib import Path
import yaml

root = Path.cwd()
config_name = os.environ["CONFIG_NAME"]
runtime_config = Path(os.environ["RUNTIME_CONFIG"])
checkpoint_asset = Path(os.environ["CHECKPOINT_ASSET"])
data_dir = Path(os.environ["DATA_DIR"])
src = Path("external/dbMiM/configs") / f"{config_name}.yaml"
if not src.exists():
    src = Path("configs") / f"{config_name}.yaml"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
samples = ["mec8nm_08.h5", "mec8nm_10.h5"]
paths = [str((root / data_dir / sample).resolve()) for sample in samples]
cfg.setdefault("data", {})
cfg["data"]["image_paths"] = paths
cfg["data"]["label_paths"] = paths
cfg["pretrained"] = str((root / checkpoint_asset).resolve())
cfg["output_dir"] = str((root / "outputs" / os.environ["RUN_NAME"]).resolve())
runtime_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(json.dumps({"runtime_config": str(runtime_config), "samples": samples, "checkpoint": cfg["pretrained"]}), flush=True)
PY

python - <<'PY'
import importlib.util
import json
import torch

mods = ["torch", "h5py", "yaml", "scipy", "mahotas", "skimage", "waterz", "cc3d"]
print(json.dumps({"module_probe": {m: bool(importlib.util.find_spec(m)) for m in mods}, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()}), flush=True)
PY

echo "[mec8nm-crop-eval] evaluating validation crop only: ${SAMPLES[*]}"
eval_start="$(date +%s)"
set +e
python "${DBMIM}/scripts/evaluate_cremi_segmentation.py" \
  --config "${RUNTIME_CONFIG}" \
  --checkpoint "${CHECKPOINT_ASSET}" \
  --data-dir "${DATA_DIR}" \
  --sample-names "${SAMPLES[@]}" \
  --output-dir "outputs/${RUN_NAME}" \
  --crop-size 64 512 512 \
  --stride 16 80 80 \
  --backends waterz \
  --thresholds "${EVAL_THRESHOLDS_ARRAY[@]}" \
  --calibration-biases "${EVAL_CALIBRATION_BIASES_ARRAY[@]}" \
  --calibration-temperatures "${EVAL_CALIBRATION_TEMPERATURES_ARRAY[@]}" \
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
  2>&1 | sed -u "s/^/[crop-eval] /" | tee "logs/${RUN_NAME}.log"
eval_code="${PIPESTATUS[0]}"
set -e
eval_end="$(date +%s)"
echo "{\"event\":\"crop_eval_done\",\"code\":${eval_code},\"elapsed_sec\":$((eval_end-eval_start))}" | tee "logs/${RUN_NAME}.status.jsonl"
if [ "${eval_code}" -ne 0 ]; then
  bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/outputs_failed" -r -conf="${TOS_CONF}" || true
  bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_failed" -r -conf="${TOS_CONF}" || true
  exit "${eval_code}"
fi

METRICS_PATH="outputs/${RUN_NAME}/cremi_segmentation_metrics.csv"
METRICS_PATH="${METRICS_PATH}" EXPECTED_METRIC_ROWS="${EXPECTED_METRIC_ROWS}" python - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ["METRICS_PATH"])
expected_rows = int(os.environ["EXPECTED_METRIC_ROWS"])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"missing metrics CSV: {path}")
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
samples = {row.get("sample") for row in rows}
backends = {row.get("backend") for row in rows}
if len(rows) != expected_rows or samples != {"mec8nm_08.h5", "mec8nm_10.h5"} or backends != {"waterz"}:
    raise SystemExit(
        f"invalid validation metrics rows={len(rows)} expected_rows={expected_rows} "
        f"samples={sorted(samples)} backends={sorted(backends)}"
    )
print({"validated_rows": len(rows), "samples": sorted(samples), "backends": sorted(backends)}, flush=True)
PY

echo "[mec8nm-crop-eval] uploading outputs to ${TOS_OUTPUT_PREFIX}"
bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/crop_eval" -r -conf="${TOS_CONF}"
bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs" -r -conf="${TOS_CONF}" || true
echo "{\"event\":\"upload_done\",\"tos_output_prefix\":\"${TOS_OUTPUT_PREFIX}\"}" | tee -a "logs/${RUN_NAME}.status.jsonl"
