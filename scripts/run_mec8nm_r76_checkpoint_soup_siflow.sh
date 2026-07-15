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

readonly RUN_NAME="eval_mec8nm_r76_r58_r71_soup_val_crop"
readonly CONFIG_NAME="finetune_mec8nm_r58_seed1789_balanced_stage1"
readonly TOS_MEC_PREFIX="tos://agi-data/users/dchen02/dbmim/assets/mec8nm_20260711/MEC_8nm"
readonly TOS_OUTPUT_PREFIX="tos://agi-data/users/dchen02/dbmim/mec8nm_runs/${RUN_NAME}"
readonly R58_URI="tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r58_seed1789_balanced_stage1/train/finetune_mec8nm_r58_seed1789_balanced_stage1/finetuned_latest.pt"
readonly R71_URI="tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r71_seed1789_ema_stage1/train/finetune_mec8nm_r71_seed1789_ema_stage1/finetuned_latest.pt"
readonly R58_SHA256="4715c6679808eac5e073159202be1cca77307ab90202bd683f8ab97cacc55e09"
readonly R71_SHA256="9c82d7ac4d4196a19966451343fcc033b7ea279a428b5913a54d9f4075790a45"
readonly DATA_DIR="data/MEC_8nm/hdf5"
readonly SOURCE_DIR="assets/r76_sources"
readonly SOUP_DIR="/tmp/mec8nm_r76_checkpoint_soup"
readonly RUNTIME_CONFIG="configs/runtime_${RUN_NAME}.yaml"
readonly EXPECTED_ROWS="80"
export EXPECTED_ROWS
SAMPLES=(mec8nm_08.h5 mec8nm_10.h5)
ALPHAS=(0.010 0.025 0.050 0.100 0.200)
TAGS=(0p010 0p025 0p050 0p100 0p200)

mkdir -p "${DATA_DIR}" "${SOURCE_DIR}" "${SOUP_DIR}" "outputs/${RUN_NAME}" configs logs bin
if [ ! -x bin/tosutil ]; then
  echo "missing bundled tosutil" >&2
  exit 2
fi

TOS_CONF=/tmp/mec8nm_r76_tosutil.conf
: > "${TOS_CONF}"
bin/tosutil config \
  -e "${TOS_ENDPOINT:?missing TOS_ENDPOINT}" \
  -re "${TOS_REGION:?missing TOS_REGION}" \
  -i "${TOS_ACCESS_KEY_ID:?missing TOS_ACCESS_KEY_ID}" \
  -k "${TOS_SECRET_ACCESS_KEY:?missing TOS_SECRET_ACCESS_KEY}" \
  -conf="${TOS_CONF}" >/dev/null

echo "[r76] installing offline evaluation dependencies"
python -m pip install --user --no-index --find-links wheelhouse_dbmim pyyaml h5py scipy mahotas connected-components-3d || true
python -m pip install --user --no-index --find-links wheelhouse_superhuman_eval scikit-image tifffile || true
if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("waterz") else 1)
PY
then
  export CPLUS_INCLUDE_PATH="${DBMIM}/third_party/boost_1_84_0:${CPLUS_INCLUDE_PATH:-}"
  export BOOST_INCLUDEDIR="${DBMIM}/third_party/boost_1_84_0"
  python -m pip install --user --no-build-isolation --no-index \
    --find-links wheelhouse_superhuman_eval third_party/waterz
fi
python - <<'PY'
import importlib.util, json, torch
mods = ["torch", "h5py", "yaml", "scipy", "mahotas", "skimage", "waterz", "cc3d"]
probe = {name: bool(importlib.util.find_spec(name)) for name in mods}
print(json.dumps({"module_probe": probe, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()}), flush=True)
if not all(probe.values()) or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("R76 requires all evaluation modules and exactly eight visible GPUs")
PY

echo "[r76] downloading validation blocks only: ${SAMPLES[*]}"
for sample in "${SAMPLES[@]}"; do
  bin/tosutil cp "${TOS_MEC_PREFIX}/hdf5/${sample}" "${DATA_DIR}/${sample}" -conf="${TOS_CONF}"
done
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_manifest.json" data/MEC_8nm/mec8nm_manifest.json -conf="${TOS_CONF}" || true
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_splits.json" data/MEC_8nm/mec8nm_splits.json -conf="${TOS_CONF}" || true

echo "[r76] downloading audited R58 and R71 endpoints"
bin/tosutil cp "${R58_URI}" "${SOURCE_DIR}/r58.pt" -conf="${TOS_CONF}"
bin/tosutil cp "${R71_URI}" "${SOURCE_DIR}/r71_ema.pt" -conf="${TOS_CONF}"
printf '%s  %s\n' "${R58_SHA256}" "${SOURCE_DIR}/r58.pt" | sha256sum -c -
printf '%s  %s\n' "${R71_SHA256}" "${SOURCE_DIR}/r71_ema.pt" | sha256sum -c -

python "${DBMIM}/scripts/build_mec8nm_checkpoint_soup.py" \
  --anchor "${SOURCE_DIR}/r58.pt" \
  --ema "${SOURCE_DIR}/r71_ema.pt" \
  --output-dir "${SOUP_DIR}" \
  --alphas "${ALPHAS[@]}" \
  --anchor-sha256 "${R58_SHA256}" \
  --ema-sha256 "${R71_SHA256}"
cp "${SOUP_DIR}/checkpoint_soup_manifest.json" "outputs/${RUN_NAME}/"

export RUN_NAME CONFIG_NAME RUNTIME_CONFIG DATA_DIR
python - <<'PY'
import json, os
from pathlib import Path
import yaml

root = Path.cwd()
config_name = os.environ["CONFIG_NAME"]
src = Path("external/dbMiM/configs") / f"{config_name}.yaml"
if not src.exists():
    src = Path("configs") / f"{config_name}.yaml"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
samples = ["mec8nm_08.h5", "mec8nm_10.h5"]
paths = [str((root / os.environ["DATA_DIR"] / sample).resolve()) for sample in samples]
cfg["data"]["image_paths"] = paths
cfg["data"]["label_paths"] = paths
cfg["output_dir"] = str((root / "outputs" / os.environ["RUN_NAME"]).resolve())
Path(os.environ["RUNTIME_CONFIG"]).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(json.dumps({"runtime_config": os.environ["RUNTIME_CONFIG"], "samples": samples}), flush=True)
PY

for index in "${!ALPHAS[@]}"; do
  alpha="${ALPHAS[$index]}"
  tag="${TAGS[$index]}"
  checkpoint="${SOUP_DIR}/r58_r71_soup_alpha_${tag}.pt"
  output="outputs/${RUN_NAME}/alpha_${tag}"
  log="logs/${RUN_NAME}.alpha_${tag}.log"
  echo "[r76] evaluating alpha=${alpha} checkpoint=${checkpoint}"
  start="$(date +%s)"
  set +e
  python "${DBMIM}/scripts/evaluate_cremi_segmentation.py" \
    --config "${RUNTIME_CONFIG}" \
    --checkpoint "${checkpoint}" \
    --data-dir "${DATA_DIR}" \
    --sample-names "${SAMPLES[@]}" \
    --output-dir "${output}" \
    --crop-size 64 512 512 \
    --stride 16 80 80 \
    --backends waterz \
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
    2>&1 | sed -u "s/^/[r76-alpha-${tag}] /" | tee "${log}"
  code="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  echo "{\"event\":\"candidate_eval_done\",\"alpha\":${alpha},\"code\":${code},\"elapsed_sec\":$((end-start))}" | tee -a "logs/${RUN_NAME}.status.jsonl"
  if [ "${code}" -ne 0 ]; then
    bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/outputs_failed" -r -conf="${TOS_CONF}" || true
    bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_failed" -r -conf="${TOS_CONF}" || true
    exit "${code}"
  fi

  METRICS="${output}/cremi_segmentation_metrics.csv" SUMMARY="${output}/cremi_segmentation_summary.json" ALPHA="${alpha}" python - <<'PY'
import csv, json, math, os
from pathlib import Path

metrics = Path(os.environ["METRICS"])
summary_path = Path(os.environ["SUMMARY"])
with metrics.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
samples = {row.get("sample") for row in rows}
backends = {row.get("backend") for row in rows}
finite = all(math.isfinite(float(row[field])) for row in rows for field in ("adapted_rand_error", "voi_sum"))
if (
    len(rows) != int(os.environ["EXPECTED_ROWS"])
    or samples != {"mec8nm_08.h5", "mec8nm_10.h5"}
    or backends != {"waterz"}
    or not finite
    or summary.get("failures") != []
):
    raise SystemExit(
        f"invalid R76 alpha={os.environ['ALPHA']} rows={len(rows)} samples={samples} "
        f"backends={backends} finite={finite} failures={summary.get('failures')}"
    )
print(json.dumps({"event": "metric_assertion_passed", "alpha": float(os.environ["ALPHA"]), "rows": len(rows), "samples": sorted(samples), "backends": sorted(backends)}), flush=True)
PY
done

echo "{\"event\":\"r76_candidates_complete\",\"candidates\":${#ALPHAS[@]},\"rows_per_candidate\":${EXPECTED_ROWS}}" | tee -a "logs/${RUN_NAME}.status.jsonl"
bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/crop_evals" -r -conf="${TOS_CONF}"
bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs" -r -conf="${TOS_CONF}" || true
echo "{\"event\":\"upload_done\",\"tos_output_prefix\":\"${TOS_OUTPUT_PREFIX}\"}" | tee -a "logs/${RUN_NAME}.status.jsonl"
